import json
import math
import traceback
import numpy as np
from simworld.agent.base_agent import BaseAgent
from utils.vector import Vector
from utils.generate_depth_map import generate_depth_from_img
from utils.prompt_utils import WAYPOINT_GENERATION_PROMPT, WAYPOINT_SYSTEM_PROMPT
from utils.pixel_utils import random_waypoint_generator
# from simworld.traffic.base.traffic_signal import TrafficSignalState
# from agent.action_space import Action, ActionSpace-
from PIL import Image
import io

class NavAgent(BaseAgent):
    _id_counter = 0
    _camera_id_counter = 1

    def __init__(self, position, direction, communicator, nav_llm, destination, config):
        super().__init__(position, direction)
        self.id = NavAgent._id_counter
        self.camera_id = NavAgent._camera_id_counter
        NavAgent._id_counter += 1
        NavAgent._camera_id_counter += 1

        self.config = config
        self.communicator = communicator
        self.destination = destination
        self.nav_llm = nav_llm
        self.history = []

    def run(self, exit_event):
        print(f"Agent {self.id} is running")
        try:
            self.navigate(exit_event)
        except Exception as e :
            print(f"Error in agent {self.id}: {e}")
            print(traceback.format_exc())

    def _parse_waypoints(self, response):
        try:
            waypoints = {}
            coordinates = json.loads(response)
            for i, coord in enumerate(coordinates):
                label = chr(65 + i)  # 65 is ASCII for 'A'
                if len(coord) >= 3:
                    waypoints[label] = (coord[0], coord[1])
                else:
                    print(f"Skipping invalid coordinate set: {coord}")
            return waypoints
        except Exception as e:
            print(f"Error parsing waypoints: {e}")
            print(f"Response received: {response}")
            return {}
    
    def _pixel_to_world_coords(self, pixel_coords, cam_info, depth_image):
        try:
            x, y = pixel_coords
            depth = depth_image[int(y), int(x)]
            
            # Use camera parameters to convert to world coordinates
            fov = cam_info.get('fov', 90)
            aspect_ratio = cam_info.get('aspect_ratio', 1.0)
            
            # Calculate world coordinates using depth and camera parameters
            world_x = x * depth / aspect_ratio
            world_z = y * depth
            world_y = depth
            
            return Vector(world_x, world_y, world_z)
        except Exception as e:
            print(f"Error converting coordinates: {e}")
            return None
    
    def navigate(self, exit_event, generate_waypoint_zeroshot=True):
        print(f"Agent {self.id} is navigating to destination {self.destination}, current position: {self.position}")

        while (exit_event is None or not exit_event.is_set()):
            self.history.append(self.position) ## Adding the agent history
            print(self.history)
            print(hasattr(self.communicator, "get_camera_observation"))
            rgb_image = self.communicator.get_camera_observation(self.camera_id, 'lit')
            try:
                depth_image = self.communicator.get_camera_observation(self.camera_id, 'depth')
            except Exception e:
                print(f"Error in getting depth map for agent {self.id} with camera {self.camera_id}: {e}")
                depth_image = self.communicator.generate_depth_model(rgb_image)
            
            try:
                segmentation_map = self.communicator.get_camera_observation(self.camera_id, 'depth')
            except Exception e:
                print(f"Error in getting depth map for agent {self.id} with camera {self.camera_id}: {e}")
                segmentation_map = self.communicator.generate_depth_model(rgb_image)
            
            cam_info = self.communicator.get_camera_information(self.camera_id, rgb_image)

            current_yaw_rad = math.radians(self.yaw)

            # Genarting naviagtable waypoints using rgb, segmentation and depth map
            if generate_waypoint_zeroshot:
                response = self.nav_llm.generate_waypoints_openai(
                    image = rgb_image,
                    depth_map = depth_image,
                    seg_mask = segmentation_map,
                    system_prompt = WAYPOINT_SYSTEM_PROMPT,
                    waypoint_prompt = WAYPOINT_GENERATION_PROMPT)

                print("waypoint repsonse zeroshot", response)
            else:
                response = random_waypoint_generator(
                    segmentation_mask = segmentation_map,
                    depth_map = depth_image, 
                    agent_position = self.position
                )
                print("waypoint repsonse random generation", response)

            
            #  Devanshi's functionality must go here

            waypoints = self._parse_waypoints(response)
        
            if not waypoints:
                print("No valid waypoints received")
                continue
                
            # Select best waypoint
            selected_waypoint = self.nav_llm.select_best_waypoint(
                image=rgb_image,
                waypoints=waypoints,
                current_pos=self.position,
                destination=self.destination,
                history=self.history
            )
            
            if not selected_waypoint or selected_waypoint not in waypoints:
                print("Invalid waypoint selection")
                continue
                
            # Convert selected waypoint to world coordinates
            world_pos = self._pixel_to_world_coords(
                waypoints[selected_waypoint],
                cam_info,
                depth_image
            )
            
            if world_pos is None:
                print("Failed to convert waypoint to world coordinates")
                continue
                
            # Move agent towards selected waypoint using movement controller
            reached = self.movement_controller.move_to_waypoint(world_pos)
            
            if reached:
                print(f"Agent {self.id} reached waypoint")
            
            # Check if destination reached
            if Vector.distance(self.position, self.destination) < self.config.get('arrival_threshold', 1.0):
                print(f"Agent {self.id} reached destination!")
                break

