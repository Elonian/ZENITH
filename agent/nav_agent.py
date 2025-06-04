
import json
import math
import traceback
import numpy as np
import matplotlib.pyplot as plt
from simworld.agent.base_agent import BaseAgent
from utils.vector import Vector
from utils.generate_segment import generate_segmentation_mask
from utils.generate_depth_map import generate_depth_from_img
from utils.prompt_utils import WAYPOINT_GENERATION_PROMPT, WAYPOINT_SYSTEM_PROMPT, WAYPOINT_SELECTION_PROMPT
from utils.pixel_utils import random_waypoint_generator, visualize_waypoints_on_image, pixel_to_world
from agent.nav_move import navigate_to_target, navigate_to_target_with_heading
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

    def _parse_waypoints(self, coordinates):
        try:
            waypoints = {}
            for i, coord in enumerate(coordinates):
                label = chr(65 + i)  # 65 is ASCII for 'A'
                if len(coord) >= 2:
                    waypoints[label] = (coord[0], coord[1])
                else:
                    print(f"Skipping invalid coordinate set: {coord}")
            return waypoints
        except Exception as e:
            print(f"Error parsing waypoints: {e}")
            print(f"Response received: {coordinates}")
            return {}
            
    def extract_waypoint_label(self, response):
        if "**" in response:
            return response.split("**")[1]  # Assuming the format "The best waypoint to choose is **G**."
        return response.strip()

    def basic_refinement(self, waypoints, height, width):
        """Waypoints are not in permissible limit of image size, remove them"""
        refined_waypoints = []
        waypoints = json.loads(waypoints)
        for wp in waypoints["waypoints"]:
            x, y = wp['x'], wp['y']
            if 0 <= x < width and 0 <= y < height:
                refined_waypoints.append(wp)
        return json.dumps({"waypoints": refined_waypoints})
    
    def random_waypoint_world_coord_selector(self, world_coords):
        """Selects a random waypoint from the list of world coordinates."""
        """ woorld_coords is of the form [[x1, y1, z1], [x2, y2, z2], ...] select one of them randomly"""
        if len(world_coords) == 0:
            print("No valid waypoints available")
            return None
        selected_index = np.random.randint(0, len(world_coords))
        selected_waypoint = world_coords[selected_index]
        print(f"Selected random waypoint: {selected_waypoint}")
        return selected_waypoint        

    
    def navigate(self, exit_event, generate_waypoint_zeroshot=True):
        print(f"Agent {self.id} is navigating to destination {self.destination}, current position: {self.position}")

        # while not self.agent_reached_destination() and (exit_event is None or not exit_event.is_set()):
        while (exit_event is None or not exit_event.is_set()): ## If the pipline is good the remove this 
                                                                ## condition and use the above one

            self.history.append(self.position) ## Adding the agent history
            print(self.history)
            print(hasattr(self.communicator, "get_camera_observation"))
            rgb_image = self.communicator.get_camera_observation(self.camera_id, 'lit')
            print(f"rgb_image type: {type(rgb_image)}")
            print("RGB image is taken from environment.")
            plt.imshow(rgb_image)
            plt.title("RGB Image")
            plt.axis('off')
            plt.show()
            try:
                depth_image = self.communicator.get_camera_observation(self.camera_id, 'depth')
                print("Depth image is taken from the environment.")
            except Exception as e:
                print(f"Error in getting depth map for agent {self.id} with camera {self.camera_id}: {e}")
                depth_image = self.communicator.generate_depth_model(rgb_image)
            print(f"depth_image type: {type(depth_image)}")
            plt.imshow(depth_image)
            plt.title("Depth Image")
            plt.axis('off')
            plt.show()
            try:
                segmentation_map = self.communicator.get_camera_observation(self.camera_id, 'object_mask')
                print("segmentation map is taken from the environment.")

            except Exception as e:
                print(f"Error in getting segmentation map for agent {self.id} with camera {self.camera_id}: {e}")
                segmentation_map = self.communicator.generate_segmentation_model(rgb_image)
            print(f"segmenatation_image type: {type(depth_image)}")
            plt.imshow(segmentation_map)
            plt.title("segment Image")
            plt.axis('off')
            plt.show()
            
            # Scene Objects
            # scene_objects = self.communicator.get_objects()
            # print(f"Scene objects: {scene_objects}")


            cam_info = self.communicator.get_camera_information(self.camera_id, rgb_image)
            print(f"Camera information: ", cam_info)
            # current_yaw_rad = math.radians(self.yaw)

            # Genarting naviagtable waypoints using rgb, segmentation and depth map
            if generate_waypoint_zeroshot:
                response1 = self.nav_llm.generate_waypoints_openai(
                    image = rgb_image,
                    depth_map = depth_image,
                    seg_mask = segmentation_map,
                    system_prompt = WAYPOINT_SYSTEM_PROMPT,
                    waypoint_prompt = WAYPOINT_GENERATION_PROMPT)

                print("waypoint repsonse zeroshot", response1)
            else:
                response1 = random_waypoint_generator(
                    segmentation_mask = segmentation_map,
                    depth_map = depth_image, 
                    agent_position = self.position
                )
                print("waypoint repsonse random generation", response1)
            visualize_waypoints_on_image(response1, rgb_image)
            print("Waypoints generated: ", response1)
            # Get true depth map
            true_depth_image = self.communicator.get_true_depth(self.camera_id)
            print("True depth image is taken from the environment.")
            # Convert them to world coordinates
            response1 = self.basic_refinement(response1, cam_info['img_height'], cam_info['img_width'])
            print("Refined waypoints: ", response1)
            waypoints_world_coords = pixel_to_world(
                json.loads(response1)['waypoints'],
                true_depth_image,
                cam_info['k'],
                cam_info['cam_position'],
                cam_info['cam_rotation']
            )
            print("Waypoints in world coordinates: ", waypoints_world_coords)
            # Select a random waypoint from the list of world coordinates
            selected_waypoint = self.random_waypoint_world_coord_selector(waypoints_world_coords)

            ## Yuyuan must test this part if it works thsi will go below devanshi's functionality
            # if selected_waypoint is None:
            #     print("No valid waypoints selected, skipping iteration")
            #     continue
            # else:
            #     navigate_to_target(
            #         self.communicator,
            #         self.camera_id,
            #         [self.position.x, self.position.y],
            #         selected_waypoint[:2]
            #     )

            ## Derick refinement module

            # pil_image = Image.fromarray(rgb_image)
            # # Select most viable waypoints
            # waypoints1 = [(p['x'], p['y']) for p in json.loads(response1)['waypoints']]
            # response2 = self.nav_llm.select_waypoints_openai(
            #     image = Image.fromarray(rgb_image),
            #     waypoints = waypoints1,
            #     system_prompt = WAYPOINT_SYSTEM_PROMPT,
            #     waypoint_prompt = WAYPOINT_SELECTION_PROMPT)

            # print("waypoint selection", response2)

            waypoints2 = [(p['x'], p['y']) for p in json.loads(response1)['waypoints']]

            # # convert into next step format
            # waypoints = {chr(65+i): p for i,p in enumerate(waypoints2)}

            #  Devanshi's functionality must go here

            waypoints = self._parse_waypoints(waypoints2)
        
            # if not waypoints:
            #     print("No valid waypoints received")
            #     continue
                
            # # Select best waypoint
            selected_waypoint = self.nav_llm.select_best_waypoint(
                image=rgb_image,
                waypoints=waypoints,
                current_pos=self.position,
                destination=self.destination,
                history=self.history
            )
            print("Selected waypoint from LLM: ", selected_waypoint)
            if not selected_waypoint:
                print("Invalid waypoint selection")
                continue
            final_waypoint = self.extract_waypoint_label(selected_waypoint)
            print("Selected waypoint: ", final_waypoint)  
            # Convert selected waypoint to world coordinates
            # world_pos = self._pixel_to_world_coords(
            #     waypoints[selected_waypoint],
            #     cam_info,
            #     depth_image
            # )
            # print("World position: ", world_pos)
            # if world_pos is None:
            #     print("Failed to convert waypoint to world coordinates")
            #     continue
                
            # # Move agent towards selected waypoint using movement controller
            # reached = self.movement_controller.move_to_waypoint(world_pos)
            
            # if reached:
            #     print(f"Agent {self.id} reached waypoint")
            
            # # Check if destination reached
            # if Vector.distance(self.position, self.destination) < self.config.get('arrival_threshold', 1.0):
            #     print(f"Agent {self.id} reached destination!")
            #     break
    
    def agent_reached_destination(self):
        """Check if the agent has reached its destination."""
        distance = Vector.distance(self.position, self.destination)
        if distance < self.config.get('navReq.arrival_threshold', 1.0):
            print(f"Agent {self.id} has reached the destination at {self.destination}.")
            return True
        return False
    
    def distance_current_to_waypoints(self, waypoints):
        """Calculate the distance to all the waypoints."""
        distances = {}
        for label, coords in waypoints.items():
            distance = Vector.distance(self.position, Vector(*coords))
            distances[label] = distance
        return distances
    
    def distance_waypoints_to_destination(self, waypoints):
        """Calculate the distance from all waypoints to the destination."""
        distances = {}
        for label, coords in waypoints.items():
            waypoint_vector = Vector(*coords)
            distance = Vector.distance(waypoint_vector, self.destination)
            distances[label] = distance
        return distances
    
    def distance_to_destination(self):
        """Calculate the distance to the destination."""
        return Vector.distance(self.position, self.destination)
        

