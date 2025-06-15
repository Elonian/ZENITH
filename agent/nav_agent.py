import json
import math
import traceback
import numpy as np
import matplotlib.pyplot as plt
from simworld.agent.base_agent import BaseAgent
from utils.vector import Vector
from utils.generate_segment import generate_segmentation_mask
from utils.generate_depth_map import generate_depth_from_img
from utils.prompt_utils import WAYPOINT_SYSTEM_PROMPT, WAYPOINT_GENERATION_PROMPT, WAYPOINT_VERIFICATION_PROMPT
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
        self.direction = 0
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
            coordinates = [(p['x'], p['y']) for p in json.loads(coordinates)['waypoints']]
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
        counter = 0
        for wp in waypoints["waypoints"]:
            x, y = wp['x'], wp['y']
            if 0 <= x < width and 0 <= y < height:
                refined_waypoints.append(wp)
            else:
                counter += 1
        if counter:
            print(f'Removed {counter} out-of-bounds waypoints')
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

    def get_images(self):
        # rgb
        rgb_image = self.communicator.get_camera_observation(self.camera_id, 'lit')
        print("RGB image is taken from environment.")
        # depth
        try:
            depth_image = self.communicator.get_camera_observation(self.camera_id, 'depth')
            print("Depth image is taken from environment.")
        except Exception as e:
            print(f"Error in getting depth map for agent {self.id} with camera {self.camera_id}: {e}")
            depth_image = self.communicator.generate_depth_model(rgb_image)
        # segmentation
        try:
            seg_image = self.communicator.get_camera_observation(self.camera_id, 'object_mask')
            print("segmentation map is taken from environment.")
        except Exception as e:
            print(f"Error in getting segmentation map for agent {self.id} with camera {self.camera_id}: {e}")
            seg_image = self.communicator.generate_segmentation_model(rgb_image)

        return rgb_image, depth_image, seg_image

    def show_images(self, rgb_image, depth_image, seg_image):
        def show(image, title):
            print(f"{title} type: {type(image)}")
            plt.imshow(image)
            plt.title(title)
            plt.axis('off')
            plt.show()
        show(rgb_image, "RGB image")
        show(depth_image, "Depth image")
        show(seg_image, "Segmentation map")

    
    def navigate(self, exit_event, generate_waypoint_zeroshot=True):
        # humanoid_ids = [str(self.id)]
        # position_and_direction = self.communicator.get_position_and_direction(humanoid_ids = humanoid_ids)
        # print(position_and_direction)
        # for id in humanoid_ids:
        #     pos, dir = position_and_direction[('humanoid', str(id))]
        #     self.position = Vector(pos.x, pos.y)
        #     heading = dir
        # print(f"Information from simulator Agent {self.id} current position: {self.position}, heading: {heading}")

        # print(f"Agent {self.id} is navigating to destination {self.destination}, current position: {self.position}")
        # humanoid_ids = [self.id]
        # position_and_direction = self.communicator.get_position_and_direction(humanoid_ids = humanoid_ids)
        # for id in humanoid_ids:
        #     pos, dir = position_and_direction[('humanoid', id)]
        #     self.position = Vector(pos['x'], pos['y'])
        #     self.yaw = dir['yaw']
        # print(f"Information from simulator Agent {self.id} current position: {self.position}, heading: {self.yaw}")

        while not self.agent_reached_distination_with_threshold():
        # while (exit_event is None or not exit_event.is_set()): ## If the pipline is good the remove this 
                                                                ## condition and use the above
            humanoid_ids = [str(self.id)]
            position_and_direction = self.communicator.get_position_and_direction(humanoid_ids = humanoid_ids)
            print(position_and_direction)
            for id in humanoid_ids:
                pos, dir = position_and_direction[('humanoid', str(id))]
                self.position = Vector(pos.x, pos.y)
                heading = dir
            print(f"Information from simulator Agent {self.id} current position: {self.position}, heading: {heading}")

            ### DATA COLLECTION
            
            self.history.append(self.position) ## Adding the agent history
            print(f"Overall history of agent: {self.history}")
            # print(hasattr(self.communicator, "get_camera_observation"))

            rgb_image, depth_image, segmentation_map = self.get_images()
            # self.show_images(rgb_image, depth_image, segmentation_map)
            
            # Get true depth map
            true_depth_image = self.communicator.get_true_depth(self.camera_id)
            print("True depth image is taken from the environment.")
            
            # Scene Objects
            # scene_objects = self.communicator.get_objects()
            # print(f"Scene objects: {scene_objects}")

            cam_info = self.communicator.get_camera_information(self.camera_id, rgb_image)
            # print(f"Camera information: ", cam_info)
            # current_yaw_rad = math.radians(self.yaw)

            ### GENERATION STEP

            # Genarting naviagtable waypoints using rgb, segmentation and depth map
            if generate_waypoint_zeroshot:
                response1 = self.nav_llm.generate_waypoints_openai(
                    image = rgb_image,
                    depth_map = depth_image,
                    seg_mask = segmentation_map
                )
                # print("waypoint repsonse zeroshot using vlm", response1)
            else:
                response1 = random_waypoint_generator(
                    segmentation_mask = segmentation_map,
                    depth_map = depth_image, 
                    agent_position = self.position
                )
                # print("waypoint repsonse random generation", response1)
            visualize_waypoints_on_image(response1, rgb_image)
            # print("Waypoints generated: ", response1)

            ### CONVERSION STEP
            # Convert them to world coordinates
            response1 = self.basic_refinement(response1, cam_info['img_height'], cam_info['img_width'])
            # print("Basic refinement done and final waypoints for selection: ", response1)
            waypoints_world_coords = pixel_to_world(
                json.loads(response1)['waypoints'],
                true_depth_image,
                cam_info['k'],
                cam_info['cam_position'],
                cam_info['cam_rotation']
            )
            # print("final waypoints in world coordinates: ", waypoints_world_coords)
            ## Getting Agent's current position and heading
            # Select a random waypoint from the list of world coordinates
            # selected_waypoint = self.random_waypoint_world_coord_selector(waypoints_world_coords)

            ## Yuyuan must test this part if it works thsi will go below devanshi's functionality
            
            # heading_correct = self.communicator.get_position_and_direction(
            #     vechile_ids = [],
            #     pedestrian_ids = [],
            #     humanoid_ids = [self.id],
            #     scooter_ids = []                
            # )[('humanoid', self.id)]['direction']
            # print(f"Heading from communicator: {heading_correct}")

            # print(f"Agent {self.id} current heading: {heading}")
            # if selected_waypoint is None:
            #     print("No valid waypoints selected, skipping iteration")
            #     continue
            # else:
            #     navigate_to_target_with_heading(
            #         self.communicator,
            #         self.id,
            #         [self.position.x, self.position.y],
            #         selected_waypoint[:2],
            #         heading,
            #     )

            ## Derick refinement module

            # pil_image = Image.fromarray(rgb_image)
            # # Select most viable waypoints
            # waypoints1 = [(p['x'], p['y']) for p in json.loads(response1)['waypoints']]
            # response2 = self.nav_llm.validate_waypoints_openai(
            #     image = Image.fromarray(rgb_image),
            #     waypoints = waypoints1)

            # print("waypoint selection", response2)

            # waypoints2 = [(p['x'], p['y']) for p in json.loads(response1)['waypoints']]


            # # convert into next step format
            # waypoints = {chr(65+i): p for i,p in enumerate(waypoints2)}

            ### SELECTION STEP
            
            #  Devanshi's functionality must go here


            #waypoints = self._parse_waypoints(response1)
            #distance_current = self.distance_current_to_waypoints(waypoints)
            #distance_destination = self.distance_waypoints_to_destination(waypoints)

            waypoints_world_coords_xy = [(x, y) for x, y, z in waypoints_world_coords]

            # waypoints_world_coords_xy = [(x, y) for x, y, z in waypoints_world_coords]
            waypoints_labelled = self._parse_waypoints(waypoints_world_coords_xy)
            # print("Labelled waypoints in world coordinates: ", waypoints_labelled)
            # print("Current agent position: ", self.position)
            distance_current = self.distance_current_to_waypoints(waypoints_labelled)
            distance_destination = self.distance_waypoints_to_destination(waypoints_labelled)

            print("Distances to current waypoints: ", distance_current)
            print("Distances from waypoints to destination: ", distance_destination)
                
            # # Select best waypoint
            selected_waypoint = self.nav_llm.select_best_waypoint(
                image=rgb_image,
                waypoints=waypoints_labelled,
                current_pos=self.position,
                destination=self.destination,
                history=self.history,
                distances_from_current=distance_current,
                distances_to_destination=distance_destination
            )
            # print("Selected waypoint from LLM: ", selected_waypoint)
            if selected_waypoint is None:
                print("Invalid waypoint selection")
                continue

            #final_waypoint = self.extract_waypoint_label(selected_waypoint)
            #print("Selected waypoint: ", final_waypoint)
            #selected_waypoint = json.loads(selected_waypoint)
            

            final_waypoint = self.extract_waypoint_label(selected_waypoint)
            print(f"Best waypoint that will reduce the distance to destination: {self.get_best_waypoint_reduce_distance(distance_destination)}")
            print(f"LLM choosed waypoint label and its distance to destination: {final_waypoint} and {distance_destination[final_waypoint]}")
            # print("Selected waypoint: ", final_waypoint)  

            ### EXECUTION STEP
            
            ## Movement code working.
            # print("current agent yaw:", heading)
            # heading = self.direction
            final_waypoint_world = waypoints_labelled[final_waypoint]
            print(f"checking the list: {[self.position.x, self.position.y]} and heading: {heading} and desired point to go: {list(final_waypoint_world)}")
            print(f"Final waypoint in world coordinates agents is going to:{final_waypoint_world}")
            # print(f"Agent {self.id} current heading: {heading} and heading towards")
            if selected_waypoint is None: # redundant check, already checked in selection step
                print("No valid waypoints selected, skipping iteration")
                continue
            else:
                navigate_to_target_with_heading(
                    self.communicator,
                    self.id,
                    [self.position.x, self.position.y],
                    list(final_waypoint_world),
                    heading,
                )
            print(f"Agent {self.id} has reached the waypoint {final_waypoint_world}.")
    
    def agent_reached_destination(self):
        """Check if the agent has reached its destination."""
        distance = Vector.distance(self.position, self.destination)
        if distance < self.config.get('navReq.arrival_threshold', 0.2):
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
    
    def get_best_waypoint_reduce_distance(self, waypoints_labelled_distance_values):
        """Get the waypoint that reduces the distance to the destination."""
        best_waypoint = None
        min_distance = float('inf')
        for label, distance in waypoints_labelled_distance_values.items():
            if distance < min_distance:
                min_distance = distance
                best_waypoint = label
        return best_waypoint, min_distance
    
    def agent_reached_distination_with_threshold(self, threshold=0.2):
        """Check if the agent has reached its destination within a threshold."""
        distance = Vector.distance(self.position, self.destination)
        if distance < threshold:
            print(f"Agent {self.id} has reached the destination at {self.destination} within threshold {threshold}.")
            return True
        return False
        

