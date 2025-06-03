import math
import traceback
import numpy as np
from simworld.agent.base_agent import BaseAgent
from simworld.utils.vector import Vector
from utils.generate_depth_map import generate_depth_from_img
from utils.prompt_utils import WAYPOINT_GENERATION_PROMPT, WAYPOINT_SYSTEM_PROMPT
# from simworld.traffic.base.traffic_signal import TrafficSignalState
# from agent.action_space import Action, ActionSpace
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
    
    def navigate(self, exit_event):
        print(f"Agent {self.id} is navigating to destination {self.destination}, current position: {self.position}")

        while (exit_event is None or not exit_event.is_set()):
            self.history.append(self.position) ## Adding the agent history
            print(self.history)
            print(hasattr(self.communicator, "get_camera_observation"))
            rgb_image = self.communicator.get_camera_observation(self.camera_id, 'lit')

            # Already an image as a NumPy array
            img_np = np.ascontiguousarray(rgb_image)
            img = Image.fromarray(img_np)
            img.show()
            print(type(img))
            # depth_image = self.communicator.get_camera_observation(self.camera_id, 'depth')

            # depth_image = self.communicator.generate_depth_model(img)
            print("Rgb image and depth image has been loaded")
            cam_info = self.communicator.get_camera_information(self.camera_id, rgb_image)

            current_yaw_rad = math.radians(self.yaw)

            # Genarting naviagtable waypoints
            response = self.nav_llm.generate_waypoints_openai(
                image = rgb_image,
                system_prompt = WAYPOINT_SYSTEM_PROMPT,
                waypoint_prompt = WAYPOINT_GENERATION_PROMPT)

            print("waypoint repsonse", response)

            #  Devanshi's functionality must go here


