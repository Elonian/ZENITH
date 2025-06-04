import math
import json
import unrealcv
import numpy as np
from simworld.communicator.communicator import Communicator
from simworld.communicator.unrealcv import UnrealCV
from unrealcv.util import read_png
from utils.generate_depth_map import generate_depth_from_img
from utils.generate_segment import generate_segmentation_mask

class nav_communicator(Communicator):
    def __init__(self, unrealcv: UnrealCV):
        super().__init__(unrealcv)
    
    def spawn_agent(self, agent, model_path):
        name = self.get_humanoid_name(agent.id)
        self.unrealcv.spawn_bp_asset(model_path, name)
        # Convert 2D position to 3D (x,y -> x,y,z)
        location_3d = (
            agent.position.x,  # Unreal X = 2D Y
            agent.position.y,  # Unreal Y = 2D X
            100  # Z coordinate (ground level)
        )
        # Convert 2D direction to 3D orientation (assuming rotation around Z axis)
        orientation_3d = (
            0,  # Pitch
            math.degrees(math.atan2(agent.direction.y, agent.direction.x)),  # Yaw
            0  # Roll
        )
        self.unrealcv.set_location(location_3d, name)
        self.unrealcv.set_orientation(orientation_3d, name)
        self.unrealcv.set_scale((1, 1, 1), name)  # Default scale
        self.unrealcv.set_collision(name, True)
        self.unrealcv.set_movable(name, True)
    
    def get_camera_observation(self, cam_id, viewmode, mode='direct'):
        return self.unrealcv.get_image(cam_id, viewmode, mode)

    
    def get_intrinsic_matrix(self, fov_deg, width, height):
        fov_rad = np.deg2rad(fov_deg)
        f_x = width / (2 * np.tan(fov_rad / 2))
        f_y = f_x 
        c_x = width / 2
        c_y = height / 2

        K = np.array([
            [f_x,    0, c_x],
            [0,    f_y, c_y],
            [0,      0,   1]
        ])
        return K
    
    def get_camera_information(self, camera_id:int, rgb_image):
        information = {}
        height, width, _ = rgb_image.shape
        information['img_height'] = height
        information['img_width'] = width
        print(information)
        information['cam_position'] = self.unrealcv.get_camera_location(camera_id)
        information['cam_rotation'] = self.unrealcv.get_camera_rotation(camera_id)
        try:
            fov = self.unrealcv.get_camera_fov(camera_id)
            print(f"FOV for camera {camera_id}: {fov}")
        except Exception as e:
            print(f"Error getting FOV for camera {camera_id}: {e}")
            fov = 90  # Default FOV if not available
        information['fov'] = float(fov) 
        # information['fov'] = self.unrealcv.get_camera_fov(camera_id)
        information['k'] = self.get_intrinsic_matrix(information['fov'], width, height)
        return information
    def get_true_depth(self, camera_id:int):
        """Get true depth map from the camera."""
        return self.unrealcv.get_depth_map(camera_id)

    def generate_depth_model(self, rgb_image):
        return generate_depth_from_img(rgb_image)
    
    def generate_segmentation_model(self, rgb_image):
        return generate_segmentation_mask(rgb_image)

    def get_objects(self):
        """Get objects in the scene."""
        objects = self.unrealcv.get_objects()
        print(f"Found {len(objects)} objects in the scene.")
        # Uncomment the following lines if you want to filter and return specific object types
        # scene_objects = []
        # for obj in objects:
        #     if obj['type'] == 'StaticMeshActor':
        #         scene_objects.append({
        #             'name': obj['name'],
        #             'location': obj['location'],
        #             'rotation': obj['rotation'],
        #             'scale': obj['scale']
        #         })
        return objects

    
    # def get_scene_objects(self):
    #     objects = self.unrealcv.get_scene_objects()
    #     print(f"Found {len(objects)} objects in the scene.")
    #     print(objects)
    #     # scene_objects = []
    #     # for obj in objects:
    #     #     if obj['type'] == 'StaticMeshActor':
    #     #         scene_objects.append({
    #     #             'name': obj['name'],
    #     #             'location': obj['location'],
    #     #             'rotation': obj['rotation'],
    #     #             'scale': obj['scale']
    #     #         })
    #     return objects