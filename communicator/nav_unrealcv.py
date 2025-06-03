from simworld.communicator.unrealcv import UnrealCV
import numpy as np
import cv2
import PIL
import json
import os
import time
from io import BytesIO
from threading import Lock
import PIL.Image
import unrealcv
from IPython.display import display
from unrealcv.util import read_png


class NavUnrealCV(UnrealCV):
    def __init__(self, port = 9000, ip = '127.0.0.1', resolution = (320, 240)):
        super().__init__(port, ip, resolution)
        
    def get_camera_location(self, camera_id: int):
        """Get camera location.

        Args:
            camera_id: ID of the camera to get location.

        Returns:
            Location (x, y, z) of the camera.
        """
        cmd = f'vget /camera/{camera_id}/location'
        with self.lock:
            return self.client.request(cmd)
    
    def get_camera_rotation(self, camera_id: int):
        """Get camera rotation.

        Args:
            camera_id: ID of the camera to get rotation.

        Returns:
            Rotation (pitch, yaw, roll) of the camera.
        """
        cmd = f'vget /camera/{camera_id}/rotation'
        with self.lock:
            return self.client.request(cmd)

    def get_camera_fov(self, camera_id: int):
        """Get camera field of view.

        Args:
            camera_id: ID of the camera to get field of view.

        Returns:
            Field of view of the camera.
        """
        cmd = f'vget /camera/{camera_id}/fov'
        with self.lock:
            return self.client.request(cmd)

    def get_camera_resolution(self, camera_id: int):
        """Get camera resolution.

        Args:
            camera_id: ID of the camera to get resolution.

        Returns:
            Resolution (width, height) of the camera.
        """
        cmd = f'vget /camera/{camera_id}/size'
        with self.lock:
            return self.client.request(cmd)


    def get_image(self, cam_id, viewmode, mode='direct', img_path=None):
        """Get image.

        Args:
            cam_id: Camera ID.
            viewmode: View mode.
            mode: Mode.
            img_path: Image path.
        """
        image = None
        try:
            if mode == 'direct':  # get image from unrealcv in png format
                if viewmode == 'depth':
                    cmd = f'vget /camera/{cam_id}/{viewmode} npy'
                    # image = read_npy(self.client.request(cmd))
                    image = self._decode_npy(self.client.request(cmd))
                else:
                    cmd = f'vget /camera/{cam_id}/{viewmode} png'
                    # image = read_png(self.client.request(cmd))
                    image = self._decode_png(self.client.request(cmd))
            elif mode == 'file':  # save image to file and read it
                img_path = os.path.join(os.getcwd(), f'{cam_id}-{viewmode}.png')
                cmd = f'vget /camera/{cam_id}/{viewmode} {img_path}'
                img_dirs = self.client.request(cmd)
                image = cv2.imread(img_dirs)

            elif mode == 'fast':  # get image from unrealcv in bmp format
                cmd = f'vget /camera/{cam_id}/{viewmode} bmp'
                image = self._decode_bmp(self.client.request(cmd))

            elif mode == 'file_path':  # save image to file and read it
                cmd = f'vget /camera/{cam_id}/{viewmode} {img_path}'
                img_dirs = self.client.request(cmd)
                image = read_png(img_dirs)

            if image is None:
                raise ValueError(f'Failed to read image with mode={mode}, viewmode={viewmode}')
            return image

        except Exception as e:
            print(f'Error reading image: {str(e)}')
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def _decode_npy(self, res):
        """Decode NPY image.

        Args:
            res: NPY image.

        Returns:
            Decoded image.
        """
        image = np.load(BytesIO(res))
        eps = 1e-6
        depth_log = np.log(image + eps)

        depth_min = np.min(depth_log)
        depth_max = np.max(depth_log)
        normalized_depth = (depth_log - depth_min) / (depth_max - depth_min)

        gamma = 0.5
        normalized_depth = np.power(normalized_depth, gamma)

        image = (normalized_depth * 255).astype(np.uint8)

        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        return image

    def _decode_png(self, res):
        """Decode PNG image.

        Args:
            res: PNG image.

        Returns:
            Decoded image.
        """
        img = np.asarray(PIL.Image.open(BytesIO(res)))
        img = img[:, :, :-1]  # delete alpha channel
        img = img[:, :, ::-1]  # transpose channel order
        return img

    def _decode_bmp(self, res, channel=4):
        """Decode BMP image.

        Args:
            res: BMP image.
            channel: Channel.

        Returns:
            Decoded image.
        """
        img = np.fromstring(res, dtype=np.uint8)
        img = img[-self.resolution[1]*self.resolution[0]*channel:]
        img = img.reshape(self.resolution[1], self.resolution[0], channel)
        return img[:, :, :-1]


