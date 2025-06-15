import os
import cv2
import numpy as np
import random 
import json
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def visualize_waypoints_on_image(response_json_str, rgb_image):

    response_dict = json.loads(response_json_str)
    waypoints = response_dict.get("waypoints", [])
    image_copy = rgb_image.copy()
    for i, wp in enumerate(waypoints):
        x, y = wp["x"], wp["y"]
        cv2.circle(image_copy, (x, y), radius=5, color=(0, 255, 0), thickness=-1)  
        cv2.putText(
            image_copy,
            str(i + 1),
            (x + 5, y - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 0, 0),  # Blue text
            thickness=1
        )

    image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title("Waypoints Visualized on Image")
    plt.axis("off")
    plt.show()


def add_pixel_points_with_dir(image_path, pixel_points, save_dir):

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    for point in pixel_points:
        cv2.circle(image, point, radius=5, color=(0, 0, 255), thickness=-1)  # red in BGR
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(save_dir, f"{base_filename}.png")
    cv2.imwrite(output_path, image)
    return output_path



def random_waypoint_generator(segmentation_mask, depth_map, agent_position, depth_threshold = 2.5, min_distance=20, num_waypoints=12):

    ground_color = np.array([0, 255, 0])
    ground_mask = np.all(segmentation_mask == ground_color, axis=-1)
    depth_mask = depth_map > depth_threshold
    valid_mask = np.logical_and(ground_mask, depth_mask)
    y_indices, x_indices = np.where(valid_mask)
    agent_x, agent_y = agent_position
    distances = np.sqrt((x_indices - agent_x)**2 + (y_indices - agent_y)**2)
    far_enough_mask = distances > min_distance
    candidate_x = x_indices[far_enough_mask]
    candidate_y = y_indices[far_enough_mask]
    if len(candidate_x) == 0:
        return []
    indices = list(range(len(candidate_x)))
    random.shuffle(indices)
    selected_indices = indices[:num_waypoints] if len(indices) >= num_waypoints else indices
    waypoints = [(int(candidate_x[i]), int(candidate_y[i])) for i in selected_indices]
    
    return waypoints


def pixel_to_world(waypoints, depth_image, intrinsics, cam_pos, cam_rot_deg):
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # Parse cam_pos if string
    if isinstance(cam_pos, str):
        cam_pos = np.array(list(map(float, cam_pos.strip().split())), dtype=float)
    else:
        cam_pos = np.array(cam_pos, dtype=float)

    # Parse rotation degrees
    cam_rot_deg = list(map(float, cam_rot_deg.strip().split()))
    # Assuming rotation order is 'xyz' (adjust if needed)
    r = R.from_euler('xyz', cam_rot_deg, degrees=True)
    R_wc = r.as_matrix()

    if depth_image.ndim == 3:
        depth_image = depth_image[:, :, 0]

    world_points = []
    for wp in waypoints:
        v, u = wp["x"], wp["y"]  # v = row, u = col
        z = depth_image[u, v]
        # print(f"Pixel: ({v}, {u}), Depth: {z}")
        if isinstance(z, np.ndarray):
            z = float(z[0])
        if z <= 0:
            continue
        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy
        z_cam = z

        p_cam = np.array([x_cam, y_cam, z_cam])
        p_world = R_wc @ p_cam + cam_pos
        world_points.append(p_world)

    return np.array(world_points)


if __name__ == "__main__":
    image_path = "/mntdata/main/sim_nav/unreal_data/Raw_Step_image_v3/Raw_Step_4.png"
    output_path = "/mntdata/main/sim_nav/unreal_data/test_pixel"
    # pixel_points = [(300, 900), (100, 750), (650, 800)]
    # pixel_points = [(450, 800), (160, 780), (850, 900)]
    pixel_points = [(400, 800), (600, 800), (800, 800)]
    path = add_pixel_points_with_dir(image_path, pixel_points, output_path)