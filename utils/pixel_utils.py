import os
import cv2

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

if __name__ == "__main__":
    image_path = "/mntdata/main/sim_nav/unreal_data/Raw_Step_image_v3/Raw_Step_4.png"
    output_path = "/mntdata/main/sim_nav/unreal_data/test_pixel"
    # pixel_points = [(300, 900), (100, 750), (650, 800)]
    # pixel_points = [(450, 800), (160, 780), (850, 900)]
    pixel_points = [(400, 800), (600, 800), (800, 800)]
    path = add_pixel_points_with_dir(image_path, pixel_points, output_path)