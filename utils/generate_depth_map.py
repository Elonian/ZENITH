import torch
import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = "DPT_Large"

midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform if "DPT" in model_type else midas_transforms.small_transform

def generate_depth(image_path: str, output_dir: str):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    input_tensor = transform(img_np).to(device)  # NO ["image"], no unsqueeze

    with torch.no_grad():
        depth = midas(input_tensor)

    depth_resized = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=img.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    depth_normalized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min())

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_depth.png")
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        cv2.imwrite(output_path, cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO))
        print(f"Depth map saved to: {output_path}")

    return depth_normalized



if __name__ == "__main__":
    generate_depth(
        image_path="/mntdata/main/sim_nav/unreal_data/Raw_Step_image_v3/Raw_Step_10.png",
        output_dir="/mntdata/main/sim_nav/unreal_data/Depth_Step_image_v3_generated"
    )
