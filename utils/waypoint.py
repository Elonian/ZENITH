import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import ast

from generate_depth_map import generate_depth
from pixel_utils import add_pixel_points_with_dir


os.environ["MODELSCOPE_CACHE"] = "/mntdata/src/sgg-llm"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

if torch.cuda.is_available():
    print("cuda is available")
    model.to("cuda")

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
model.eval()


def generate_waypoints(messages):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=20480)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text


def way_point_prediction(image_path: str, output_path: str):
    depth_map = generate_depth(image_path=image_path, output_dir=output_path)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    depth_filename = f"{base_filename}_depth.png"
    depth_map_path = os.path.join("/mntdata/main/sim_nav/unreal_data/Depth_Step_image_v3_generated", depth_filename)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/mntdata/main/sim_nav/unreal_data/Raw_Step_image_v3/Raw_Step_1.png"},
                {"type": "image", "image": "/mntdata/main/sim_nav/unreal_data/Depth_Step_image_v3_generated/Raw_Step_1_depth.png"},
                {"type": "text", "text": "RGB and depth image of the environment."}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Waypoints: [(300, 900), (100, 750), (650, 800)]"}]
        },
        # {
        #     "role": "user",
        #     "content":[
        #         {"type": "image", "image": "/mntdata/main/sim_nav/unreal_data/test_pixel/Raw_Step_1.png"},
        #         {"type": "text", "text": "The waypoints are projected on the image, all the waypoints are on ground. they are not on any objects or wall"}
        #     ]
        # },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "image", "image": depth_map_path},
                {
                    "type": "text",
                    "text": (
                        "Now for the current RGB image and its depth map, generate navigable waypoints in 2D pixel coordinates. "
                        "Waypoints should lie only on traversable ground, avoiding obstacles. "
                        "Output a list like [(x1, y1), (x2, y2), ...]."
                    )
                }
            ]
        }
    ]

    return generate_waypoints(messages)


if __name__ == "__main__":
    image_dir = "/mntdata/main/sim_nav/unreal_data/Raw_Step_image_v3"
    output_dir = "/mntdata/main/sim_nav/unreal_data/Depth_Step_image_v3_generated"

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png") and f.startswith("Raw_Step")])
    full_image_paths = [os.path.join(image_dir, f) for f in image_files]
    save_dir = "/mntdata/main/sim_nav/unreal_data/test_pixel"
    # output = way_point_prediction(
    #     image_path="/mntdata/main/sim_nav/unreal_data/Raw_Step_image_v3/Raw_Step_4.png",
    #     output_path="/mntdata/main/sim_nav/unreal_data/Depth_Step_image_v3_generated"
    # )
    for image_path in full_image_paths:
        output = way_point_prediction(
            image_path=image_path,
            output_path=output_dir
        )
        print(f"{os.path.basename(image_path)} -> {output}")
        waypoints_str = output.split("Waypoints:")[1].strip()
        pixel_points = ast.literal_eval(waypoints_str)
        output_img_path = add_pixel_points_with_dir(image_path, pixel_points, save_dir)
        print(f"Saved image with waypoints to: {output_img_path}")

        print(output)
