import torch
import torchvision.transforms as T
import torchvision.models.segmentation as models
import numpy as np
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seg_model = models.deeplabv3_resnet101(pretrained=True)
seg_model.to(device).eval()

seg_transform = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def generate_segmentation_mask(img_input, output_dir=None, output_name="segmentation_map"):
    if isinstance(img_input, np.ndarray):
        img = Image.fromarray(img_input)
    elif isinstance(img_input, Image.Image):
        img = img_input.convert("RGB")
    else:
        raise ValueError("img_input must be a NumPy array or PIL.Image")

    orig_size = img.size
    input_tensor = seg_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = seg_model(input_tensor)["out"]
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    mask[pred == 0] = [0, 255, 0]      # Green for ground
    mask[pred != 0] = [255, 0, 0]      # Red for objects

    mask_img = Image.fromarray(mask).resize(orig_size)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{output_name}.png")
        mask_img.save(path)
    print("Segmenattion map created using deeplabv3")
    return np.array(mask_img)

# seg_mask = generate_segmentation_mask(agent_view_img, output_dir="./seg_output", output_name="agent_seg")
