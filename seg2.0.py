import json
import os
import numpy as np
from PIL import Image, ImageFilter
from segment_anything import sam_model_registry, SamPredictor

# CONFIG -------------------
pref_size = 20
IMAGE_NAME = "eco_00000.png"  # just change this, extension included
OUT_DIR = "masks"

CLASS_MAP = {
    "large_orange_cone": "orange_cone",  # treat as same prompt
}
# ---------------------------

image_path = f"Images/{IMAGE_NAME}"
ann_path = f"Ann/{IMAGE_NAME}.json"

os.makedirs(OUT_DIR, exist_ok=True)


def load_cone_boxes(ann_path):
    with open(ann_path) as f:
        ann = json.load(f)
    boxes = []
    for obj in ann["objects"]:
        if obj["geometryType"] != "rectangle":
            continue
        (x0, y0), (x1, y1) = obj["points"]["exterior"]
        cls = obj["classTitle"]
        cls = CLASS_MAP.get(cls, cls)  # normalize class name
        boxes.append((cls, (x0, y0, x1, y1)))
    return boxes


def is_big_enough(box, min_size=pref_size):
    x0, y0, x1, y1 = box
    return (x1 - x0) >= min_size and (y1 - y0) >= min_size


def get_cone_mask(predictor, box):
    x0, y0, x1, y1 = box
    box_arr = np.array([x0, y0, x1, y1])
    masks, scores, _ = predictor.predict(box=box_arr, multimask_output=False)
    full_mask = masks[0]
    mask_crop = full_mask[y0:y1, x0:x1]
    return Image.fromarray((mask_crop * 255).astype(np.uint8), mode="L")


# --- Load model ---
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam.to(device="cuda")
predictor = SamPredictor(sam)

# --- Load image + boxes ---
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)
ann = load_cone_boxes(ann_path)

valid_boxes = [(cls, box) for cls, box in ann if is_big_enough(box)]
print(f"Found {len(ann)} cones, {len(valid_boxes)} pass the size filter")

predictor.set_image(image_np)

# --- Segment each cone and save mask + metadata ---
metadata = []

for i, (cls, box) in enumerate(valid_boxes):
    print(f"Segmenting cone {i+1}/{len(valid_boxes)} ({cls})...")
    mask = get_cone_mask(predictor, box)
    mask.save(f"{OUT_DIR}/{IMAGE_NAME}_mask_{i}.png")
    metadata.append({"index": i, "cls": cls, "box": box})

with open(f"{OUT_DIR}/{IMAGE_NAME}_metadata.json", "w") as f:
    json.dump({"image_name": IMAGE_NAME, "cones": metadata}, f, indent=2)

print(f"Saved {len(valid_boxes)} masks to {OUT_DIR}/")