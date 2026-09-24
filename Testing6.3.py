import os
import json
import torch
import numpy as np
import cv2
from sdnq import SDNQConfig
from diffusers import DiffusionPipeline
from PIL import Image

pipe = DiffusionPipeline.from_pretrained(
    "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic",
    dtype=torch.bfloat16,
    device_map="cuda"
)

PROMPTS = {
    "yellow_cone": "Remove the black stripe from the traffic cone and make it a solid, uniform yellow color all the way from top to base. Keep the cone's shape, lighting, shadow, texture, and background exactly the same, only change the black band to match the surrounding yellow.",
    "orange_cone": "Remove the white stripes from the traffic cone and make it a solid, uniform orange color from top to base. Keep the cone's shape, lighting, shadow, texture, base color, and background exactly the same — only change the white bands to match the surrounding orange.",
    "blue_cone": "Remove the white stripe from the traffic cone and make it a solid, uniform blue color from top to base. Keep the cone's shape, lighting, shadow, texture, base color, and background exactly the same — only change the white band to match the surrounding blue.",
    "large_orange_cone": "Remove the white stripes from the traffic cone and make it a solid, uniform orange color from top to base. Keep the cone's shape, lighting, shadow, texture, base color, and background exactly the same — only change the white bands to match the surrounding orange."
}

# CONFIGS -------------------
pref_size = 18
TARGET_SIZE = 256

TEAM_NAME = "amz"
ASSETS_DIR = "Assets"
IMG_DIR = os.path.join(ASSETS_DIR, TEAM_NAME, "img")
ANN_DIR = os.path.join(ASSETS_DIR, TEAM_NAME, "ann")
OUTPUT_DIR = os.path.join(ASSETS_DIR, TEAM_NAME, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --------------------------


def load_cone_boxes(ann_path):
    with open(ann_path) as f:
        ann = json.load(f)
    boxes = []
    for obj in ann["objects"]:
        if obj["geometryType"] != "rectangle":
            continue
        (x0, y0), (x1, y1) = obj["points"]["exterior"]
        boxes.append((obj["classTitle"], (x0, y0, x1, y1)))
    return boxes


def is_big_enough(box, min_size=pref_size):
    x0, y0, x1, y1 = box
    return (x1 - x0) >= min_size and (y1 - y0) >= min_size


def box_area(box):
    x0, y0, x1, y1 = box
    return (x1 - x0) * (y1 - y0)


def edit_small_cone(crop, prompt):
    orig_w, orig_h = crop.size
    scale = max(1, TARGET_SIZE // max(orig_w, orig_h))
    work_image = crop.resize((orig_w * scale, orig_h * scale), Image.LANCZOS)

    result = pipe(
        prompt=prompt,
        image=work_image,
        guidance_scale=2.5,
        num_inference_steps=10,
        generator=torch.manual_seed(0)
    ).images[0]

    return result.resize((orig_w, orig_h), Image.LANCZOS)


def poisson_paste(base_img, patch_img, box, mode=cv2.NORMAL_CLONE):
    base_bgr = cv2.cvtColor(np.array(base_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    patch_bgr = cv2.cvtColor(np.array(patch_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    center = (x0 + w // 2, y0 + h // 2)

    mask = 255 * np.ones(patch_bgr.shape[:2], dtype=np.uint8)
    blended = cv2.seamlessClone(patch_bgr, base_bgr, mask, center, mode)
    return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))


def process_image(image_name):
    image_path = os.path.join(IMG_DIR, image_name)
    ann_path = os.path.join(ANN_DIR, f"{image_name}.json")  # assumes e.g. amz_01961.png.json

    if not os.path.exists(ann_path):
        print(f"  No annotation found for {image_name}, skipping")
        return

    image = Image.open(image_path)
    ann = load_cone_boxes(ann_path)
    valid_boxes = [(cls, box) for cls, box in ann if is_big_enough(box)]

    # process smallest cones first, largest last
    valid_boxes.sort(key=lambda item: box_area(item[1]))

    print(f"  {len(ann)} cones total, {len(valid_boxes)} pass size filter")

    edited_cones = []
    for cls, box in valid_boxes:
        if cls not in PROMPTS:
            print(f"    skipping unknown class '{cls}'")
            continue

        x0, y0, x1, y1 = box
        crop = image.crop((x0, y0, x1, y1))

        try:
            edited = edit_small_cone(crop, PROMPTS[cls])
        except Exception as e:
            print(f"    skip cone {cls} {box}: {e}")
            continue

        edited_cones.append((box, edited))

    result_image = image.copy()
    for box, edited in edited_cones:
        result_image = poisson_paste(result_image, edited, box)

    out_path = os.path.join(OUTPUT_DIR, image_name)
    result_image.save(out_path)
    print(f"  saved {out_path}")


# --- main loop ---
image_files = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))
print(f"Found {len(image_files)} images in {IMG_DIR}")

for i, image_name in enumerate(image_files):
    out_path = os.path.join(OUTPUT_DIR, image_name)

    if os.path.exists(out_path):
        print(f"[{i+1}/{len(image_files)}] {image_name} already done, skipping")
        continue

    print(f"[{i+1}/{len(image_files)}] Processing {image_name}...")
    process_image(image_name)

print("Done.")