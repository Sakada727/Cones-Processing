import json
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROMPTS = {
    "yellow_cone": "Remove the black stripe from the traffic cone and make it a solid, uniform yellow color all the way from top to base. Keep the cone's shape, lighting, shadow, texture, and background exactly the same, only change the black band to match the surrounding yellow.",
    "orange_cone": "Remove the white stripes from the traffic cone and make it a solid, uniform orange color from top to base. Keep the cone's shape, lighting, shadow, texture, base color, and background exactly the same — only change the white bands to match the surrounding orange.",
    "blue_cone": "Remove the white stripe from the traffic cone and make it a solid, uniform blue color from top to base. Keep the cone's shape, lighting, shadow, texture, base color, and background exactly the same — only change the white band to match the surrounding blue.",
}

#CONFIGS -------------------
pref_size = 15
MIN_REQUIRED = 64
TARGET_SIZE = 256
# --------------------------

def show_valid_cones(image, valid_boxes, colors=None): #checking selected cones
    if colors is None:
        colors = {
            "yellow_cone": "yellow",
            "blue_cone": "blue",
            "orange_cone": "orange",
        }

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    for cls, box in valid_boxes:
        x0, y0, x1, y1 = box
        width = x1 - x0
        height = y1 - y0

        rect = patches.Rectangle(
            (x0, y0), width, height,
            linewidth=1, edgecolor=colors.get(cls, "lime"), facecolor="none"
        )
        ax.add_patch(rect)

    ax.set_title(f"{len(valid_boxes)} cones passing size filter")
    plt.axis("off")
    plt.show()

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
    width = x1 - x0
    height = y1 - y0
    return width >= min_size and height >= min_size











IMAGE_NAME = "amz_01961.png"  # just change this, extension included

image_path = f"Images/{IMAGE_NAME}"
ann_path = f"Ann/{IMAGE_NAME}.json"

image = Image.open(image_path)
ann = load_cone_boxes(ann_path)


valid_boxes = [(cls, box) for cls, box in ann if is_big_enough(box, min_size=pref_size)]

print(f"Found {len(ann)} cones, {len(valid_boxes)} pass the size filter")

show_valid_cones(image, valid_boxes)
