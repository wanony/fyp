import json
from pathlib import Path
import cv2
import numpy as np

# Set paths to MAS3K dataset and COCO format output file
mas3k_path = "path/to/mas3k/dataset"
coco_path = "path/to/coco/format/output/file"

# Create COCO format dictionary
coco_dict = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "object"}
    ]
}


# Define function to add image to COCO dictionary
def add_image(image_path, image_id):
    image = {
        "file_name": image_path.name,
        "height": 512,  # Set image height to 512 pixels
        "width": 512,  # Set image width to 512 pixels
        "id": image_id
    }
    coco_dict["images"].append(image)


# Define function to add annotation to COCO dictionary
def add_annotation(mask_path, image_id, annotation_id):
    # Load mask image and convert to binary
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]

    # Find contours in mask image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate bounding box coordinates from contours
    x, y, w, h = cv2.boundingRect(np.concatenate(contours))

    # Create annotation dictionary
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": [x, y, w, h],
        "area": w * h,
        "iscrowd": 0
    }
    coco_dict["annotations"].append(annotation)


# Iterate through MAS3K dataset and add images and annotations to COCO dictionary
image_id = 1
annotation_id = 1
for image_path in Path(mas3k_path).rglob("*.jpg"):
    add_image(image_path, image_id)
    mask_path = image_path.with_suffix(".png")
    add_annotation(mask_path, image_id, annotation_id)
    image_id += 1
    annotation_id += 1

# Save COCO format output file
with open(coco_path, "w") as f:
    json.dump(coco_dict, f)
