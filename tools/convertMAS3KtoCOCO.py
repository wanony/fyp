import json
from pathlib import Path
import cv2 as cv2
import numpy as np
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert MAS3K Dataset to COCO format"
    )
    parser.add_argument(
        'MAS3K_path',
        type=str,
        help='Path to the MAS3K dataset root directory'
    )
    parser.add_argument(
        'output_location',
        type=str,
        help='Output location for the COCO annotations'
    )
    args = parser.parse_args()

    return args


def create_coco_dict():
    return {
        "images": [],
        "annotations": [],
        "categories": []
    }


def add_all_categories(mas3k_path, coco_dict):
    categories = {}
    for file in tqdm(list(Path(mas3k_path).rglob("*.txt")), desc="Adding categories..."):
        with open(file, 'r') as f:
            for line in f:
                # MAS_Coelenterate_JellyFish_Cam_308
                s = line.split('_')
                # First we take the supercategory
                super_category = s[1]
                # Second we take the category
                category = s[2]
                # Add to temp dict
                categories[category] = super_category

    # Add the main category and count how many
    coco_dict["categories"] = [
        {"id": i, "name": v, "supercategory": None} for i, v in enumerate(set(categories.values()), 1)
    ]
    # Use count to preserve where to start adding categories by ID
    coco_dict["categories"].extend([{"id": i, "name": k, "supercategory": v} for i, (k, v) in
                                    enumerate(categories.items(), len(coco_dict["categories"]) + 1)])


# Define function to add image to COCO dictionary
def add_image(image_path, image_id):
    # Get image shape and pass it into W and H
    img = cv2.imread(str(image_path))
    h, w, _ = img.shape
    return {
        "file_name": image_path.name,
        "height": h,  # Set image height to 512 pixels
        "width": w,  # Set image width to 512 pixels
        "id": image_id
    }


# Define function to add annotation to COCO dictionary
def add_annotation(mask_path, image_id, annotation_id):
    # Load mask image and convert to binary
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]

    # Find contours in mask image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate bounding box coordinates from contours
    if contours:
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
    else:
        # No contours found
        x, y, w, h = 0, 0, 0, 0

    # Create annotation dictionary
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": [x, y, w, h],
        "area": w * h,
        "iscrowd": 0
    }


def populate_annotations(mas3k_path, coco_dict):
    image_id = 1
    annotation_id = 1
    for image_path in tqdm(list(Path(mas3k_path).rglob("*.jpg")), desc="Populating Annotations..."):
        img = add_image(image_path, image_id)
        coco_dict["images"].append(img)
        # Hardcode the image path due to directory layout
        mask_path = f"{image_path.parent.parent}/Mask/{image_path.stem}.png"
        anno = add_annotation(mask_path, image_id, annotation_id)
        coco_dict["annotations"].append(anno)
        image_id += 1
        annotation_id += 1


def main():
    args = parse_args()
    # Set paths to MAS3K dataset and COCO format output file
    mas3k_path = args.MAS3K_path
    mas3k_train = mas3k_path + "train" if mas3k_path[-1] == '/' else '/train'
    mas3k_test = mas3k_path + "test" if mas3k_path[-1] == '/' else '/test'

    # Create COCO format dictionaries
    train_dict = create_coco_dict()
    test_dict = create_coco_dict()

    # Get the classes from MAS dataset
    add_all_categories(mas3k_path, train_dict)
    add_all_categories(mas3k_path, test_dict)

    # Iterate through MAS3K dataset and add images and annotations to COCO dictionaries
    populate_annotations(mas3k_train, train_dict)
    populate_annotations(mas3k_test, test_dict)

    train_dest = mas3k_train + "/annotations_coco.json"
    # Save COCO format output files
    with open(train_dest, "w") as f:
        json.dump(train_dict, f, indent=4)

    test_dest = mas3k_test + "/annotations_coco.json"
    with open(test_dest, "w") as f:
        json.dump(test_dict, f, indent=4)

    print(f"Completed! Annotation files saved to:\nTrain: {train_dest}\nTest: {test_dest}")


if __name__ == '__main__':
    main()
