import json
import sys
import pandas

try:
    import cv2 as cv2
except ImportError:
    print("Failed to import cv2, please install this requirement")
    sys.exit(1)
try:
    import numpy as np
except ImportError:
    print("Failed to import numpy, please install this requirement")
    sys.exit(1)
try:
    from pycocotools import mask
except ImportError:
    print("Failed to import pycocotools, please install this requirement")
    sys.exit(1)
try:
    from skimage import measure
except ImportError:
    print("Failed to import skimage, please install this requirement")
    sys.exit(1)
import argparse
from convert_from_coco import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert DeepFish Dataset to COCO format"
    )
    parser.add_argument(
        'DeepFish_path',
        type=str,
        help='Path to the DeepFish dataset root directory'
    )
    args = parser.parse_args()

    return args


def populate_annotations(mas3k_path, coco_dict):
    image_id = 1
    annotation_id = 1
    images = list(Path(mas3k_path).rglob("*.jpg"))
    for original_image_path in tqdm(images, desc="Populating Annotations..."):
        img = Image.open(original_image_path)
        w, h = img.size[0], img.size[1]  # extract width and height
        # annotate and add the original image
        img_anno = {
            "file_name": original_image_path.name,
            "height": h,
            "width": w,
            "id": image_id
        }
        coco_dict["images"].append(img_anno)

        # get path to mask of original image as per directory layout
        mask_path = f"{original_image_path.parent.parent}/Mask/{original_image_path.stem}.png"
        img = Image.open(mask_path)
        # Load mask image and convert to binary
        binary_mask = np.asarray(img.convert('1')).astype(np.uint8)

        cat_id = \
            [c_id["id"] for c_id in coco_dict["categories"] if
             c_id["name"] == original_image_path.name.split("_")[1]][0]

        annotation_info = create_annotation_info(annotation_id=annotation_id,
                                                 image_id=image_id,
                                                 category_id=cat_id,
                                                 binary_mask=binary_mask,
                                                 crowd=is_crowd(binary_mask),
                                                 image_size=img.size,
                                                 tolerance=2,
                                                 bounding_box=None)

        if annotation_info is not None:
            coco_dict["annotations"].append(annotation_info)

        annotation_id += 1

        # increment image and annotation id by 1
        image_id += 1


def main():
    args = parse_args()
    # Set paths to DeepFish dataset and COCO format output file
    df_path = args.DeepFish_path + ("Segmentation" if args.DeepFish_path[-1] == '/' else '/Segmentation')
    df_seg_train_csv = df_path + "/train.csv"
    df_seg_test_csv = df_path + "/test.csv"
    train_files = []
    test_files = []

    # get the train and test files from the CSV data
    df = pandas.read_csv(df_seg_train_csv)

    # Create COCO format dictionaries
    train_dict = create_coco_dict()
    test_dict = create_coco_dict()

    # Get the classes from MAS dataset
    print("Train Categories")
    add_all_categories(mas3k_path, train_dict)
    print("Test Categories")
    add_all_categories(mas3k_path, test_dict)

    # Iterate through MAS3K dataset and add images and annotations to COCO dictionaries
    print("Train Annotations")
    populate_annotations(mas3k_train, train_dict)
    print("Test Annotations")
    populate_annotations(mas3k_test, test_dict)

    train_dest = mas3k_train + "/annotations_coco.json"
    # Save COCO format output files
    with open(train_dest, "w") as f:
        json.dump(train_dict, f, indent=4)

    test_dest = mas3k_test + "/annotations_coco.json"
    with open(test_dest, "w") as f:
        json.dump(test_dict, f, indent=4)

    print(f"Completed! Annotation files saved to:\nTrain: {train_dest}\nTest: {test_dest}")


if __name__ == "__main__":
    # Get the standard COCO JSON format
    main()