import sys
from pathlib import Path
from itertools import groupby
from PIL import Image


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
from tqdm import tqdm


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
    print([c["name"] for c in coco_dict["categories"]])


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))

    return rle


def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)


def is_crowd(b_mask):
    """Determines if a binary mask represents a crowd.

    Args:
        b_mask: A binary mask.

    Returns:
        A boolean indicating whether the mask represents a crowd.
    """
    # Label the connected regions in the mask
    _, labels = cv2.connectedComponents(b_mask)

    # Find the label with the largest area
    areas = [np.sum(labels == i) for i in range(1, np.max(labels) + 1)]
    if len(areas) == 0:
        # If there are no connected regions, the mask is not a crowd mask
        return False
    max_area = np.max(areas)

    # If the largest connected region covers more than 50% of the mask area, it is a crowd
    if max_area / np.prod(b_mask.shape) > 0.5:
        return True
    else:
        return False


def create_annotation_info(annotation_id, image_id, category_id, binary_mask, crowd,
                           image_size=None, tolerance=2, bounding_box=None):

    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask, image_size)

    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None

    if bounding_box is None:
        bounding_box = mask.toBbox(binary_mask_encoded)

    if crowd:
        crowd = 1
        segmentation = binary_mask_to_rle(binary_mask)
    else:
        crowd = 0
        segmentation = binary_mask_to_polygon(binary_mask, tolerance)
        if not segmentation:
            return None

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": crowd,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    }

    return annotation_info