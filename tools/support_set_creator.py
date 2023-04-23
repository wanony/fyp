import random
import numpy as np
from pycocotools.coco import COCO
import mmcv
import torch


def create_support_set_and_labels(coco, num_classes, num_shots):
    n_classes = num_classes
    n_examples = num_shots
    path_to_coco_images = '../data/FishDataset/train_MAIS2K/raw/'

    # Select random classes from the COCO dataset
    all_classes = coco.getCatIds()
    selected_classes = random.sample(all_classes, n_classes)

    support_set = []
    support_labels = []

    for label, class_id in enumerate(selected_classes):
        # Get image ids for the selected class
        img_ids = coco.getImgIds(catIds=[class_id])

        # Randomly pick n_examples images for each class
        selected_img_ids = random.sample(img_ids, n_examples)

        for img_id in selected_img_ids:
            img_data = coco.loadImgs([img_id])[0]
            img_path = f"{path_to_coco_images}/{img_data['file_name']}"
            img = mmcv.imread(img_path)  # Read the image using mmcv library

            # Add the image to the support set and its label to support_labels
            support_set.append(img)
            support_labels.append(label)

    # Convert to tensors
    support_set = torch.stack([torch.from_numpy(np.transpose(img, (2, 0, 1))).float() for img in support_set])
    support_labels = torch.tensor(support_labels)

    return support_set, support_labels
