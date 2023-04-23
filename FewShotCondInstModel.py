import torch
import torch.nn as nn
import torch.optim as optim
# from mmcv.runner import load_checkpoint
from mmdet.apis import init_detector, inference_detector
from mmengine import Config
from mmengine.runner import Runner
from fyp.runners import maml_runner
from mmengine.runner import set_random_seed
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mmdet.models import CondInst
from mmdet.registry import RUNNERS
from mmdet.registry import MODELS
from datasets.FewShotCocoDataset import FewShotCocoDataset


@MODELS.register_module()
class ProtoNetCondInst(CondInst):
    def __init__(self, 
                 data_preprocessor,
                 backbone,
                 neck,
                 mask_head,
                 bbox_head,
                 test_cfg):
        super(ProtoNetCondInst, self).__init__(
            data_preprocessor=data_preprocessor,
            backbone=backbone,
            neck=neck,
            mask_head=mask_head,
            bbox_head=bbox_head,
            test_cfg=test_cfg,
        )

    def forward(self, x, support_set, support_labels):
        # Extract feats from support set
        support_features = self.model.backbone(support_set)

        # Calculate the class prototypes by averaging the features of support samples belonging to the same class
        num_classes = support_labels.max().item() + 1
        class_prototypes = torch.zeros((num_classes, support_features.size(1)))

        for c in range(num_classes):
            class_indices = (support_labels == c).nonzero(as_tuple=True)[0]
            class_features = support_features[class_indices].mean(dim=0)
            class_prototypes[c] = class_features

        # Extract feats from input using model backbone
        query_features = self.model.backbone(x)

        # Calculate the distance between the query features and the class prototypes
        distances = F.pairwise_distance(query_features.unsqueeze(1), class_prototypes.unsqueeze(0))

        # Predict the class labels based on the distances
        predictions = torch.argmin(distances, dim=-1)

        # Get the instance segmentation results
        instance_results = self.model.simple_test(query_features, predictions)

        return instance_results


if __name__ == '__main__':
    # TODO implement argparse for config
    # Step 1: Configuring the model and dataset
    cfg = Config.fromfile('condinst_r50_fpn_ms-poly-90k_coco_instance_few_shot.py')

    # Step 2: Build the model
    runner = maml_runner.from_cfg(cfg)
    # Load a pre-trained backbone
    # TODO if checkpoint:
    #   load_checkpoint(model.backbone, 'path/to/pretrained/backbone/checkpoint.pth', map_location='cpu')
    set_random_seed(0, deterministic=False)
    # Step 4: Replace the original model with the few-shot learning model
    runner.train()