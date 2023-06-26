from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS
import random


@DATASETS.register_module()
class FishDataset(CocoDataset):

    metainfo = {

        'classes': (
            "fish1",
            "fish2",
            "fish3",
            "fish4",
            "fish5",
            "fish6",
            "fish7",
            "fish8",
            "fish9",
            "fish10",
            "fish11",
            "fish12",
            "fish13",
            "fish14",
            "fish15",
            "fish16",
            "fish17",
        ),
        'palette': [
            (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
            (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
            (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
            (165, 42, 42), (255, 77, 255), (0, 226, 252),
        ]
    }

    def __init__(self, *args, num_shots=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_shots = num_shots

    def _get_support_sample(self, class_id):
        # Get image ids for the given class
        img_ids = self.cat_img_map[class_id]

        # Randomly choose a support image for the class
        support_img_id = random.choice(img_ids)

        # Load support image and annotations
        support_img_info = self.img_infos[support_img_id]
        support_ann_info = self.get_ann_info(support_img_id)

        return support_img_info, support_ann_info

    def __getitem__(self, idx):
        query_img_info, query_ann_info = super().__getitem__(idx)

        support_samples = []
        for class_id in query_ann_info['labels'].unique():
            for _ in range(self.num_shots):
                support_img_info, support_ann_info = self._get_support_sample(class_id)
                support_samples.append((support_img_info, support_ann_info))

        return query_img_info, query_ann_info, support_samples
