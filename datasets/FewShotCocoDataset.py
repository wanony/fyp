from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS
import numpy as np
import random


@DATASETS.register_module()
class FewShotCocoDataset(CocoDataset):

    def __init__(self, *args, n_shots=5, n_queries=5, n_ways=5, **kwargs):
        super(FewShotCocoDataset, self).__init__(*args, **kwargs)
        self.n_shots = n_shots
        self.n_queries = n_queries
        self.n_ways = n_ways

    def get_support_query_data(self):
        support_data = []
        query_data = []

        # Iterate through classes
        for class_id in self.cat_ids:
            # Get image ids for the given class
            img_ids = list(self.cat_img_map[class_id])

            # Shuffle the image ids
            random.shuffle(img_ids)

            # Extract support and query image ids
            support_img_ids = img_ids[:self.n_shots]
            query_img_ids = img_ids[self.n_shots:self.n_shots + self.n_queries]

            # Add support and query data
            support_data.extend(self.get_data_by_img_ids(support_img_ids))
            query_data.extend(self.get_data_by_img_ids(query_img_ids))

        return support_data, query_data

    def get_data_by_img_ids(self, img_ids):
        data = []
        for img_id in img_ids:
            data_info = [info for info in self.data_list if info['img_id'] == img_id]
            if data_info:
                data.append(data_info[0])
        return data

