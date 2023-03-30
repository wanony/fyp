from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class FishDataset(CocoDataset):

    metainfo = {

        'classes': (
            "clupeid",
            "cod",
            "dab",
            "dogfish",
            "haddock",
            "herring",
            "plaice",
            "prawn",
            "sole",
            "sprat",
            "squid",
            "unk fish",
            "unk flatfish",
            "unk gadoid",
            "unk organism",
            "unk round fish",
            "whiting",
        ),
        'palette': [
            (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
            (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
            (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
            (165, 42, 42), (255, 77, 255), (0, 226, 252),
        ]
    }

    def __init__(self):
        super().__init__(

        )