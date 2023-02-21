from mmengine import Config
from mmengine.runner import Runner
from mmengine.runner import set_random_seed

# Set path to the config file for the model being used
config_file_location = '../mmdetection/configs/condinst/condinst_r50_fpn_ms-poly-90k_coco_instance.py'
# Load the config from the file location
cfg = Config.fromfile(config_file_location)

# Initialise metainfo for config
# Dictionary of classes and palette
# Tuple of Strings of the classes used in the annotation file
cfg.classes = ("clupeid", "cod", "dab", "dogfish", "haddock", "herring", "plaice", "prawn", "sole", "sprat", "squid",
               "unk fish", "unk flatfish", "unk gadoid", "unk organism", "unk round fish", "whiting",)

# List of Tuples of RGB colours, taken from mmdetection/mmdet/datasets/coco.py
palettes = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
            (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
            (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
            (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
            (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
            (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
            (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
            (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
            (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
            (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
            (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
            (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
            (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
            (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
            (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
            (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
            (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
            (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
            (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
            (246, 0, 122), (191, 162, 208)]

# only take the colours needed for our classes
cfg.palette = palettes[:len(cfg.classes)]

cfg.total_epochs = 12

cfg.data_root = 'data/FishDataset'

cfg.data.samples_per_gpu = 2
cfg.data.workers_per_gpu = 8

# TODO data is not the correct work, change this
cfg.data.train.type = 'COCODataset'
cfg.data.train.ann_file = cfg.data_root + 'annotations/MAIS2K_train.json'
cfg.data.train.img_prefix = cfg.data_root + 'train_MAIS2K/raw/'

cfg.data.test.ann_file = cfg.data_root + 'annotations/MAIS2K_test.json'
cfg.data.test.img_prefix = cfg.data_root + 'train_MAIS2K/raw/'

cfg.data.val.ann_file = cfg.data.test.ann_file
cfg.data.val.img_prefix = cfg.data.test.img_prefix

cfg.work_dir = './outputs/CondInst/test_1'

cfg.model.bbox_head.num_classes = 17

# build the runner from config
runner = Runner.from_cfg(cfg)

runner.train()
