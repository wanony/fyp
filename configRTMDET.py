from mmengine import Config
from mmengine.runner import Runner
from mmengine.runner import set_random_seed

# Set path to the config file for the model being used
config_file_locations = '../mmdetection/configs/rtmdet/rtmdet-ins_x_8xb16-300e_coco.py'
# Load the config from the file location
cfg = Config.fromfile(config_file_locations)

# Initialise metainfo for config
# Dictionary of classes and palette
# Tuple of Strings of the classes used in the annotation file
classes = ("clupeid", "cod", "dab", "dogfish", "haddock", "herring", "plaice", "prawn", "sole", "sprat", "squid",
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
palette = palettes[:len(classes)]

# finally, set the metainfo in the config
cfg.metainfo = {
    'CLASSES': classes,
    'PALETTE': palette
}

# set the root of our dataset
cfg.data_root = "data/FishDataset"
# set a directory to output our model and config
cfg.work_dir = './outputs/test_2_8_2023'

# TRAIN DATASET CONFIG
cfg.train_dataloader.dataset.type = "CocoDataset"  # dataset is annotated in COCO format
cfg.train_dataloader.dataset.ann_file = "annotations/MAIS2K_train.json"  # point to the train annotation file
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix.img = 'train_MAIS2K/raw/'
cfg.train_dataloader.dataset.metainfo = cfg.metainfo
cfg.train_dataloader.batch_size = 2  # for single GPU, set to 2
cfg.train_dataloader.num_workers = 8

# TEST DATASET CONFIG
cfg.test_dataloader.dataset.type = "CocoDataset"
cfg.test_dataloader.dataset.ann_file = "annotations/MAIS2K_test.json"
cfg.test_dataloader.dataset.data_root = cfg.data_root
cfg.test_dataloader.dataset.data_prefix.img = cfg.train_dataloader.dataset.data_prefix.img  # set to training dataset
cfg.test_dataloader.dataset.metainfo = cfg.metainfo
cfg.test_dataloader.batch_size = 2  # for single GPU, set to 2
cfg.test_dataloader.num_workers = 8


# VALID DATASET CONFIG
cfg.val_dataloader.dataset.type = "CocoDataset"
cfg.val_dataloader.dataset.ann_file = cfg.test_dataloader.dataset.ann_file  # there is none, so point to test
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix.img = 'val_MAIS2K/raw/'
cfg.val_dataloader.dataset.metainfo = cfg.metainfo
cfg.val_dataloader.batch_size = 2  # for single GPU, set to 2
cfg.val_dataloader.num_workers = 8

# EVALUATOR CONFIG
cfg.test_evaluator.ann_file = cfg.data_root + "/annotations/MAIS2K_test.json"
cfg.val_evaluator.ann_file = cfg.test_evaluator.ann_file
# Modify num classes of the model in box head and mask head
cfg.model.bbox_head.num_classes = 17

# cfg.load_from = 'outputs/test/epoch_1.pth'

cfg.train_cfg.val_interval = 10
cfg.train_cfg.max_epochs = 30
cfg.default_hooks.checkpoint.interval = 10
#
# cfg.optim_wrapper.optimizer.lr = 0.02 / 8
# cfg.default_hooks.logger.interval = 10

cfg.seed = 0
set_random_seed(0, deterministic=False)

# We can also use tensorboard to log the training process
cfg.visualizer.vis_backends.append({"type": 'TensorboardVisBackend'})

# build the runner from config
runner = Runner.from_cfg(cfg)

runner.train()
