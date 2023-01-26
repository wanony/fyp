from mmengine import Config
from mmengine.runner import Runner
from mmengine.runner import set_random_seed


cfg = Config.fromfile('../mmdetection/configs/rtmdet/rtmdet-ins_x_8xb16-300e_coco.py')

cfg.metainfo = {
    'CLASSES': (
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
    'PALETTE': [
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
        (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
        (165, 42, 42), (255, 77, 255), (0, 226, 252),
    ]
}

cfg.data_root = "data/FishDataset"
cfg.work_dir = './outputs'

# TRAIN DATASET CONFIG
cfg.train_dataloader.dataset.type = "CocoDataset"  # dataset is annotated in COCO format
cfg.train_dataloader.dataset.ann_file = "annotations/MAIS2K_train.json"  # point to the train annotation file
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix.img = 'train_MAIS2K/raw/'
cfg.train_dataloader.dataset.metainfo = cfg.metainfo

# TEST DATASET CONFIG
cfg.test_dataloader.dataset.type = "CocoDataset"
cfg.test_dataloader.dataset.ann_file = "annotations/MAIS2K_test.json"
cfg.test_dataloader.dataset.data_root = cfg.data_root
# cfg.test_dataloader.dataset.data_prefix.img = '' there are none...
cfg.test_dataloader.dataset.metainfo = cfg.metainfo

# VALID DATASET CONFIG
cfg.val_dataloader.dataset.type = "CocoDataset"
# cfg.val_dataloader.dataset.ann_file = 'valid/_annotations.coco.json' there is none
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix.img = 'val_MAIS2K/raw/'
cfg.val_dataloader.dataset.metainfo = cfg.metainfo

# EVALUATOR CONFIG
cfg.test_evaluator = cfg.data_root + "annotations/MAIS2K_test.json"
# Modify num classes of the model in box head and mask head
cfg.model.roi_head.bbox_head.num_classes = 17
# cfg.model.roi_head.mask_head.num_classes = 27

# cfg.load_from = '' can be used once we trained some model

cfg.work_dir = './outputs'

cfg.train_cfg.val_interval = 12
cfg.default_hooks.checkpoint.interval = 12

cfg.optim_wrapper.optimizer.lr = 0.02 / 8
cfg.default_hooks.logger.interval = 10

cfg.seed = 0
set_random_seed(0, deterministic=False)

# We can also use tensorboard to log the training process
cfg.visualizer.vis_backends.append({"type": 'TensorboardVisBackend'})

# build the runner from config
runner = Runner.from_cfg(cfg)

runner.train()
