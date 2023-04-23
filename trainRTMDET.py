from mmdet.apis import init_detector, inference_detector
import mmcv
from mmdet.utils import register_all_modules
from mmdet.visualization import DetLocalVisualizer
from mmengine.runner import Runner
from mmengine import Config
from mmdet.registry import RUNNERS

# Specify the path to model config and checkpoint file
# config_file = '../mmdetection/configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco.py'
config_file = 'condinst_r50_fpn_ms-poly-90k_coco_instance.py'
# checkpoint_file = 'outputs/RTMDet/epoch_80.pth'
checkpoint_file = 'outputs/CondInst/test_1/iter_90000.pth'
register_all_modules()

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
# test a single image and show the results
fish = '../data/FishDataset/val_MAIS2K/raw/00429_D20211130-T140727.410_19462899.jpg'
shrimp = '../data/FishDataset/val_MAIS2K/raw/00001_D20190719-T103259.835_19104538.jpg'
img = mmcv.imread(fish, channel_order='rgb')  # specify rgb otherwise colours are wrong for humans
# or img = mmcv.imread(img), which will only load it once
cfg = Config.fromfile(config_file)
runner = RUNNERS.build(cfg)
# This will run the test part, so will take a short time and display AP results etc.
# runner.test()
result = inference_detector(model, img)

visualizer_now = DetLocalVisualizer(alpha=0.3).get_instance(name="You're Awesome :D")
visualizer_now.dataset_meta = model.dataset_meta
visualizer_now.add_datasample('new result', img, data_sample=result, draw_gt=False, wait_time=0,
                              out_file=None, pred_score_thr=0.3,)
visualizer_now.show()
# visualize the results in a new window
# model.show_result(img, result)
# or save the visualization results to image files
# model.show_result(img, result, out_file='result.jpg')
visualizer_now.close()


# # test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     model.show_result(frame, result, wait_time=1)
