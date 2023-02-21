from mmdet.apis import init_detector, inference_detector
import mmcv
from mmdet.utils import register_all_modules
from mmdet.visualization import DetLocalVisualizer
from mmengine.runner import Runner
from mmengine import Config
from mmdet.registry import RUNNERS

# Specify the path to model config and checkpoint file
# config_file = '../mmdetection/configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco.py'
config_file = 'rtmdet-ins_x_8xb16-300e_coco.py'
checkpoint_file = 'outputs/feb_10/epoch_80.pth'
register_all_modules()

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
# test a single image and show the results
fish = 'data/FishDataset/val_MAIS2K/raw/00429_D20211130-T140727.410_19462899.jpg'
shrimp = 'data/FishDataset/val_MAIS2K/raw/00001_D20190719-T103259.835_19104538.jpg'
img = mmcv.imread(fish, channel_order='rgb')
# or img = mmcv.imread(img), which will only load it once
cfg = Config.fromfile(config_file)
runner = RUNNERS.build(cfg)
runner.test()
result = inference_detector(model, img)

visualizer_now = DetLocalVisualizer.get_instance(name="You're Awesome :D")
visualizer_now.dataset_meta = model.dataset_meta
visualizer_now.add_datasample('new result', img, data_sample=result, draw_gt=False, wait_time=0,
                              out_file=None, pred_score_thr=0.3,)
visualizer_now.show()
# visualize the results in a new window
# model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='result.jpg')


# # test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     model.show_result(frame, result, wait_time=1)
