from mmdet.apis import init_detector, inference_detector
import mmcv
from mmdet.utils import register_all_modules
from mmdet.visualization import DetLocalVisualizer

# Specify the path to model config and checkpoint file
# config_file = '../mmdetection/configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco.py'
config_file = 'outputs/20230125_093824/vis_data/config.py'
checkpoint_file = 'outputs/epoch_12.pth'
register_all_modules()

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
# test a single image and show the results
img = mmcv.imread('data/fish/test/FishDataset223_png.rf.335cf10c5756f48fc2a948a9e8a88814.jpg', channel_order='rgb')
# or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)

visualizer_now = DetLocalVisualizer.get_instance(name="You're Awesome :D")
visualizer_now.dataset_meta = model.dataset_meta
visualizer_now.add_datasample('new result', img, data_sample=result, draw_gt=False, wait_time=0,
                              out_file=None, pred_score_thr=0.3,)
visualizer_now.show()
# # visualize the results in a new window
# model.show_result(img, result)
# # or save the visualization results to image files
# model.show_result(img, result, out_file='result.jpg')

# # test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     model.show_result(frame, result, wait_time=1)
