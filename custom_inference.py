from mmdet.apis import init_detector, inference_detector
import mmcv
from custom_train import cfg

# Specify the path to model config and checkpoint file
# config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'tutorial_exps/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(cfg, checkpoint_file, device='cuda:0')

# test a single image and show the results
# img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
img = mmcv.imread("./data/kitti_tiny/training/image_2/000002.jpeg")
result = inference_detector(model, img)
# visualize the results in a new window
# model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='result.jpg')

# test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
    # result = inference_detector(model, frame)
    # model.show_result(frame, result, wait_time=1)