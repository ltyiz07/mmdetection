from mmdet.apis import init_detector, inference_detector
import mmcv

import os
import glob

# Specify the path to model config and checkpoint file
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'faster_rcnn_kitti_results/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

dir_list = glob.glob("./data/kitti/testing/image_2/*.png")
path = './tutorial_exps2/outputs/'

for dir in dir_list:
    img = mmcv.imread(dir)
    result = inference_detector(model, img)
    model.show_result(img, result, out_file=os.path.join(path, os.path.basename(dir)))


'''
img = mmcv.imread("./data/kitti/testing/image_2/000002.jpeg")
result = inference_detector(model, img)
model.show_result(img, result, out_file='result.jpg')
'''

# test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
    # result = inference_detector(model, frame)
    # model.show_result(frame, result, wait_time=1)