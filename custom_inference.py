from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import cv2
from custom_train import cfg, model
from mmdet.models.detectors import base


model.cfg = cfg

# BGR Image 사용 
img = cv2.imread('./data/kitti_tiny/training/image_2/000007.jpeg')
h, w, c = img.shape


result = inference_detector(model, img)
# base.show_result(model, img, result)
print(result)
img = model.show_result(img, result, out_file = "./output_img.jpeg")
# cv2.imwrite("output_img.jpeg", img)

# result = inference_detector(model, img)
# # print(result.shape)
# for objects in result:
#     for obj in objects:
#         print(obj)
#         a  = int(obj[0])
#         b  = int(obj[1])
#         c  = int(obj[2])
#         d  = int(obj[3])
#         # img[xmin:xmax, ymin:ymax, [0, 1, 2]] = [0, 255, 0]
#         cv2.rectangle(img, (a, b), (c, d), (255,0,0), 2)
# cv2.imwrite("output_1.jpeg", img)