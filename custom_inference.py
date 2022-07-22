from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import cv2
from custom_train import cfg, model


model.cfg = cfg

# BGR Image 사용 
img = cv2.imread('./data/kitti_tiny/training/image_2/000068.jpeg')
h, w, c = img.shape


# result = inference_detector(model, img)
# show_result_pyplot(model, img, result)

result = inference_detector(model, img)
# print(result.shape)
for objects in result:
    for obj in objects:
        print(obj)
        xmin  = int(obj[0])
        ymin  = int(obj[1])
        xmax  = int(obj[2])
        ymax  = int(obj[3])
        # img[xmin:xmax, ymin:ymax, [0, 1, 2]] = [0, 255, 0]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
cv2.imwrite("output_1.jpeg", img)