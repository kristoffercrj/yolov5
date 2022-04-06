import cv2 as cv
import numpy as np
import torch
import pandas

from camera_utils.camera_parameters_lab import *
from utils.plots import Annotator

# Load yolov5 model
model = torch.hub.load('./', 'custom', path='./checkpoints/best-yolov5-lab', source='local', force_reload=True)

# Camera parameters
P1, P2, dist_coeff1, dist_coeff2 = camera_params_lab()
baseline = 230e-03 # Baseline of the camera
f = 1.28e-03 # Focal length of the camera
px_size = 3.75e-06 # Pixel size

# Input images
img_l25 = cv.imread('../../lab-dataset/test-imgs/left-25cm.jpg')
img_r25 = cv.imread('../../lab-dataset/test-imgs/right-25cm.jpg')

img_l50 = cv.imread('../../lab-dataset/test-imgs/left-50cm.jpg')
img_r50 = cv.imread('../../lab-dataset/test-imgs/right-50cm.jpg')

img_l75 = cv.imread('../../lab-dataset/test-imgs/left-75cm.jpg')
img_r75 = cv.imread('../../lab-dataset/test-imgs/right-75cm.jpg')

img_l100 = cv.imread('../../lab-dataset/test-imgs/left-100cm.jpg')
img_r100 = cv.imread('../../lab-dataset/test-imgs/right-100cm.jpg')

img_l125 = cv.imread('../../lab-dataset/test-imgs/left-125cm.jpg')
img_r125 = cv.imread('../../lab-dataset/test-imgs/right-125cm.jpg')

img_l150 = cv.imread('../../lab-dataset/test-imgs/left-150cm.jpg')
img_r150 = cv.imread('../../lab-dataset/test-imgs/right-150cm.jpg')

img_l175 = cv.imread('../../lab-dataset/test-imgs/left-175cm.jpg')
img_r175 = cv.imread('../../lab-dataset/test-imgs/right-175cm.jpg')


imgs = [img_l25, img_r25, img_l50, img_r50, img_l75, img_r75, img_l100, img_r100, img_l125, img_r125, img_l150, img_r150, img_l175, img_r175]

results = model(imgs)
res = results.pandas()
# print(res.xyxy[0].iloc[0]['xmin'])

# # print(res.xyxy[1].iloc[1])
length = 25

for img_nr in range(0, 14, 2):
    # Round the values to nearest integer because OpenCV is scared of float values
    xmax = res.xyxy[img_nr].iloc[0]['xmax']
    xmax = int(round(xmax))
    ymax = res.xyxy[img_nr].iloc[0]['ymax']
    ymax = int(round(ymax))
    xmin = res.xyxy[img_nr].iloc[0]['xmin']
    xmin = int(round(xmin))
    ymin = res.xyxy[img_nr].iloc[0]['ymin']
    ymin = int(round(ymin))

    bbox_x_l = (res.xyxy[img_nr].iloc[0]['xmin'] + res.xyxy[img_nr].iloc[0]['xmax']) / 2 # Left camera
    bbox_y_l = (res.xyxy[img_nr].iloc[0]['ymin'] + res.xyxy[img_nr].iloc[0]['ymax']) / 2

    bbox_x_r = (res.xyxy[img_nr + 1].iloc[0]['xmin'] + res.xyxy[img_nr + 1].iloc[0]['xmax']) / 2 # Right camera
    bbox_y_r = (res.xyxy[img_nr + 1].iloc[0]['ymin'] + res.xyxy[img_nr + 1].iloc[0]['ymax']) / 2

    disparity = (bbox_x_l - bbox_x_r) * px_size
    Z = (baseline * f) / (bbox_x_l - bbox_x_r) / px_size
    X = (bbox_x_l - bbox_x_r) * Z * px_size / f
    Y = (bbox_y_l - bbox_y_r) * Z * px_size / f
    distance = np.sqrt(X**2 + Y**2 + Z**2)
    print(distance)

    # # Creates a bounding box with information generated from Yolov5 bounding boxes
    # imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin, ymax), (xmax, ymin), (0, 0, 255), 2)
    # font = cv.FONT_HERSHEY_SIMPLEX
    # imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(distance, 3)} m', (xmin, ymin - 5), font, 0.8, (0, 0, 255), 2)
    # cv.imwrite(f'../../result_images/estimated_{str(length)}cm.png', imgs[img_nr])
    # length = length + 25

# for img_nr in range(0, 14, 2):
#     # Round the values to nearest integer because OpenCV is scared of float values
#     xmax = res.xyxy[img_nr].iloc[0]['xmax']
#     xmax = int(round(xmax))
#     ymax = res.xyxy[img_nr].iloc[0]['ymax']
#     ymax = int(round(ymax))
#     xmin = res.xyxy[img_nr].iloc[0]['xmin']
#     xmin = int(round(xmin))
#     ymin = res.xyxy[img_nr].iloc[0]['ymin']
#     ymin = int(round(ymin))

#     bbox_x_l = (res.xyxy[img_nr].iloc[0]['xmin'] + res.xyxy[img_nr].iloc[0]['xmax']) / 2 # Left camera
#     bbox_y_l = (res.xyxy[img_nr].iloc[0]['ymin'] + res.xyxy[img_nr].iloc[0]['ymax']) / 2

#     bbox_x_r = (res.xyxy[img_nr + 1].iloc[0]['xmin'] + res.xyxy[img_nr + 1].iloc[0]['xmax']) / 2 # Right camera
#     bbox_y_r = (res.xyxy[img_nr + 1].iloc[0]['ymin'] + res.xyxy[img_nr + 1].iloc[0]['ymax']) / 2

#     left_coords = np.array([
#         [bbox_x_l],
#         [bbox_y_l]])

#     right_coords = np.array([
#         [bbox_x_r],
#         [bbox_y_r]])

#     # Triangulate
#     points = cv.triangulatePoints(P1, P2, left_coords, right_coords)

#     # Normalize
#     points /= points[3]
#     points = [val for sublist in points for val in sublist] # Converts nested list to array

#     # Calculate the euclidian distance
#     X = points[0]
#     Y = points[1]
#     Z = points[2]

#     distance1 = np.sqrt(X**2 + Y**2 + Z**2) / 1000 # Convert it to meter

#     print(f'{distance1}')
#     # Creates a bounding box with information generated from Yolov5 bounding boxes
#     imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin, ymax), (xmax, ymin), (0, 0, 255), 2)
#     font = cv.FONT_HERSHEY_SIMPLEX
#     imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(distance1, 3)} m', (xmin, ymin - 5), font, 0.8, (0, 0, 255), 2)
#     cv.imwrite(f'../../result_images/opencv_estimated_{str(length)}cm.png', imgs[img_nr])
#     length = length + 25





