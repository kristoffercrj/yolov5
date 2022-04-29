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

npz_25 = np.load('labmatching/left-25cm_right-25cm_matches.npz')

imgs = [img_l25, img_r25]

results = model(imgs)
# results.save()
res = results.pandas()
length = 25
# disparity = []
X = []
Y = []
Z = []


# # print(npz_25['keypoints1'][74]) outputs (x, y) coordinates
# # print(kps1[matches[matches_idx[i]]])
 
# bbox_x_l = (res.xyxy[0].iloc[0]['xmin'] + res.xyxy[0].iloc[0]['xmax']) / 2 # Left camera
# bbox_y_l = (res.xyxy[0].iloc[0]['ymin'] + res.xyxy[0].iloc[0]['ymax']) / 2

for img_nr in range(0, 2, 2):
    # Round the values to nearest integer because OpenCV is scared of float values
    xmax = res.xyxy[img_nr].iloc[0]['xmax']
    xmax = int(round(xmax))
    ymax = res.xyxy[img_nr].iloc[0]['ymax']
    ymax = int(round(ymax))
    xmin = res.xyxy[img_nr].iloc[0]['xmin']
    xmin = int(round(xmin))
    ymin = res.xyxy[img_nr].iloc[0]['ymin']
    ymin = int(round(ymin))

    # The indices for the ms in keypoints0
    # The value of the indice is the indice in keypoints1
    ms_idx = np.where(npz_25['matches']>-1)[0]
    ms = npz_25['matches']
    kps0 = npz_25['keypoints0']
    kps1 = npz_25['keypoints1']
    print(res.xyxy[img_nr].iloc[0]['xmin'])
    print(kps0[ms_idx[0]][1] > res.xyxy[img_nr].iloc[0]['ymin'])

#     for i in range(0, len(ms_idx)):
#         if (kps0[ms_idx[i]][1] > res.xyxy[img_nr].iloc[0]['xmin']) and \
#             (kps0[ms_idx[i]][1] < res.xyxy[img_nr].iloc[0]['xmax']) and \
#             (kps0[ms_idx[i]][0] > res.xyxy[img_nr].iloc[0]['ymin']) and \
#             (kps0[ms_idx[i]][0] < res.xyxy[img_nr].iloc[0]['ymax']):
#             x_disparity = (kps0[ms_idx[i]][0] - kps1[ms[ms_idx[i]]][0])
#             y_disparity = (kps0[ms_idx[i]][1] - kps1[ms[ms_idx[i]]][1])

#             Z.append((baseline * f) / x_disparity / px_size)
#             X.append(x_disparity * Z * px_size / f)
#             Y.append(y_disparity * Z * px_size / f)

#             # disparity = (bbox_x_l - bbox_x_r) * px_size
#             # Z = (baseline * f) / (bbox_x_l - bbox_x_r) / px_size
#             # X = (bbox_x_l - bbox_x_r) * Z * px_size / f
#             # Y = (bbox_y_l - bbox_y_r) * Z * px_size / f
#     Z = sum(Z) / len(Z)
#     X = sum(X) / len(Y)
#     Y = sum(Y) / len(Y)

#     distance = np.sqrt(X**2 + Y**2 + Z**2)
#     print(distance)
                        

#     # bbox_x_l = (res.xyxy[img_nr].iloc[0]['xmin'] + res.xyxy[img_nr].iloc[0]['xmax']) / 2 # Left camera
#     # bbox_y_l = (res.xyxy[img_nr].iloc[0]['ymin'] + res.xyxy[img_nr].iloc[0]['ymax']) / 2

#     # bbox_x_r = (res.xyxy[img_nr + 1].iloc[0]['xmin'] + res.xyxy[img_nr + 1].iloc[0]['xmax']) / 2 # Right camera
#     # bbox_y_r = (res.xyxy[img_nr + 1].iloc[0]['ymin'] + res.xyxy[img_nr + 1].iloc[0]['ymax']) / 2

#     # Creates a bounding box with information generated from Yolov5 bounding boxes
#     imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin, ymax), (xmax, ymin), (0, 0, 255), 2)
#     font = cv.FONT_HERSHEY_SIMPLEX
#     imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(distance, 3)} m', (xmin, ymin - 5), font, 0.8, (0, 0, 255), 2)
#     cv.imwrite(f'../../result_images/estimated_{str(length)}cm.png', imgs[img_nr])
# #    length = length + 25


# # print(npz_25['ms'])
# # print(npz_25['keypoints0'][204])
# print(npz_25['keypoints1'][74])
# print(npz_25['keypoints1'][102])
# # print(npz_25['keypoints1'][324])
# # print(npz_25['keypoints1'][311])
# # print(npz_25['keypoints0'][225][1], npz_25['keypoints0'][225][0])

# print(np.where(npz_25['ms']>-1)[0])

# imgl = cv.imread('./labmatching/lab25/left-25cm.jpg')
# imgr = cv.imread('./labmatching/lab25/right-25cm.jpg')
# # imgl[188, 751] = [0, 0, 255]
# # imgl[207, 736] = [0, 0, 255]
# imgr[139, 444] = [0, 0, 255]
# imgr[162, 436] = [0, 0, 255]
# # cv.circle(imgr, (302, 1048), 1, [0, 0, 255], -1)
# # cv.circle(imgr, (302, 1048), 1, [0, 0, 255], -1)
# # cv.circle(imgl, (188, 751), 5, (0, 0, 255), -1)
# # img = np.hstack((imgl, imgr))

# cv.imshow('img', imgr)
# cv.waitKey(5000)