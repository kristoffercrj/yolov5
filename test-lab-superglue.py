import cv2 as cv
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas

from camera_utils.camera_parameters_lab import *
from utils.plots import Annotator

# Load yolov5 model
model = torch.hub.load('./', 'custom', path='./checkpoints/best-yolov5-lab', source='local', force_reload=True)

# Camera parameters
P1, P2, dist_coeff1, dist_coeff2 = camera_params_lab()
baseline = 230.0e-03 # Baseline of the camera
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
# results.save()
res = results.pandas()
length = 25

X = []
Y = []
Z = []

Z_points = []
X_points = []
Y_points = []

# print(npz_25['keypoints1'][74]) outputs (x, y) coordinates
# print(kps1[matches[matches_idx[i]]])

for img_nr in range(0, len(imgs), 2):
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
    npz_file = np.load(f'labmatching/left-{length}cm_right-{length}cm_matches.npz') # Load .npz file from superglue matching 
    ms_idx = np.where(npz_file['matches']>-1)[0]
    ms = npz_file['matches']
    kps0 = npz_file['keypoints0']
    kps1 = npz_file['keypoints1']

    print(f'Found {len(ms_idx)} matches!')

    for i in range(0, len(ms_idx)):
        kp0_x = kps0[ms_idx[i]][0] # x-coordinate of keypoint
        kp0_y = kps0[ms_idx[i]][1] # y-coordinate of keypoint

        if ((kp0_y > res.xyxy[img_nr].iloc[0]['ymin']) and \
            (kp0_y < res.xyxy[img_nr].iloc[0]['ymax']) and \
            (kp0_x > res.xyxy[img_nr].iloc[0]['xmin']) and \
            (kp0_x < res.xyxy[img_nr].iloc[0]['xmax'])):
            x_disparity = (kps0[ms_idx[i]][0] - kps1[ms[ms_idx[i]]][0])
            y_disparity = (kps0[ms_idx[i]][1] - kps1[ms[ms_idx[i]]][1])

            Z_t = (baseline * f) / x_disparity / px_size # Temporary value
            # round(Z_t, 3)
            X_t = (x_disparity * Z_t * px_size / f)
            # round(X_t, 3)
            Y_t = (y_disparity * Z_t * px_size / f)
            # round(Y_t, 3)

            Z = np.append(Z, Z_t)
            print(len(Z))
            X = np.append(X, X_t)
            # print(X)
            Y = np.append(Y, Y_t)
                # print(Y)
        
    Z = sum(Z) / len(Z)
    X = sum(X) / len(X)
    Y = sum(Y) / len(Y)
    
    Z_points.append(Z)
    X_points.append(X)
    Y_points.append(Y)

    distance = np.sqrt(X**2 + Y**2 + Z**2)
    print(f'Distance{str(length)} is: {distance}')
                        
    # Creates a bounding box with information generated from Yolov5 bounding boxes
    imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin, ymax), (xmax, ymin), (0, 0, 255), 2)
    font = cv.FONT_HERSHEY_SIMPLEX
    imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(distance, 3)} m', (xmin, ymin - 5), font, 0.8, (0, 0, 255), 2)
    cv.imwrite(f'../../result_imgs_feature/estimated_{str(length)}cm.png', imgs[img_nr])
    length += 25
# print(X)

# # Plot 3D trajectory
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(X_points, Y_points, Z_points)
# n = [1, 2, 3, 4, 5, 6, 7]
# for o in range(len(n)):
#     ax.text(X_points[o], Y_points[o], Z_points[o], f'{str(o)}')

# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')

# plt.savefig(f'../../result_imgs_feature/3D-trajectory.png')



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