import cv2 as cv
import numpy as np
import torch
import pandas
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from camera_utils.camera_parameters_sintef import *

# Load yolov5 model
model = torch.hub.load('./', 'custom', path='./checkpoints/best-yolov5l-sintef-640', source='local')

# Camera parameters
P1, P2, dist_coeff1, dist_coeff2 = camera_params_sintef()
baseline = 230e-03 # Baseline of the camera [m]
f = 1.28e-03 # Focal length of the camera [m]
px_size = 3.75e-06 # Pixel size of camera in [m]
fps = 30

start_img = 200
end_img = 205

imgs = []
# All even numbers are left images
# All odd numbers are right images
for i in range(start_img, end_img):
    im = cv.imread(f'../../test-images-sintef/1200rpm/left/l1200-{i}.jpg')
    imgs.append(im)
    im = cv.imread(f'../../test-images-sintef/1200rpm/right/r1200-{i}.jpg')
    imgs.append(im)

results = model(imgs)
res = results.pandas()

# results.save() # Show predictions
print(res.xyxy)

X_points1 = []
Y_points1 = []
Z_points1 = []

X_points2 = []
Y_points2 = []
Z_points2 = []

img_nr = 0
# Middle of the bounding boxes
# Fish 1
x1_l1 = (res.xyxy[img_nr].iloc[0]['xmin'] + res.xyxy[img_nr].iloc[0]['xmax']) / 2 # Left camera
y1_l1 = (res.xyxy[img_nr].iloc[0]['ymin'] + res.xyxy[img_nr].iloc[0]['ymax']) / 2 # Left camera

x1_r1 = (res.xyxy[img_nr + 1].iloc[0]['xmin'] + res.xyxy[img_nr + 1].iloc[0]['xmax']) / 2 # Right camera
y1_r1 = (res.xyxy[img_nr + 1].iloc[0]['ymin'] + res.xyxy[img_nr + 1].iloc[0]['ymax']) / 2 # Right camera

# Fish 2
x1_l1 = (res.xyxy[img_nr].iloc[0]['xmin'] + res.xyxy[img_nr].iloc[0]['xmax']) / 2 # Left camera
y1_l1 = (res.xyxy[img_nr].iloc[0]['ymin'] + res.xyxy[img_nr].iloc[0]['ymax']) / 2 # Left camera

x1_r1 = (res.xyxy[img_nr + 1].iloc[0]['xmin'] + res.xyxy[img_nr + 1].iloc[0]['xmax']) / 2 # Right camera
y1_r1 = (res.xyxy[img_nr + 1].iloc[0]['ymin'] + res.xyxy[img_nr + 1].iloc[0]['ymax']) / 2 # Right camera

# Triangulation
Z11 = (baseline * f) / (x1_l1 - x1_r1) / px_size
X11 = (x1_l1 - x1_r1) * px_size * Z11 / f
Y11 = (y1_l1 - y1_r1) * px_size * Z11 / f

Z12 = (baseline * f) / (x1_l1 - x1_r1) / px_size
X12 = (x1_l1 - x1_r1) * px_size * Z11 / f
Y12 = (y1_l1 - y1_r1) * px_size * Z11 / f

Z_points1.append(Z11)
X_points1.append(X11)
Y_points1.append(Y11)

Z_points2.append(Z12)
X_points2.append(X12)
Y_points2.append(Y12)

# Calculate the euclidan distance
dist11 = np.sqrt(X11**2 + Y11**2 + Z11**2)
print(dist11)

dist12 = np.sqrt(X12**2 + Y12**2 + Z12**2)
print(dist12)

# img_nr = 2
# # Middle of the bounding boxes
# x2_l1 = (res.xyxy[img_nr].iloc[0]['xmin'] + res.xyxy[img_nr].iloc[0]['xmax']) / 2 # Left camera
# y2_l1 = (res.xyxy[img_nr].iloc[0]['ymin'] + res.xyxy[img_nr].iloc[0]['ymax']) / 2 # Left camera

# x2_r1 = (res.xyxy[img_nr + 1].iloc[0]['xmin'] + res.xyxy[img_nr + 1].iloc[0]['xmax']) / 2 # Right camera
# y2_r1 = (res.xyxy[img_nr + 1].iloc[0]['ymin'] + res.xyxy[img_nr + 1].iloc[0]['ymax']) / 2 # Right camera

# # Triangulation
# Z2 = (baseline * f) / (x2_l1 - x2_r1) / px_size
# X2 = (x2_l1 - x2_r1) * px_size * Z2 / f
# Y2 = (y2_l1 - y2_r1) * px_size * Z2 / f

# # Calculate the euclidan distance
# dist2 = np.sqrt(X2**2 + Y2**2 + Z2**2)
# print(dist2)

# Z_points.append(Z2)
# X_points.append(X2)
# Y_points.append(Y2)

# img_nr = 4
# # Middle of the bounding boxes
# x3_l1 = (res.xyxy[img_nr].iloc[1]['xmin'] + res.xyxy[img_nr].iloc[1]['xmax']) / 2 # Left camera
# y3_l1 = (res.xyxy[img_nr].iloc[1]['ymin'] + res.xyxy[img_nr].iloc[1]['ymax']) / 2 # Left camera

# x3_r1 = (res.xyxy[img_nr + 1].iloc[1]['xmin'] + res.xyxy[img_nr + 1].iloc[1]['xmax']) / 2 # Right camera
# y3_r1 = (res.xyxy[img_nr + 1].iloc[1]['ymin'] + res.xyxy[img_nr + 1].iloc[1]['ymax']) / 2 # Right camera

# # Triangulation
# Z3 = (baseline * f) / (x3_l1 - x3_r1) / px_size
# X3 = (x3_l1 - x3_r1) * px_size * Z3 / f
# Y3 = (y3_l1 - y3_r1) * px_size * Z3 / f

# Z_points.append(Z3)
# X_points.append(X3)
# Y_points.append(Y3)

# # Calculate the euclidan distance
# dist3 = np.sqrt(X3**2 + Y3**2 + Z3**2)
# print(dist3)

# img_nr = 6
# # Middle of the bounding boxes
# x4_l1 = (res.xyxy[img_nr].iloc[1]['xmin'] + res.xyxy[img_nr].iloc[1]['xmax']) / 2 # Left camera
# y4_l1 = (res.xyxy[img_nr].iloc[1]['ymin'] + res.xyxy[img_nr].iloc[1]['ymax']) / 2 # Left camera

# x4_r1 = (res.xyxy[img_nr + 1].iloc[2]['xmin'] + res.xyxy[img_nr + 1].iloc[2]['xmax']) / 2 # Right camera
# y4_r1 = (res.xyxy[img_nr + 1].iloc[2]['ymin'] + res.xyxy[img_nr + 1].iloc[2]['ymax']) / 2 # Right camera

# # Triangulation
# Z4 = (baseline * f) / (x4_l1 - x4_r1) / px_size
# X4 = (x4_l1 - x4_r1) * px_size * Z4 / f
# Y4 = (y4_l1 - y4_r1) * px_size * Z4 / f

# Z_points.append(Z4)
# X_points.append(X4)
# Y_points.append(Y4)

# # Calculate the euclidan distance
# dist4 = np.sqrt(X4**2 + Y4**2 + Z4**2)
# print(dist4)

# img_nr = 8
# # Middle of the bounding boxes
# x5_l1 = (res.xyxy[img_nr].iloc[0]['xmin'] + res.xyxy[img_nr].iloc[0]['xmax']) / 2 # Left camera
# y5_l1 = (res.xyxy[img_nr].iloc[0]['ymin'] + res.xyxy[img_nr].iloc[0]['ymax']) / 2 # Left camera

# x5_r1 = (res.xyxy[img_nr + 1].iloc[1]['xmin'] + res.xyxy[img_nr + 1].iloc[1]['xmax']) / 2 # Right camera
# y5_r1 = (res.xyxy[img_nr + 1].iloc[1]['ymin'] + res.xyxy[img_nr + 1].iloc[1]['ymax']) / 2 # Right camera

# # Triangulation
# Z5 = (baseline * f) / (x5_l1 - x5_r1) / px_size
# X5 = (x5_l1 - x5_r1) * px_size * Z5 / f
# Y5 = (y5_l1 - y5_r1) * px_size * Z5 / f

# Z_points.append(Z5)
# X_points.append(X5)
# Y_points.append(Y5)

# # Calculate the euclidan distance
# dist5 = np.sqrt(X5**2 + Y5**2 + Z5**2)
# print(dist5)

# # Plotting
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(X_points, Y_points, Z_points)
# n = [1, 2, 3, 4, 5]

# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')

# plt.savefig(f'../../result_images/3D-trajectory-{start_img}.png')

# # Just to visualize
# obj_nr = 0
# xmax = res.xyxy[img_nr].iloc[obj_nr]['xmax']
# xmax = int(round(xmax))
# ymax = res.xyxy[img_nr].iloc[obj_nr]['ymax']
# ymax = int(round(ymax))
# xmin = res.xyxy[img_nr].iloc[obj_nr]['xmin']
# xmin = int(round(xmin))
# ymin = res.xyxy[img_nr].iloc[obj_nr]['ymin']
# ymin = int(round(ymin))

# color = (0, 255, 255) # BGR

# # Creates a bounding box with information generated from Yolov5 bounding boxes
# imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin, ymax), (xmax, ymin), color, 2)
# font = cv.FONT_HERSHEY_SIMPLEX
# # imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(fish_distance1, 3)} m', (xmin, ymin - 5), font, 0.8, color, 2)
# imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(swim_speed1, 3)} m/s', (xmin, ymin - 5), font, 0.8, color, 2)

# # # obj_nr = 3
# # # xmax = res.xyxy[img_nr].iloc[obj_nr]['xmax']
# # # xmax = int(round(xmax))
# # # ymax = res.xyxy[img_nr].iloc[obj_nr]['ymax']
# # # ymax = int(round(ymax))
# # # xmin = res.xyxy[img_nr].iloc[obj_nr]['xmin']
# # # xmin = int(round(xmin))
# # # ymin = res.xyxy[img_nr].iloc[obj_nr]['ymin']
# # # ymin = int(round(ymin))

# # # # Creates a bounding box with information generated from Yolov5 bounding boxes
# # # imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin, ymax), (xmax, ymin), color, 2)
# # # imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(fish_distance160_2, 3)} m', (xmin, ymin - 5), font, 0.8, color, 2)
# cv.imwrite('../../result_images/img_101_swim_speed.png', imgs[img_nr])
