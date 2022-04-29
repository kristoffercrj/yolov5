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

start_img = 1276
end_img = start_img + 9

folder_name = f'img{str(start_img)}'

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

# results.save() # Save predictions
print(res.xyxy)

X_points = []
Y_points = []
Z_points = []

X1, Y1, Z1 = [], [], []
X2, Y2, Z2 = [], [], []
X3, Y3, Z3 = [], [], []
X4, Y4, Z4 = [], [], []
X5, Y5, Z5 = [], [], []
X6, Y6, Z6 = [], [], []
X7, Y7, Z7 = [], [], []
X8, Y8, Z8 = [], [], []
X9, Y9, Z9 = [], [], []
X10, Y10, Z10 = [], [], []

img_nr = 0
# Round to integers
xmax1 = res.xyxy[img_nr].iloc[0]['xmax']
xmax1 = int(round(xmax1))
ymax1 = res.xyxy[img_nr].iloc[0]['ymax']
ymax1 = int(round(ymax1))
xmin1 = res.xyxy[img_nr].iloc[0]['xmin']
xmin1 = int(round(xmin1))
ymin1 = res.xyxy[img_nr].iloc[0]['ymin']
ymin1 = int(round(ymin1))

# Load data from superglue matching
npz_file = np.load(f'sintefmatching/{folder_name}/l1200-{start_img}_r1200-{start_img}_matches.npz') # Load .npz file from superglue matching 
ms_idx = np.where(npz_file['matches']>-1)[0] # Indices of matches for keypoints 0
ms = npz_file['matches']
kps0 = npz_file['keypoints0']
kps1 = npz_file['keypoints1']

print(f'Found {len(ms_idx)} matches!')

for i in range(0, len(ms_idx)):
    kp0_x = kps0[ms_idx[i]][0] # x-coordinate of keypoint 0
    kp0_y = kps0[ms_idx[i]][1] # y-coordinate of keypoint 0

    if ((kp0_y >= res.xyxy[img_nr].iloc[0]['ymin']) and \
        (kp0_y <= res.xyxy[img_nr].iloc[0]['ymax']) and \
        (kp0_x >= res.xyxy[img_nr].iloc[0]['xmin']) and \
        (kp0_x <= res.xyxy[img_nr].iloc[0]['xmax'])):

        x_disparity = (kps0[ms_idx[i]][0] - kps1[ms[ms_idx[i]]][0])
        y_disparity = (kps0[ms_idx[i]][1] - kps1[ms[ms_idx[i]]][1])

        Z_t = (baseline * f) / x_disparity / px_size
        X_t = (x_disparity * Z_t * px_size / f)
        Y_t = (y_disparity * Z_t * px_size / f)

        Z1 = np.append(Z1, Z_t)
        X1 = np.append(X1, X_t)
        Y1 = np.append(Y1, Y_t)

Z1 = sum(Z1) / len(Z1)
X1 = sum(X1) / len(X1)
Y1 = sum(Y1) / len(Y1)

Z_points.append(Z1)
X_points.append(X1)
Y_points.append(Y1)

# Calculate the euclidan distance
dist1 = np.sqrt(X1**2 + Y1**2 + Z1**2)
print(f'distance1: {dist1}')

# Creates a bounding box with information generated from Yolov5 bounding boxes
imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin1, ymax1), (xmax1, ymin1), (0, 0, 255), 2)
font = cv.FONT_HERSHEY_SIMPLEX
imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(dist1, 3)} m', (xmin1, ymin1 - 5), font, 0.8, (0, 0, 255), 2)
cv.imwrite(f'../../result_images/{folder_name}/estimated_{start_img}.png', imgs[img_nr])
start_img += 1

start_img += 1

img_nr = 2
# Round to integers
xmax2 = res.xyxy[img_nr].iloc[1]['xmax']
xmax2 = int(round(xmax2))
ymax2 = res.xyxy[img_nr].iloc[1]['ymax']
ymax2 = int(round(ymax2))
xmin2 = res.xyxy[img_nr].iloc[1]['xmin']
xmin2 = int(round(xmin2))
ymin2 = res.xyxy[img_nr].iloc[1]['ymin']
ymin2 = int(round(ymin2))

# Load data from superglue matching
npz_file = np.load(f'sintefmatching/{folder_name}/l1200-{start_img}_r1200-{start_img}_matches.npz') # Load .npz file from superglue matching 
ms_idx = np.where(npz_file['matches']>-1)[0] # Indices of matches for keypoints 0
ms = npz_file['matches']
kps0 = npz_file['keypoints0']
kps1 = npz_file['keypoints1']

print(f'Found {len(ms_idx)} matches!')

for i in range(0, len(ms_idx)):
    kp0_x = kps0[ms_idx[i]][0] # x-coordinate of keypoint 0
    kp0_y = kps0[ms_idx[i]][1] # y-coordinate of keypoint 0

    if ((kp0_y >= res.xyxy[img_nr].iloc[1]['ymin']) and \
        (kp0_y <= res.xyxy[img_nr].iloc[1]['ymax']) and \
        (kp0_x >= res.xyxy[img_nr].iloc[1]['xmin']) and \
        (kp0_x <= res.xyxy[img_nr].iloc[1]['xmax'])):

        x_disparity = (kps0[ms_idx[i]][0] - kps1[ms[ms_idx[i]]][0])
        y_disparity = (kps0[ms_idx[i]][1] - kps1[ms[ms_idx[i]]][1])

        Z_t = (baseline * f) / x_disparity / px_size
        X_t = (x_disparity * Z_t * px_size / f)
        Y_t = (y_disparity * Z_t * px_size / f)

        Z2 = np.append(Z2, Z_t)
        X2 = np.append(X2, X_t)
        Y2 = np.append(Y2, Y_t)

Z2 = sum(Z2) / len(Z2)
X2 = sum(X2) / len(X2)
Y2 = sum(Y2) / len(Y2)

# Calculate the euclidan distance
dist2 = np.sqrt(X2**2 + Y2**2 + Z2**2)
print(f'distance2: {dist2}')
speed1 = abs((dist2 - dist1)) * fps
print(speed1)

Z_points.append(Z2)
X_points.append(X2)
Y_points.append(Y2)

# Creates a bounding box with information generated from Yolov5 bounding boxes
imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin2, ymax2), (xmax2, ymin2), (0, 0, 255), 2)
imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(speed1, 3)} m/s', (xmin2, ymin2 - 5), font, 0.8, (0, 0, 255), 2)
cv.imwrite(f'../../result_images/{folder_name}/estimated_{start_img}.png', imgs[img_nr])
start_img += 1

img_nr = 4
# Round to integers
xmax3 = res.xyxy[img_nr].iloc[1]['xmax']
xmax3 = int(round(xmax3))
ymax3 = res.xyxy[img_nr].iloc[1]['ymax']
ymax3 = int(round(ymax3))
xmin3 = res.xyxy[img_nr].iloc[1]['xmin']
xmin3 = int(round(xmin3))
ymin3 = res.xyxy[img_nr].iloc[1]['ymin']
ymin3 = int(round(ymin3))

# Load data from superglue matching
npz_file = np.load(f'sintefmatching/{folder_name}/l1200-{start_img}_r1200-{start_img}_matches.npz') # Load .npz file from superglue matching 
ms_idx = np.where(npz_file['matches']>-1)[0] # Indices of matches for keypoints 0
ms = npz_file['matches']
kps0 = npz_file['keypoints0']
kps1 = npz_file['keypoints1']

print(f'Found {len(ms_idx)} matches!')

for i in range(0, len(ms_idx)):
    kp0_x = kps0[ms_idx[i]][0] # x-coordinate of keypoint 0
    kp0_y = kps0[ms_idx[i]][1] # y-coordinate of keypoint 0

    if ((kp0_y >= res.xyxy[img_nr].iloc[1]['ymin']) and \
        (kp0_y <= res.xyxy[img_nr].iloc[1]['ymax']) and \
        (kp0_x >= res.xyxy[img_nr].iloc[1]['xmin']) and \
        (kp0_x <= res.xyxy[img_nr].iloc[1]['xmax'])):

        x_disparity = (kps0[ms_idx[i]][0] - kps1[ms[ms_idx[i]]][0])
        y_disparity = (kps0[ms_idx[i]][1] - kps1[ms[ms_idx[i]]][1])

        Z_t = (baseline * f) / x_disparity / px_size
        X_t = (x_disparity * Z_t * px_size / f)
        Y_t = (y_disparity * Z_t * px_size / f)

        Z3 = np.append(Z3, Z_t)
        X3 = np.append(X3, X_t)
        Y3 = np.append(Y3, Y_t)

Z3 = sum(Z3) / len(Z3)
X3 = sum(X3) / len(X3)
Y3 = sum(Y3) / len(Y3)

Z_points.append(Z3)
X_points.append(X3)
Y_points.append(Y3)

# Calculate the euclidan distance
dist3 = np.sqrt(X3**2 + Y3**2 + Z3**2)
print(f'distance3: {dist3}')
speed2 = abs((dist3 - dist2)) * fps
print(speed2)

# Creates a bounding box with information generated from Yolov5 bounding boxes
imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin3, ymax3), (xmax3, ymin3), (0, 0, 255), 2)
imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(speed2, 3)} m/s', (xmin3, ymin3 - 5), font, 0.8, (0, 0, 255), 2)
cv.imwrite(f'../../result_images/{folder_name}/estimated_{start_img}.png', imgs[img_nr])
start_img += 1

img_nr = 6
# Round to integers
xmax4 = res.xyxy[img_nr].iloc[1]['xmax']
xmax4 = int(round(xmax4))
ymax4 = res.xyxy[img_nr].iloc[1]['ymax']
ymax4 = int(round(ymax4))
xmin4 = res.xyxy[img_nr].iloc[1]['xmin']
xmin4 = int(round(xmin4))
ymin4 = res.xyxy[img_nr].iloc[1]['ymin']
ymin4 = int(round(ymin4))

# Load data from superglue matching
npz_file = np.load(f'sintefmatching/{folder_name}/l1200-{start_img}_r1200-{start_img}_matches.npz') # Load .npz file from superglue matching 
ms_idx = np.where(npz_file['matches']>-1)[0] # Indices of matches for keypoints 0
ms = npz_file['matches']
kps0 = npz_file['keypoints0']
kps1 = npz_file['keypoints1']

print(f'Found {len(ms_idx)} matches!')

for i in range(0, len(ms_idx)):
    kp0_x = kps0[ms_idx[i]][0] # x-coordinate of keypoint 0
    kp0_y = kps0[ms_idx[i]][1] # y-coordinate of keypoint 0

    state = ((kp0_y >= res.xyxy[img_nr].iloc[1]['ymin']) and \
        (kp0_y <= res.xyxy[img_nr].iloc[1]['ymax']) and \
        (kp0_x >= res.xyxy[img_nr].iloc[1]['xmin']) and \
        (kp0_x <= res.xyxy[img_nr].iloc[1]['xmax']))
    print(state)

    if ((kp0_y >= res.xyxy[img_nr].iloc[1]['ymin']) and \
        (kp0_y <= res.xyxy[img_nr].iloc[1]['ymax']) and \
        (kp0_x >= res.xyxy[img_nr].iloc[1]['xmin']) and \
        (kp0_x <= res.xyxy[img_nr].iloc[1]['xmax'])):
        
        x_disparity = (kps0[ms_idx[i]][0] - kps1[ms[ms_idx[i]]][0])
        y_disparity = (kps0[ms_idx[i]][1] - kps1[ms[ms_idx[i]]][1])

        Z_t = (baseline * f) / x_disparity / px_size
        X_t = (x_disparity * Z_t * px_size / f)
        Y_t = (y_disparity * Z_t * px_size / f)

        Z4 = np.append(Z4, Z_t)
        X4 = np.append(X4, X_t)
        Y4 = np.append(Y4, Y_t)

Z4 = sum(Z4) / len(Z4)
X4 = sum(X4) / len(X4)
Y4 = sum(Y4) / len(Y4)

Z_points.append(Z4)
X_points.append(X4)
Y_points.append(Y4)

# Calculate the euclidan distance
dist4 = np.sqrt(X4**2 + Y4**2 + Z4**2)
print(f'distance4: {dist4}')
speed3 = abs((dist4 - dist3)) * fps
print(speed3)

# Creates a bounding box with information generated from Yolov5 bounding boxes
imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin4, ymax4), (xmax4, ymin4), (0, 0, 255), 2)
imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(speed3, 3)} m/s', (xmin4, ymin4 - 5), font, 0.8, (0, 0, 255), 2)
cv.imwrite(f'../../result_images/{folder_name}/estimated_{start_img}.png', imgs[img_nr])
start_img += 1

img_nr = 8
# Round to integers
xmax5 = res.xyxy[img_nr].iloc[1]['xmax']
xmax5 = int(round(xmax5))
ymax5 = res.xyxy[img_nr].iloc[1]['ymax']
ymax5 = int(round(ymax5))
xmin5 = res.xyxy[img_nr].iloc[1]['xmin']
xmin5 = int(round(xmin5))
ymin5 = res.xyxy[img_nr].iloc[1]['ymin']
ymin5 = int(round(ymin5))

# Load data from superglue matching
npz_file = np.load(f'sintefmatching/{folder_name}/l1200-{start_img}_r1200-{start_img}_matches.npz') # Load .npz file from superglue matching 
ms_idx = np.where(npz_file['matches']>-1)[0] # Indices of matches for keypoints 0
ms = npz_file['matches']
kps0 = npz_file['keypoints0']
kps1 = npz_file['keypoints1']

print(f'Found {len(ms_idx)} matches!')

for i in range(0, len(ms_idx)):
    kp0_x = kps0[ms_idx[i]][0] # x-coordinate of keypoint 0
    kp0_y = kps0[ms_idx[i]][1] # y-coordinate of keypoint 0

    if ((kp0_y >= res.xyxy[img_nr].iloc[1]['ymin']) and \
        (kp0_y <= res.xyxy[img_nr].iloc[1]['ymax']) and \
        (kp0_x >= res.xyxy[img_nr].iloc[1]['xmin']) and \
        (kp0_x <= res.xyxy[img_nr].iloc[1]['xmax'])):

        x_disparity = (kps0[ms_idx[i]][0] - kps1[ms[ms_idx[i]]][0])
        y_disparity = (kps0[ms_idx[i]][1] - kps1[ms[ms_idx[i]]][1])

        Z_t = (baseline * f) / x_disparity / px_size
        X_t = (x_disparity * Z_t * px_size / f)
        Y_t = (y_disparity * Z_t * px_size / f)

        Z5 = np.append(Z5, Z_t)
        X5 = np.append(X5, X_t)
        Y5 = np.append(Y5, Y_t)

Z5 = sum(Z5) / len(Z5)
X5 = sum(X5) / len(X5)
Y5 = sum(Y5) / len(Y5)

Z_points.append(Z5)
X_points.append(X5)
Y_points.append(Y5)

# Calculate the euclidan distance
dist5 = np.sqrt(X5**2 + Y5**2 + Z5**2)
print(f'distance5: {dist5}')
speed4 = abs((dist5 - dist4)) * fps
print(speed4)

# Creates a bounding box with information generated from Yolov5 bounding boxes
imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin5, ymax5), (xmax5, ymin5), (0, 0, 255), 2)
imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(speed4, 3)} m/s', (xmin5, ymin5 - 5), font, 0.8, (0, 0, 255), 2)
cv.imwrite(f'../../result_images/{folder_name}/estimated_{start_img}.png', imgs[img_nr])
start_img += 1

img_nr = 10
# Round to integers
xmax6 = res.xyxy[img_nr].iloc[0]['xmax']
xmax6 = int(round(xmax6))
ymax6 = res.xyxy[img_nr].iloc[0]['ymax']
ymax6 = int(round(ymax6))
xmin6 = res.xyxy[img_nr].iloc[0]['xmin']
xmin6 = int(round(xmin6))
ymin6 = res.xyxy[img_nr].iloc[0]['ymin']
ymin6 = int(round(ymin6))

# Load data from superglue matching
npz_file = np.load(f'sintefmatching/{folder_name}/l1200-{start_img}_r1200-{start_img}_matches.npz') # Load .npz file from superglue matching 
ms_idx = np.where(npz_file['matches']>-1)[0] # Indices of matches for keypoints 0
ms = npz_file['matches']
kps0 = npz_file['keypoints0']
kps1 = npz_file['keypoints1']

print(f'Found {len(ms_idx)} matches!')

for i in range(0, len(ms_idx)):
    kp0_x = kps0[ms_idx[i]][0] # x-coordinate of keypoint 0
    kp0_y = kps0[ms_idx[i]][1] # y-coordinate of keypoint 0

    if ((kp0_y >= res.xyxy[img_nr].iloc[0]['ymin']) and \
        (kp0_y <= res.xyxy[img_nr].iloc[0]['ymax']) and \
        (kp0_x >= res.xyxy[img_nr].iloc[0]['xmin']) and \
        (kp0_x <= res.xyxy[img_nr].iloc[0]['xmax'])):

        x_disparity = (kps0[ms_idx[i]][0] - kps1[ms[ms_idx[i]]][0])
        y_disparity = (kps0[ms_idx[i]][1] - kps1[ms[ms_idx[i]]][1])

        Z_t = (baseline * f) / x_disparity / px_size
        X_t = (x_disparity * Z_t * px_size / f)
        Y_t = (y_disparity * Z_t * px_size / f)

        Z6 = np.append(Z6, Z_t)
        X6 = np.append(X6, X_t)
        Y6 = np.append(Y6, Y_t)

Z6 = sum(Z6) / len(Z6)
X6 = sum(X6) / len(X6)
Y6 = sum(Y6) / len(Y6)

Z_points.append(Z6)
X_points.append(X6)
Y_points.append(Y6)

# Calculate the euclidan distance
dist6 = np.sqrt(X6**2 + Y6**2 + Z6**2)
print(f'distance6: {dist6}')
speed5 = abs((dist6 - dist5)) * fps
print(speed5)

# Creates a bounding box with information generated from Yolov5 bounding boxes
imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin6, ymax6), (xmax6, ymin6), (0, 0, 255), 2)
imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(speed5, 3)} m/s', (xmin6, ymin6 - 5), font, 0.8, (0, 0, 255), 2)
cv.imwrite(f'../../result_images/{folder_name}/estimated_{start_img}.png', imgs[img_nr])
start_img += 1

img_nr = 12
# Round to integers
xmax7 = res.xyxy[img_nr].iloc[0]['xmax']
xmax7 = int(round(xmax7))
ymax7 = res.xyxy[img_nr].iloc[0]['ymax']
ymax7 = int(round(ymax7))
xmin7 = res.xyxy[img_nr].iloc[0]['xmin']
xmin7 = int(round(xmin7))
ymin7 = res.xyxy[img_nr].iloc[0]['ymin']
ymin7 = int(round(ymin7))

# Load data from superglue matching
npz_file = np.load(f'sintefmatching/{folder_name}/l1200-{start_img}_r1200-{start_img}_matches.npz') # Load .npz file from superglue matching 
ms_idx = np.where(npz_file['matches']>-1)[0] # Indices of matches for keypoints 0
ms = npz_file['matches']
kps0 = npz_file['keypoints0']
kps1 = npz_file['keypoints1']

print(f'Found {len(ms_idx)} matches!')

for i in range(0, len(ms_idx)):
    kp0_x = kps0[ms_idx[i]][0] # x-coordinate of keypoint 0
    kp0_y = kps0[ms_idx[i]][1] # y-coordinate of keypoint 0

    if ((kp0_y >= res.xyxy[img_nr].iloc[0]['ymin']) and \
        (kp0_y <= res.xyxy[img_nr].iloc[0]['ymax']) and \
        (kp0_x >= res.xyxy[img_nr].iloc[0]['xmin']) and \
        (kp0_x <= res.xyxy[img_nr].iloc[0]['xmax'])):

        x_disparity = (kps0[ms_idx[i]][0] - kps1[ms[ms_idx[i]]][0])
        y_disparity = (kps0[ms_idx[i]][1] - kps1[ms[ms_idx[i]]][1])

        Z_t = (baseline * f) / x_disparity / px_size
        X_t = (x_disparity * Z_t * px_size / f)
        Y_t = (y_disparity * Z_t * px_size / f)

        Z7 = np.append(Z7, Z_t)
        X7 = np.append(X7, X_t)
        Y7 = np.append(Y7, Y_t)

Z7 = sum(Z7) / len(Z7)
X7 = sum(X7) / len(X7)
Y7 = sum(Y7) / len(Y7)

Z_points.append(Z7)
X_points.append(X7)
Y_points.append(Y7)

# Calculate the euclidan distance
dist7 = np.sqrt(X7**2 + Y7**2 + Z7**2)
print(f'distance7: {dist7}')
speed6 = abs((dist7 - dist6)) * fps
print(speed6)

# Creates a bounding box with information generated from Yolov5 bounding boxes
imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin7, ymax7), (xmax7, ymin7), (0, 0, 255), 2)
imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(speed6, 3)} m/s', (xmin7, ymin7 - 5), font, 0.8, (0, 0, 255), 2)
cv.imwrite(f'../../result_images/{folder_name}/estimated_{start_img}.png', imgs[img_nr])
start_img += 1

img_nr = 14
# Round to integers
xmax8 = res.xyxy[img_nr].iloc[0]['xmax']
xmax8 = int(round(xmax8))
ymax8 = res.xyxy[img_nr].iloc[0]['ymax']
ymax8 = int(round(ymax8))
xmin8 = res.xyxy[img_nr].iloc[0]['xmin']
xmin8 = int(round(xmin8))
ymin8 = res.xyxy[img_nr].iloc[0]['ymin']
ymin8 = int(round(ymin8))

# Load data from superglue matching
npz_file = np.load(f'sintefmatching/{folder_name}/l1200-{start_img}_r1200-{start_img}_matches.npz') # Load .npz file from superglue matching 
ms_idx = np.where(npz_file['matches']>-1)[0] # Indices of matches for keypoints 0
ms = npz_file['matches']
kps0 = npz_file['keypoints0']
kps1 = npz_file['keypoints1']

print(f'Found {len(ms_idx)} matches!')

for i in range(0, len(ms_idx)):
    kp0_x = kps0[ms_idx[i]][0] # x-coordinate of keypoint 0
    kp0_y = kps0[ms_idx[i]][1] # y-coordinate of keypoint 0

    if ((kp0_y >= res.xyxy[img_nr].iloc[0]['ymin']) and \
        (kp0_y <= res.xyxy[img_nr].iloc[0]['ymax']) and \
        (kp0_x >= res.xyxy[img_nr].iloc[0]['xmin']) and \
        (kp0_x <= res.xyxy[img_nr].iloc[0]['xmax'])):

        x_disparity = (kps0[ms_idx[i]][0] - kps1[ms[ms_idx[i]]][0])
        y_disparity = (kps0[ms_idx[i]][1] - kps1[ms[ms_idx[i]]][1])

        Z_t = (baseline * f) / x_disparity / px_size
        X_t = (x_disparity * Z_t * px_size / f)
        Y_t = (y_disparity * Z_t * px_size / f)

        Z8 = np.append(Z8, Z_t)
        X8 = np.append(X8, X_t)
        Y8 = np.append(Y8, Y_t)

Z8 = sum(Z8) / len(Z8)
X8 = sum(X8) / len(X8)
Y8 = sum(Y8) / len(Y8)

Z_points.append(Z8)
X_points.append(X8)
Y_points.append(Y8)

# Calculate the euclidan distance
dist8 = np.sqrt(X8**2 + Y8**2 + Z8**2)
print(f'distance8: {dist8}')
speed7 = abs((dist8 - dist7)) * fps
print(speed7)

# Creates a bounding box with information generated from Yolov5 bounding boxes
imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin8, ymax8), (xmax8, ymin8), (0, 0, 255), 2)
imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(speed7, 3)} m/s', (xmin8, ymin8 - 5), font, 0.8, (0, 0, 255), 2)
cv.imwrite(f'../../result_images/{folder_name}/estimated_{start_img}.png', imgs[img_nr])
start_img += 1

img_nr = 16
# Round to integers
xmax9 = res.xyxy[img_nr].iloc[1]['xmax']
xmax9 = int(round(xmax9))
ymax9 = res.xyxy[img_nr].iloc[1]['ymax']
ymax9 = int(round(ymax9))
xmin9 = res.xyxy[img_nr].iloc[1]['xmin']
xmin9 = int(round(xmin9))
ymin9 = res.xyxy[img_nr].iloc[1]['ymin']
ymin9 = int(round(ymin9))

# Load data from superglue matching
npz_file = np.load(f'sintefmatching/{folder_name}/l1200-{start_img}_r1200-{start_img}_matches.npz') # Load .npz file from superglue matching 
ms_idx = np.where(npz_file['matches']>-1)[0] # Indices of matches for keypoints 0
ms = npz_file['matches']
kps0 = npz_file['keypoints0']
kps1 = npz_file['keypoints1']

print(f'Found {len(ms_idx)} matches!')

for i in range(0, len(ms_idx)):
    kp0_x = kps0[ms_idx[i]][0] # x-coordinate of keypoint 0
    kp0_y = kps0[ms_idx[i]][1] # y-coordinate of keypoint 0

    if ((kp0_y >= res.xyxy[img_nr].iloc[1]['ymin']) and \
        (kp0_y <= res.xyxy[img_nr].iloc[1]['ymax']) and \
        (kp0_x >= res.xyxy[img_nr].iloc[1]['xmin']) and \
        (kp0_x <= res.xyxy[img_nr].iloc[1]['xmax'])):

        x_disparity = (kps0[ms_idx[i]][0] - kps1[ms[ms_idx[i]]][0])
        y_disparity = (kps0[ms_idx[i]][1] - kps1[ms[ms_idx[i]]][1])

        Z_t = (baseline * f) / x_disparity / px_size
        X_t = (x_disparity * Z_t * px_size / f)
        Y_t = (y_disparity * Z_t * px_size / f)

        Z9 = np.append(Z9, Z_t)
        X9 = np.append(X9, X_t)
        Y9 = np.append(Y9, Y_t)

Z9 = sum(Z9) / len(Z9)
X9 = sum(X9) / len(X9)
Y9 = sum(Y9) / len(Y9)

Z_points.append(Z9)
X_points.append(X9)
Y_points.append(Y9)

# Calculate the euclidan distance
dist9 = np.sqrt(X9**2 + Y9**2 + Z9**2)
print(f'distance9: {dist9}')
speed8 = abs((dist9 - dist8)) * fps
print(speed8)

# Creates a bounding box with information generated from Yolov5 bounding boxes
imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin9, ymax9), (xmax9, ymin9), (0, 0, 255), 2)
imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(speed8, 3)} m/s', (xmin9, ymin9 - 5), font, 0.8, (0, 0, 255), 2)
cv.imwrite(f'../../result_images/{folder_name}/estimated_{start_img}.png', imgs[img_nr])
start_img += 1

img_nr = 18
# Round to integers
xmax10 = res.xyxy[img_nr].iloc[0]['xmax']
xmax10 = int(round(xmax10))
ymax10 = res.xyxy[img_nr].iloc[0]['ymax']
ymax10 = int(round(ymax10))
xmin10 = res.xyxy[img_nr].iloc[0]['xmin']
xmin10 = int(round(xmin10))
ymin10 = res.xyxy[img_nr].iloc[0]['ymin']
ymin10 = int(round(ymin10))

# Load data from superglue matching
npz_file = np.load(f'sintefmatching/{folder_name}/l1200-{start_img}_r1200-{start_img}_matches.npz') # Load .npz file from superglue matching 
ms_idx = np.where(npz_file['matches']>-1)[0] # Indices of matches for keypoints 0
ms = npz_file['matches']
kps0 = npz_file['keypoints0']
kps1 = npz_file['keypoints1']

print(f'Found {len(ms_idx)} matches!')

for i in range(0, len(ms_idx)):
    kp0_x = kps0[ms_idx[i]][0] # x-coordinate of keypoint 0
    kp0_y = kps0[ms_idx[i]][1] # y-coordinate of keypoint 0

    if ((kp0_y >= res.xyxy[img_nr].iloc[0]['ymin']) and \
        (kp0_y <= res.xyxy[img_nr].iloc[0]['ymax']) and \
        (kp0_x >= res.xyxy[img_nr].iloc[0]['xmin']) and \
        (kp0_x <= res.xyxy[img_nr].iloc[0]['xmax'])):

        x_disparity = (kps0[ms_idx[i]][0] - kps1[ms[ms_idx[i]]][0])
        y_disparity = (kps0[ms_idx[i]][1] - kps1[ms[ms_idx[i]]][1])

        Z_t = (baseline * f) / x_disparity / px_size
        X_t = (x_disparity * Z_t * px_size / f)
        Y_t = (y_disparity * Z_t * px_size / f)

        Z10 = np.append(Z10, Z_t)
        X10 = np.append(X10, X_t)
        Y10 = np.append(Y10, Y_t)

Z10 = sum(Z10) / len(Z10)
X10 = sum(X10) / len(X10)
Y10 = sum(Y10) / len(Y10)

Z_points.append(Z10)
X_points.append(X10)
Y_points.append(Y10)

# Calculate the euclidan distance
dist10 = np.sqrt(X10**2 + Y10**2 + Z10**2)
print(f'distance10: {dist10}')
speed9 = abs((dist10 - dist9)) * fps
print(speed9)

# Creates a bounding box with information generated from Yolov5 bounding boxes
imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin10, ymax10), (xmax10, ymin10), (0, 0, 255), 2)
imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(speed9, 3)} m/s', (xmin10, ymin10 - 5), font, 0.8, (0, 0, 255), 2)
cv.imwrite(f'../../result_images/{folder_name}/estimated_{start_img}.png', imgs[img_nr])
start_img += 1

# Plotting
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_points, Y_points, Z_points)
n = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for i in range(len(n)):
    ax.text(X_points[i], Y_points[i], Z_points[i], f'{str(i)}')

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

plt.savefig(f'../../result_images/{folder_name}/3D-trajectory-{start_img}.png')

