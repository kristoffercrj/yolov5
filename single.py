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

start_img = 1790
end_img = 1790

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

# results.save() # Show predictions
print(res.xyxy)

X_points = []
Y_points = []
Z_points = []

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

# Middle of the bounding boxes
x1_l1 = (res.xyxy[img_nr].iloc[0]['xmin'] + res.xyxy[img_nr].iloc[0]['xmax']) / 2 # Left camera
y1_l1 = (res.xyxy[img_nr].iloc[0]['ymin'] + res.xyxy[img_nr].iloc[0]['ymax']) / 2 # Left camera

x1_r1 = (res.xyxy[img_nr + 1].iloc[0]['xmin'] + res.xyxy[img_nr + 1].iloc[0]['xmax']) / 2 # Right camera
y1_r1 = (res.xyxy[img_nr + 1].iloc[0]['ymin'] + res.xyxy[img_nr + 1].iloc[0]['ymax']) / 2 # Right camera

# Triangulation
Z1 = (baseline * f) / (x1_l1 - x1_r1) / px_size
X1 = (x1_l1 - x1_r1) * px_size * Z1 / f
Y1 = (y1_l1 - y1_r1) * px_size * Z1 / f

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
cv.imwrite(f'../../result_images/{folder_name}/estimated_{str(img_nr)}.png', imgs[img_nr])

img_nr = 2
# Round to integers
xmax2 = res.xyxy[img_nr].iloc[0]['xmax']
xmax2 = int(round(xmax2))
ymax2 = res.xyxy[img_nr].iloc[0]['ymax']
ymax2 = int(round(ymax2))
xmin2 = res.xyxy[img_nr].iloc[0]['xmin']
xmin2 = int(round(xmin2))
ymin2 = res.xyxy[img_nr].iloc[0]['ymin']
ymin2 = int(round(ymin2))

# Middle of the bounding boxes
x2_l1 = (res.xyxy[img_nr].iloc[0]['xmin'] + res.xyxy[img_nr].iloc[0]['xmax']) / 2 # Left camera
y2_l1 = (res.xyxy[img_nr].iloc[0]['ymin'] + res.xyxy[img_nr].iloc[0]['ymax']) / 2 # Left camera

x2_r1 = (res.xyxy[img_nr + 1].iloc[0]['xmin'] + res.xyxy[img_nr + 1].iloc[0]['xmax']) / 2 # Right camera
y2_r1 = (res.xyxy[img_nr + 1].iloc[0]['ymin'] + res.xyxy[img_nr + 1].iloc[0]['ymax']) / 2 # Right camera

# Triangulation
Z2 = (baseline * f) / (x2_l1 - x2_r1) / px_size
X2 = (x2_l1 - x2_r1) * px_size * Z2 / f
Y2 = (y2_l1 - y2_r1) * px_size * Z2 / f

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
cv.imwrite(f'../../result_images/{folder_name}/estimated_{str(img_nr)}.png', imgs[img_nr])

img_nr = 4
# Round to integers
xmax3 = res.xyxy[img_nr].iloc[0]['xmax']
xmax3 = int(round(xmax3))
ymax3 = res.xyxy[img_nr].iloc[0]['ymax']
ymax3 = int(round(ymax3))
xmin3 = res.xyxy[img_nr].iloc[0]['xmin']
xmin3 = int(round(xmin3))
ymin3 = res.xyxy[img_nr].iloc[0]['ymin']
ymin3 = int(round(ymin3))

# Middle of the bounding boxes
x3_l1 = (res.xyxy[img_nr].iloc[0]['xmin'] + res.xyxy[img_nr].iloc[0]['xmax']) / 2 # Left camera
y3_l1 = (res.xyxy[img_nr].iloc[0]['ymin'] + res.xyxy[img_nr].iloc[0]['ymax']) / 2 # Left camera

x3_r1 = (res.xyxy[img_nr + 1].iloc[0]['xmin'] + res.xyxy[img_nr + 1].iloc[0]['xmax']) / 2 # Right camera
y3_r1 = (res.xyxy[img_nr + 1].iloc[0]['ymin'] + res.xyxy[img_nr + 1].iloc[0]['ymax']) / 2 # Right camera

# Triangulation
Z3 = (baseline * f) / (x3_l1 - x3_r1) / px_size
X3 = (x3_l1 - x3_r1) * px_size * Z3 / f
Y3 = (y3_l1 - y3_r1) * px_size * Z3 / f

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
cv.imwrite(f'../../result_images/{folder_name}/estimated_{str(img_nr)}.png', imgs[img_nr])

img_nr = 6
# Round to integers
xmax4 = res.xyxy[img_nr].iloc[0]['xmax']
xmax4 = int(round(xmax4))
ymax4 = res.xyxy[img_nr].iloc[0]['ymax']
ymax4 = int(round(ymax4))
xmin4 = res.xyxy[img_nr].iloc[0]['xmin']
xmin4 = int(round(xmin4))
ymin4 = res.xyxy[img_nr].iloc[0]['ymin']
ymin4 = int(round(ymin4))

# Middle of the bounding boxes
x4_l1 = (res.xyxy[img_nr].iloc[0]['xmin'] + res.xyxy[img_nr].iloc[0]['xmax']) / 2 # Left camera
y4_l1 = (res.xyxy[img_nr].iloc[0]['ymin'] + res.xyxy[img_nr].iloc[0]['ymax']) / 2 # Left camera

x4_r1 = (res.xyxy[img_nr + 1].iloc[0]['xmin'] + res.xyxy[img_nr + 1].iloc[0]['xmax']) / 2 # Right camera
y4_r1 = (res.xyxy[img_nr + 1].iloc[0]['ymin'] + res.xyxy[img_nr + 1].iloc[0]['ymax']) / 2 # Right camera

# Triangulation
Z4 = (baseline * f) / (x4_l1 - x4_r1) / px_size
X4 = (x4_l1 - x4_r1) * px_size * Z4 / f
Y4 = (y4_l1 - y4_r1) * px_size * Z4 / f

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
cv.imwrite(f'../../result_images/{folder_name}/estimated_{str(img_nr)}.png', imgs[img_nr])

img_nr = 8
# Round to integers
xmax5 = res.xyxy[img_nr].iloc[0]['xmax']
xmax5 = int(round(xmax5))
ymax5 = res.xyxy[img_nr].iloc[0]['ymax']
ymax5 = int(round(ymax5))
xmin5 = res.xyxy[img_nr].iloc[0]['xmin']
xmin5 = int(round(xmin5))
ymin5 = res.xyxy[img_nr].iloc[0]['ymin']
ymin5 = int(round(ymin5))

# Middle of the bounding boxes
x5_l1 = (res.xyxy[img_nr].iloc[0]['xmin'] + res.xyxy[img_nr].iloc[0]['xmax']) / 2 # Left camera
y5_l1 = (res.xyxy[img_nr].iloc[0]['ymin'] + res.xyxy[img_nr].iloc[0]['ymax']) / 2 # Left camera

x5_r1 = (res.xyxy[img_nr + 1].iloc[0]['xmin'] + res.xyxy[img_nr + 1].iloc[0]['xmax']) / 2 # Right camera
y5_r1 = (res.xyxy[img_nr + 1].iloc[0]['ymin'] + res.xyxy[img_nr + 1].iloc[0]['ymax']) / 2 # Right camera

# Triangulation
Z5 = (baseline * f) / (x5_l1 - x5_r1) / px_size
X5 = (x5_l1 - x5_r1) * px_size * Z5 / f
Y5 = (y5_l1 - y5_r1) * px_size * Z5 / f

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
cv.imwrite(f'../../result_images/{folder_name}/estimated_{str(img_nr)}.png', imgs[img_nr])

# Plotting
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_points, Y_points, Z_points)
n = [1, 2, 3, 4, 5]
for i in range(len(n)):
    ax.text(X_points[i], Y_points[i], Z_points[i], f'{str(i)}')

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

plt.savefig(f'../../result_images/{folder_name}/3D-trajectory-{start_img}.png')