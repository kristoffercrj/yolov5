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

start_img = 2
end_img = start_img + 10

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

# Initialize arrays for 3D points
X_points1 = []
Y_points1 = []
Z_points1 = []

X_points2 = []
Y_points2 = []
Z_points2 = []

### IMG 1 
img_nr = 0
# Round to integers
xmax11 = res.xyxy[img_nr].iloc[0]['xmax']
xmax11 = int(round(xmax11))
ymax11 = res.xyxy[img_nr].iloc[0]['ymax']
ymax11 = int(round(ymax11))
xmin11 = res.xyxy[img_nr].iloc[0]['xmin']
xmin11 = int(round(xmin11))
ymin11 = res.xyxy[img_nr].iloc[0]['ymin']
ymin11 = int(round(ymin11))

# Round to integers
xmax12 = res.xyxy[img_nr].iloc[1]['xmax']
xmax12 = int(round(xmax12))
ymax12 = res.xyxy[img_nr].iloc[1]['ymax']
ymax12 = int(round(ymax12))
xmin12 = res.xyxy[img_nr].iloc[1]['xmin']
xmin12 = int(round(xmin12))
ymin12 = res.xyxy[img_nr].iloc[1]['ymin']
ymin12 = int(round(ymin12))

# Middle of the bounding boxes
# Fish 1
x1_l1 = (res.xyxy[img_nr].iloc[0]['xmin'] + res.xyxy[img_nr].iloc[0]['xmax']) / 2 # Left camera
y1_l1 = (res.xyxy[img_nr].iloc[0]['ymin'] + res.xyxy[img_nr].iloc[0]['ymax']) / 2 # Left camera

x1_r1 = (res.xyxy[img_nr + 1].iloc[0]['xmin'] + res.xyxy[img_nr + 1].iloc[0]['xmax']) / 2 # Right camera
y1_r1 = (res.xyxy[img_nr + 1].iloc[0]['ymin'] + res.xyxy[img_nr + 1].iloc[0]['ymax']) / 2 # Right camera

# Fish 2
x1_l2 = (res.xyxy[img_nr].iloc[1]['xmin'] + res.xyxy[img_nr].iloc[1]['xmax']) / 2 # Left camera
y1_l2 = (res.xyxy[img_nr].iloc[1]['ymin'] + res.xyxy[img_nr].iloc[1]['ymax']) / 2 # Left camera

x1_r2 = (res.xyxy[img_nr + 1].iloc[2]['xmin'] + res.xyxy[img_nr + 1].iloc[2]['xmax']) / 2 # Right camera
y1_r2 = (res.xyxy[img_nr + 1].iloc[2]['ymin'] + res.xyxy[img_nr + 1].iloc[2]['ymax']) / 2 # Right camera

# Triangulation
Z11 = (baseline * f) / (x1_l1 - x1_r1) / px_size
X11 = (x1_l1 - x1_r1) * px_size * Z11 / f
Y11 = (y1_l1 - y1_r1) * px_size * Z11 / f

Z12 = (baseline * f) / (x1_l2 - x1_r2) / px_size
X12 = (x1_l2 - x1_r2) * px_size * Z12 / f
Y12 = (y1_l2 - y1_r2) * px_size * Z12 / f

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

# Creates a bounding box with information generated from Yolov5 bounding boxes
imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin11, ymax11), (xmax11, ymin11), (0, 0, 255), 2)
font = cv.FONT_HERSHEY_SIMPLEX
imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(dist11, 3)} m', (xmin11, ymin11 - 5), font, 0.8, (0, 0, 255), 2)
imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin12, ymax12), (xmax12, ymin12), (0, 0, 255), 2)
imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(dist12, 3)} m', (xmin12, ymin12 - 5), font, 0.8, (0, 0, 255), 2)
cv.imwrite(f'../../result_images/{folder_name}/estimated_{str(img_nr)}.png', imgs[img_nr])

### IMG 2
img_nr = 2
# Round to integers
xmax21 = res.xyxy[img_nr].iloc[1]['xmax']
xmax21 = int(round(xmax21))
ymax21 = res.xyxy[img_nr].iloc[1]['ymax']
ymax21 = int(round(ymax21))
xmin21 = res.xyxy[img_nr].iloc[1]['xmin']
xmin21 = int(round(xmin21))
ymin21 = res.xyxy[img_nr].iloc[1]['ymin']
ymin21 = int(round(ymin21))

# Round to integers
xmax22 = res.xyxy[img_nr].iloc[0]['xmax']
xmax22 = int(round(xmax22))
ymax22 = res.xyxy[img_nr].iloc[0]['ymax']
ymax22 = int(round(ymax22))
xmin22 = res.xyxy[img_nr].iloc[0]['xmin']
xmin22 = int(round(xmin22))
ymin22 = res.xyxy[img_nr].iloc[0]['ymin']
ymin22 = int(round(ymin22))

# Middle of the bounding boxes
# Fish 1
x2_l1 = (res.xyxy[img_nr].iloc[1]['xmin'] + res.xyxy[img_nr].iloc[1]['xmax']) / 2 # Left camera
y2_l1 = (res.xyxy[img_nr].iloc[1]['ymin'] + res.xyxy[img_nr].iloc[1]['ymax']) / 2 # Left camera

x2_r1 = (res.xyxy[img_nr + 1].iloc[0]['xmin'] + res.xyxy[img_nr + 1].iloc[0]['xmax']) / 2 # Right camera
y2_r1 = (res.xyxy[img_nr + 1].iloc[0]['ymin'] + res.xyxy[img_nr + 1].iloc[0]['ymax']) / 2 # Right camera

# Fish 2
x2_l2 = (res.xyxy[img_nr].iloc[0]['xmin'] + res.xyxy[img_nr].iloc[0]['xmax']) / 2 # Left camera
y2_l2 = (res.xyxy[img_nr].iloc[0]['ymin'] + res.xyxy[img_nr].iloc[0]['ymax']) / 2 # Left camera

x2_r2 = (res.xyxy[img_nr + 1].iloc[1]['xmin'] + res.xyxy[img_nr + 1].iloc[1]['xmax']) / 2 # Right camera
y2_r2 = (res.xyxy[img_nr + 1].iloc[1]['ymin'] + res.xyxy[img_nr + 1].iloc[1]['ymax']) / 2 # Right camera

# Triangulation
Z21 = (baseline * f) / (x2_l1 - x2_r1) / px_size
X21 = (x2_l1 - x2_r1) * px_size * Z21 / f
Y21 = (y2_l1 - y2_r1) * px_size * Z21 / f

Z22 = (baseline * f) / (x2_l2 - x2_r2) / px_size
X22 = (x2_l2 - x2_r2) * px_size * Z22 / f
Y22 = (y2_l2 - y2_r2) * px_size * Z22 / f

Z_points1.append(Z21)
X_points1.append(X21)
Y_points1.append(Y21)

Z_points2.append(Z22)
X_points2.append(X22)
Y_points2.append(Y22)

# Calculate the euclidan distance
dist21 = np.sqrt(X21**2 + Y21**2 + Z21**2)
print(dist21)

dist22 = np.sqrt(X22**2 + Y22**2 + Z22**2)
print(dist22)

speed11 = abs((dist21 - dist11)) * fps
speed12 = abs((dist22 - dist12)) * fps

# Creates a bounding box with information generated from Yolov5 bounding boxes
font = cv.FONT_HERSHEY_SIMPLEX
imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin21, ymax21), (xmax21, ymin21), (0, 0, 255), 2)
imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(speed11, 3)} m/s', (xmin21, ymin21 - 5), font, 0.8, (0, 0, 255), 2)
imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin22, ymax22), (xmax22, ymin22), (0, 0, 255), 2)
imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(speed12, 3)} m/s', (xmin22, ymin22 - 5), font, 0.8, (0, 0, 255), 2)
cv.imwrite(f'../../result_images/{folder_name}/estimated_{str(img_nr)}.png', imgs[img_nr])

### IMG 3
img_nr = 4
# Round to integers
xmax31 = res.xyxy[img_nr].iloc[1]['xmax']
xmax31 = int(round(xmax31))
ymax31 = res.xyxy[img_nr].iloc[1]['ymax']
ymax31 = int(round(ymax31))
xmin31 = res.xyxy[img_nr].iloc[1]['xmin']
xmin31 = int(round(xmin31))
ymin31 = res.xyxy[img_nr].iloc[1]['ymin']
ymin31 = int(round(ymin31))

# Round to integers
xmax32 = res.xyxy[img_nr].iloc[0]['xmax']
xmax32 = int(round(xmax32))
ymax32 = res.xyxy[img_nr].iloc[0]['ymax']
ymax32 = int(round(ymax32))
xmin32 = res.xyxy[img_nr].iloc[0]['xmin']
xmin32 = int(round(xmin32))
ymin32 = res.xyxy[img_nr].iloc[0]['ymin']
ymin32 = int(round(ymin32))

# Middle of the bounding boxes
# Fish 1
x3_l1 = (res.xyxy[img_nr].iloc[1]['xmin'] + res.xyxy[img_nr].iloc[1]['xmax']) / 2 # Left camera
y3_l1 = (res.xyxy[img_nr].iloc[1]['ymin'] + res.xyxy[img_nr].iloc[1]['ymax']) / 2 # Left camera

x3_r1 = (res.xyxy[img_nr + 1].iloc[1]['xmin'] + res.xyxy[img_nr + 1].iloc[1]['xmax']) / 2 # Right camera
y3_r1 = (res.xyxy[img_nr + 1].iloc[1]['ymin'] + res.xyxy[img_nr + 1].iloc[1]['ymax']) / 2 # Right camera

# Fish 2
x3_l2 = (res.xyxy[img_nr].iloc[0]['xmin'] + res.xyxy[img_nr].iloc[0]['xmax']) / 2 # Left camera
y3_l2 = (res.xyxy[img_nr].iloc[0]['ymin'] + res.xyxy[img_nr].iloc[0]['ymax']) / 2 # Left camera

x3_r2 = (res.xyxy[img_nr + 1].iloc[0]['xmin'] + res.xyxy[img_nr + 1].iloc[0]['xmax']) / 2 # Right camera
y3_r2 = (res.xyxy[img_nr + 1].iloc[0]['ymin'] + res.xyxy[img_nr + 1].iloc[0]['ymax']) / 2 # Right camera

# Triangulation
Z31 = (baseline * f) / (x3_l1 - x3_r1) / px_size
X31 = (x3_l1 - x3_r1) * px_size * Z31 / f
Y31 = (y3_l1 - y3_r1) * px_size * Z31 / f

Z32 = (baseline * f) / (x3_l2 - x3_r2) / px_size
X32 = (x3_l2 - x3_r2) * px_size * Z32 / f
Y32 = (y3_l2 - y3_r2) * px_size * Z32 / f

Z_points1.append(Z31)
X_points1.append(X31)
Y_points1.append(Y31)

Z_points2.append(Z32)
X_points2.append(X32)
Y_points2.append(Y32)

# Calculate the euclidan distance
dist31 = np.sqrt(X31**2 + Y31**2 + Z31**2)
print(dist31)

dist32 = np.sqrt(X32**2 + Y32**2 + Z32**2)
print(dist32)

speed21 = abs((dist31 - dist21)) * fps
speed22 = abs((dist32 - dist22)) * fps

# Creates a bounding box with information generated from Yolov5 bounding boxes
font = cv.FONT_HERSHEY_SIMPLEX
imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin31, ymax31), (xmax31, ymin31), (0, 0, 255), 2)
imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(speed21, 3)} m/s', (xmin31, ymin31 - 5), font, 0.8, (0, 0, 255), 2)
imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin32, ymax32), (xmax32, ymin32), (0, 0, 255), 2)
imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(speed22, 3)} m/s', (xmin32, ymin32 - 5), font, 0.8, (0, 0, 255), 2)
cv.imwrite(f'../../result_images/{folder_name}/estimated_{str(img_nr)}.png', imgs[img_nr])

### IMG 4
img_nr = 6
# Round to integers
xmax41 = res.xyxy[img_nr].iloc[1]['xmax']
xmax41 = int(round(xmax41))
ymax41 = res.xyxy[img_nr].iloc[1]['ymax']
ymax41 = int(round(ymax41))
xmin41 = res.xyxy[img_nr].iloc[1]['xmin']
xmin41 = int(round(xmin41))
ymin41 = res.xyxy[img_nr].iloc[1]['ymin']
ymin41 = int(round(ymin41))

# Round to integers
xmax42 = res.xyxy[img_nr].iloc[0]['xmax']
xmax42 = int(round(xmax42))
ymax42 = res.xyxy[img_nr].iloc[0]['ymax']
ymax42 = int(round(ymax42))
xmin42 = res.xyxy[img_nr].iloc[0]['xmin']
xmin42 = int(round(xmin42))
ymin42 = res.xyxy[img_nr].iloc[0]['ymin']
ymin42 = int(round(ymin42))

# Middle of the bounding boxes
# Fish 1
x4_l1 = (res.xyxy[img_nr].iloc[1]['xmin'] + res.xyxy[img_nr].iloc[1]['xmax']) / 2 # Left camera
y4_l1 = (res.xyxy[img_nr].iloc[1]['ymin'] + res.xyxy[img_nr].iloc[1]['ymax']) / 2 # Left camera

x4_r1 = (res.xyxy[img_nr + 1].iloc[0]['xmin'] + res.xyxy[img_nr + 1].iloc[0]['xmax']) / 2 # Right camera
y4_r1 = (res.xyxy[img_nr + 1].iloc[0]['ymin'] + res.xyxy[img_nr + 1].iloc[0]['ymax']) / 2 # Right camera

# Fish 2
x4_l2 = (res.xyxy[img_nr].iloc[0]['xmin'] + res.xyxy[img_nr].iloc[0]['xmax']) / 2 # Left camera
y4_l2 = (res.xyxy[img_nr].iloc[0]['ymin'] + res.xyxy[img_nr].iloc[0]['ymax']) / 2 # Left camera

x4_r2 = (res.xyxy[img_nr + 1].iloc[1]['xmin'] + res.xyxy[img_nr + 1].iloc[1]['xmax']) / 2 # Right camera
y4_r2 = (res.xyxy[img_nr + 1].iloc[1]['ymin'] + res.xyxy[img_nr + 1].iloc[1]['ymax']) / 2 # Right camera

# Triangulation
Z41 = (baseline * f) / (x4_l1 - x4_r1) / px_size
X41 = (x4_l1 - x4_r1) * px_size * Z41 / f
Y41 = (y4_l1 - y4_r1) * px_size * Z41 / f

Z42 = (baseline * f) / (x4_l2 - x4_r2) / px_size
X42 = (x4_l2 - x4_r2) * px_size * Z42 / f
Y42 = (y4_l2 - y4_r2) * px_size * Z42 / f

Z_points1.append(Z41)
X_points1.append(X41)
Y_points1.append(Y41)

Z_points2.append(Z42)
X_points2.append(X42)
Y_points2.append(Y42)

# Calculate the euclidan distance
dist41 = np.sqrt(X41**2 + Y41**2 + Z41**2)
print(dist41)

dist42 = np.sqrt(X42**2 + Y42**2 + Z42**2)
print(dist42)

speed31 = abs((dist41 - dist31)) * fps
speed32 = abs((dist42 - dist32)) * fps

# Creates a bounding box with information generated from Yolov5 bounding boxes
font = cv.FONT_HERSHEY_SIMPLEX
imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin41, ymax41), (xmax41, ymin41), (0, 0, 255), 2)
imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(speed31, 3)} m/s', (xmin41, ymin41 - 5), font, 0.8, (0, 0, 255), 2)
imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin42, ymax42), (xmax42, ymin42), (0, 0, 255), 2)
imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(speed32, 3)} m/s', (xmin42, ymin42 - 5), font, 0.8, (0, 0, 255), 2)
cv.imwrite(f'../../result_images/{folder_name}/estimated_{str(img_nr)}.png', imgs[img_nr])

## IMG 5
img_nr = 8

# Round to integers
xmax51 = res.xyxy[img_nr].iloc[1]['xmax']
xmax51 = int(round(xmax51))
ymax51 = res.xyxy[img_nr].iloc[1]['ymax']
ymax51 = int(round(ymax51))
xmin51 = res.xyxy[img_nr].iloc[1]['xmin']
xmin51 = int(round(xmin51))
ymin51 = res.xyxy[img_nr].iloc[1]['ymin']
ymin51 = int(round(ymin51))

# Round to integers
xmax52 = res.xyxy[img_nr].iloc[0]['xmax']
xmax52 = int(round(xmax52))
ymax52 = res.xyxy[img_nr].iloc[0]['ymax']
ymax52 = int(round(ymax52))
xmin52 = res.xyxy[img_nr].iloc[0]['xmin']
xmin52 = int(round(xmin52))
ymin52 = res.xyxy[img_nr].iloc[0]['ymin']
ymin52 = int(round(ymin52))

# Middle of the bounding boxes
# Fish 1
x5_l1 = (res.xyxy[img_nr].iloc[1]['xmin'] + res.xyxy[img_nr].iloc[1]['xmax']) / 2 # Left camera
y5_l1 = (res.xyxy[img_nr].iloc[1]['ymin'] + res.xyxy[img_nr].iloc[1]['ymax']) / 2 # Left camera

x5_r1 = (res.xyxy[img_nr + 1].iloc[0]['xmin'] + res.xyxy[img_nr + 1].iloc[0]['xmax']) / 2 # Right camera
y5_r1 = (res.xyxy[img_nr + 1].iloc[0]['ymin'] + res.xyxy[img_nr + 1].iloc[0]['ymax']) / 2 # Right camera

# Fish 2
x5_l2 = (res.xyxy[img_nr].iloc[0]['xmin'] + res.xyxy[img_nr].iloc[0]['xmax']) / 2 # Left camera
y5_l2 = (res.xyxy[img_nr].iloc[0]['ymin'] + res.xyxy[img_nr].iloc[0]['ymax']) / 2 # Left camera

x5_r2 = (res.xyxy[img_nr + 1].iloc[1]['xmin'] + res.xyxy[img_nr + 1].iloc[1]['xmax']) / 2 # Right camera
y5_r2 = (res.xyxy[img_nr + 1].iloc[1]['ymin'] + res.xyxy[img_nr + 1].iloc[1]['ymax']) / 2 # Right camera

# Triangulation
Z51 = (baseline * f) / (x5_l1 - x5_r1) / px_size
X51 = (x5_l1 - x5_r1) * px_size * Z51 / f
Y51 = (y5_l1 - y5_r1) * px_size * Z51 / f

Z52 = (baseline * f) / (x5_l2 - x5_r2) / px_size
X52 = (x5_l2 - x5_r2) * px_size * Z52 / f
Y52 = (y5_l2 - y5_r2) * px_size * Z52 / f

Z_points1.append(Z51)
X_points1.append(X51)
Y_points1.append(Y51)

Z_points2.append(Z52)
X_points2.append(X52)
Y_points2.append(Y52)

# Calculate the euclidan distance
dist51 = np.sqrt(X51**2 + Y51**2 + Z51**2)
print(dist51)

dist52 = np.sqrt(X52**2 + Y52**2 + Z52**2)
print(dist52)

speed41 = abs((dist51 - dist41)) * fps
speed42 = abs((dist52 - dist42)) * fps

# Creates a bounding box with information generated from Yolov5 bounding boxes
font = cv.FONT_HERSHEY_SIMPLEX
imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin51, ymax51), (xmax51, ymin51), (0, 0, 255), 2)
imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(speed41, 3)} m/s', (xmin51, ymin51 - 5), font, 0.8, (0, 0, 255), 2)
imgs[img_nr] = cv.rectangle(imgs[img_nr], (xmin52, ymax52), (xmax52, ymin52), (0, 0, 255), 2)
imgs[img_nr] = cv.putText(imgs[img_nr], f'{round(speed42, 3)} m/s', (xmin52, ymin52 - 5), font, 0.8, (0, 0, 255), 2)
cv.imwrite(f'../../result_images/{folder_name}/estimated_{str(img_nr)}.png', imgs[img_nr])

# Plotting
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_points1, Y_points1, Z_points1)
n = [1, 2, 3, 4, 5]
for i in range(len(n)):
    ax.text(X_points1[i], Y_points1[i], Z_points1[i], f'{str(i)}')

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

plt.savefig(f'../../result_images/{folder_name}/3D-trajectory-{start_img}-1.png')

# Plotting
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_points2, Y_points2, Z_points2)
n = [1, 2, 3, 4, 5]
for i in range(len(n)):
    ax.text(X_points2[i], Y_points2[i], Z_points2[i], f'{str(i)}')

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

plt.savefig(f'../../result_images/{folder_name}/3D-trajectory-{start_img}-2.png')
