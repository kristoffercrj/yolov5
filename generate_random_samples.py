import cv2 as cv
import numpy as np
import torch
import pandas
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from camera_utils.camera_parameters_sintef import *
import random

# Load yolov5 model
model = torch.hub.load('./', 'custom', path='./checkpoints/best-yolov5l-sintef-640', source='local')

# Camera parameters
P1, P2, dist_coeff1, dist_coeff2 = camera_params_sintef()
baseline = 230e-03 # Baseline of the camera [m]
f = 1.28e-03 # Focal length of the camera [m]
px_size = 3.75e-06 # Pixel size of camera in [m]
fps = 30

randomlist = random.sample(range(0, 2990), 40)
print(randomlist)

for i in range(0, len(randomlist)):

    start_img = randomlist[i]
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

    results.save() # Show predictions
    # print(res.xyxy)

    X_points = []
    Y_points = []
    Z_points = []

# [1701, 2722, 466, 1141, 367, 1954, 1633, 2322, 53, 2838, 2955, 1366, 2413, 1790, 2, 1276, 2677, 6, 2228, 476]

# exp8 = 2322
# exp12 = 1366
# exp14 = 1790
# exp15 = 2
# exp16 = 1276
# exp19 = 2228

