import numpy as np
import cv2 as cv

""" 
Calibrated at NTNU ITK lab
The cameras are calibrated with Matlab calibration tool
Explanation from found in https://se.mathworks.com/help/vision/ug/camera-calibration.html

Camera matrix form:
[fx 0 cx]
[0 fy cy]
[0 0 1]


"""

def camera_params_lab():
    # From Matlab camera calibrator 

    K1 = np.array( # Left camera matrix
        [[338.20, 0, 634.71],
        [0, 337.85, 511.30],
        [0, 0, 1]]
    )

    dist_coeff1 = np.array([0, 0, 0.0131, -0.0053, 0])

    K2 = np.array( # Right camera matrix
        [[352.03 , 0, 632.25],
        [0, 352.35, 489.13],
        [0, 0, 1]]
    )

    dist_coeff2 = np.array([0, 0, 0.0301,-0.0105, 0])

    R = np.array( # Rotation matrix of right camera
        [[0.9473, -0.3133 , -0.0673],
        [0.3139, 0.9494, -0.0022],
        [0.0646, -0.0191, 0.9977]]
    ) 

    T = np.array( # Translation of right camera
        [[-239.7760], 
        [34.3002],
        [-51.5370]]
        )

    img_size = (1288, 964) # Image capturing size

    # Projection matrices
    R0 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
        ])

    T0 = np.array([
        [0],
        [0],
        [0]
    ])

    R0T0 = np.hstack((R0, T0))
    R1T1 = np.hstack((R, T))

    P1 = np.matmul(K1, R0T0) # Left projection matrix
    P2 = np.matmul(K2, R1T1)

    return P1, P2, dist_coeff1, dist_coeff2