import numpy as np
import cv2 as cv

""" 
Calibrated at SINTEF Ocean labs
The cameras are calibrated with Matlab calibration tool
Explanation from found in https://se.mathworks.com/help/vision/ug/camera-calibration.html

Camera matrix form:
[fx 0 cx]
[0 fy cy]
[0 0 1]


"""

def camera_params_sintef():
    # From Matlab camera calibrator 

    K1 = np.array(
        [[335.02, 0, 634.23],
        [0, 334.61, 513.80],
        [0, 0, 1]]
    )

    dist_coeff1 = np.array([0, 0, 0.0076, 7.6808e-04, 0]) # Distortion coefficients for left camera 

    K2 = np.array(
        [[363.83, 0, 628.77],
        [0, 364.17, 488.85],
        [0, 0, 1]]
    )

    dist_coeff2 = np.array([0, 0, 0.0272, -0.0035, 0]) # Distortion coefficients for right camera

    R = np.array( # Rotation matrix of right camera
        [[0.9471, -0.3105, -0.0811],
        [0.3114, 0.9503, -0.0024],
        [0.0779, -0.0230, 0.9967]]
    ) 

    T = np.array([
        [-213.15], 
        [18.79],
        [-38.37]
    ]) # Translation of right camera

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

    Q = np.array( # Extracted from projection_params
        [[ 1.00000000e+00,  0.00000000e+00, 0.00000000e+00, -3.68279987e+02],
        [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, -5.41326801e+02],
        [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.49390000e+02],
        [ 0.00000000e+00, 0.00000000e+00, 4.60003581e-03, -0.00000000e+00]]
        )
    return P1, P2, dist_coeff1, dist_coeff2