import numpy as np
import cv2
import cv2.aruco as aruco
import glob
from objloader import *

font = cv2.FONT_HERSHEY_COMPLEX

def make_marker(index):
    '''
    Function to create marker from default dict
    and return index marker
    Input   : index (int32)
    Return  : marker (numpy array)
    '''
    if index > 249:
        print('Exceeded dict size')
        return 0
    #Get predefined dictionary - 6x6 size and 250 markers
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    #drawMarker(dict, markerid, output size)
    marker = aruco.drawMarker(aruco_dict, index, 700)
    cv2.imshow('marker', marker)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()
    cv2.imwrite('markers/m'+str(index)+'.jpg', marker)
    return marker

def detect_marker(show):
    '''
    Funtion to perform marker detection
    Input   : show - 0  : return values without display
                   - 1  : return values after displaying
    Return  : corners   : numpy array of corners of markers
              ids       : ids of markers
              rejected  : corners of rejected points
    '''
    cap = cv2.VideoCapture(0)
    while(1):
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        #Get the params for the dictionary
        params = aruco.DetectorParameters_create()
        #Dectection
        #detectMarkers(inp, dict, corners, ids, params, rejetcs)
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict,
                                                    parameters = params)
        if show == 0:
            cap.release()
            return corners, ids, rejected, frame
        detected = aruco.drawDetectedMarkers(frame, corners)
        if np.all(ids != None):
            #print('Detected :',len(ids))
            for i in range(len(ids)):
                cv2.putText(detected, str(ids[i][0]),
                                    tuple(corners[i][0][2]),
                                    font, 0.5, (0, 0, 255), 1, 4)
        cv2.imshow('Detection', detected)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return corners, ids, rejected

def calib():
    '''
    By njanuridh - github - https://github.com/njanirudh/Aruco_Tracker/blob/master/
    Function to perform camera calibration
    Return  : mtx   : camera matrix
              dist  : distortion coefficients
              rvecs  : rotation vectors
              tvecs  : translation vectors
    '''
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('calib_images/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                    imgpoints, gray.shape[::-1],None,None)
    return mtx, dist, rvecs, tvecs

def multi_marker_pose():
    '''
    Function to perform pose estimation on ArUco markers
    Return  : rvec_list : list of estimated rotation vectors
              tvec_list : list of estimated translation vectors
    '''
    cap = cv2.VideoCapture(0)
    print('Calibrating....')
    mtx, dist, _, _ = calib()
    print('Calibrated')
    while(1):

        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        params = aruco.DetectorParameters_create()
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict,
                                                    parameters = params)
        if np.all(ids != None):
            rvec_list = []
            tvec_list = []
            aruco.drawDetectedMarkers(frame, corners)
            for i in range(len(ids)):
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners[i], 0.05, mtx, dist)
                aruco.drawAxis(frame, mtx, dist, rvecs[0], tvecs[0], 0.05)
                cv2.putText(frame, str(ids[i][0]),
                                    tuple(corners[i][0][2]),
                                    font, 0.5, (0, 0, 255), 1, 4)
                rvec_list.append(rvecs)
                tvec_list.append(tvecs)

        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return rvec_list, tvec_list

if __name__ == '__main__':
    multi_marker_pose()