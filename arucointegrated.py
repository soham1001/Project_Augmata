#Robotics Club - IITG

import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import math
import os
import time
import requests
from bs4 import BeautifulSoup
from objloader_simple import *




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
    cap1 = cv2.VideoCapture(0)
    while(1):
        _, frame = cap1.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        #Get the params for the dictionary
        params = aruco.DetectorParameters_create()
        #Dectection
        #detectMarkers(inp, dict, corners, ids, params, rejetcs)
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict,
                                                    parameters = params)
        if show == 0:
            cap1.release()
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
    cap1.release()
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

def render(img, obj, projection, model, color=False):
    """
    By juang - github - https://github.com/juangallostra/augmented-reality
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (0, 69, 255))
        else:
            color = face[-1]
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def projection_matrix(camera_parameters, homography):
    """
    By juang - github - https://github.com/juangallostra/augmented-reality
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    #print(rot_1)
    return np.dot(camera_parameters, projection)

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

def showmatch():
    result = requests.get("http://www.espncricinfo.com/series/18645/game/1144171/south-africa-vs-sri-lanka-5th-odi-sl-in-sa-2018-19")
    src = result.content
    soup = BeautifulSoup(src,'lxml')

    content_last_ball = soup.find('div', {'class': 'inner-wrapper'})
    content_teams_ab = soup.find_all('span', {'class': 'cscore_name cscore_name--abbrev'})
    content_teams = soup.find_all('span', {'class': 'cscore_name cscore_name--long'})
    content_score = soup.find_all('div', {'class': 'cscore_score'})

    over = content_last_ball.find('div', {'class': 'time-stamp'})
    commentry = content_last_ball.find('div', {'class': 'description'})

    return [over.text+" overs: "+commentry.text, content_teams_ab, content_teams, content_score]

def multi_marker_pose():
    '''
    Function to perform pose estimation on ArUco markers
    Return  : rvec_list : list of estimated rotation vectors
              tvec_list : list of estimated translation vectors
    '''
    cap1 = cv2.VideoCapture(0)
    print('Calibrating....')
    mtx, dist, _, _ = calib()
    print('Calibrated')
    dir_name = os.getcwd()
    obj_1 = OBJ(os.path.join(dir_name, 'models/1.obj'), swapyz=True)
    obj_2 = OBJ(os.path.join(dir_name, 'models/2.obj'), swapyz=True)
    obj_3 = OBJ(os.path.join(dir_name, 'models/3.obj'), swapyz=True)
    obj_4 = OBJ(os.path.join(dir_name, 'models/4.obj'), swapyz=True)
    obj_5 = OBJ(os.path.join(dir_name, 'models/5.obj'), swapyz=True)
    obj_6 = OBJ(os.path.join(dir_name, 'models/6.obj'), swapyz=True)
    obj_7 = OBJ(os.path.join(dir_name, 'models/7.obj'), swapyz=True)
    obj_8 = OBJ(os.path.join(dir_name, 'models/8.obj'), swapyz=True)
    obj_9 = OBJ(os.path.join(dir_name, 'models/9.obj'), swapyz=True)
    obj_10 = OBJ(os.path.join(dir_name, 'models/10.obj'), swapyz=True)
    obj_11 = OBJ(os.path.join(dir_name, 'models/11.obj'), swapyz=True)
    obj_12 = OBJ(os.path.join(dir_name, 'models/12.obj'), swapyz=True)
    obj_13 = OBJ(os.path.join(dir_name, 'models/13.obj'), swapyz=True)
    obj_14 = OBJ(os.path.join(dir_name, 'models/14.obj'), swapyz=True)
    obj_15 = OBJ(os.path.join(dir_name, 'models/15.obj'), swapyz=True)
    obj_16 = OBJ(os.path.join(dir_name, 'models/16.obj'), swapyz=True)
    obj_17 = OBJ(os.path.join(dir_name, 'models/17.obj'), swapyz=True)
    obj_18 = OBJ(os.path.join(dir_name, 'models/18.obj'), swapyz=True)
    obj_19 = OBJ(os.path.join(dir_name, 'models/19.obj'), swapyz=True)
    obj_20 = OBJ(os.path.join(dir_name, 'models/20.obj'), swapyz=True)

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    params = aruco.DetectorParameters_create()
    ref_1 = cv2.imread(os.path.join(dir_name, 'markers/m1.jpg'), 0)
    corners_src1, ids_src1, rejected_src1 = aruco.detectMarkers(ref_1, aruco_dict,
                                            parameters = params)
    ref_2 = cv2.imread(os.path.join(dir_name, 'markers/m2.jpg'), 0)
    corners_src2, ids_src2, rejected_src2 = aruco.detectMarkers(ref_2, aruco_dict,
                                            parameters = params)
    ref_3 = cv2.imread(os.path.join(dir_name, 'markers/m3.jpg'), 0)
    corners_src3, ids_src3, rejected_src3 = aruco.detectMarkers(ref_3, aruco_dict,
                                            parameters = params)
    ref_4 = cv2.imread(os.path.join(dir_name, 'markers/m4.jpg'), 0)
    corners_src4, ids_src4, rejected_src4 = aruco.detectMarkers(ref_4, aruco_dict,
                                            parameters = params)
    ref_5 = cv2.imread(os.path.join(dir_name, 'markers/m5.jpg'), 0)
    corners_src5, ids_src5, rejected_src5 = aruco.detectMarkers(ref_5, aruco_dict,
                                            parameters = params)
    ref_6 = cv2.imread(os.path.join(dir_name, 'markers/m6.jpg'), 0)
    corners_src6, ids_src6, rejected_src6 = aruco.detectMarkers(ref_6, aruco_dict,
                                            parameters = params)
    ref_7 = cv2.imread(os.path.join(dir_name, 'markers/m7.jpg'), 0)
    corners_src7, ids_src7, rejected_src7 = aruco.detectMarkers(ref_7, aruco_dict,
                                            parameters = params)
    ref_8 = cv2.imread(os.path.join(dir_name, 'markers/m8.jpg'), 0)
    corners_src8, ids_src8, rejected_src8 = aruco.detectMarkers(ref_8, aruco_dict,
                                            parameters = params)
    ref_9 = cv2.imread(os.path.join(dir_name, 'markers/m9.jpg'), 0)
    corners_src9, ids_src9, rejected_src9 = aruco.detectMarkers(ref_9, aruco_dict,
                                            parameters = params)
    ref_10 = cv2.imread(os.path.join(dir_name, 'markers/m10.jpg'), 0)
    corners_src10, ids_src10, rejected_src10 = aruco.detectMarkers(ref_10, aruco_dict,
                                            parameters = params)
    ref_11 = cv2.imread(os.path.join(dir_name, 'markers/m11.jpg'), 0)
    corners_src11, ids_src11, rejected_src11 = aruco.detectMarkers(ref_11, aruco_dict,
                                            parameters = params)
    ref_12 = cv2.imread(os.path.join(dir_name, 'markers/m12.jpg'), 0)
    corners_src12, ids_src12, rejected_src12 = aruco.detectMarkers(ref_12, aruco_dict,
                                            parameters = params)
    ref_13 = cv2.imread(os.path.join(dir_name, 'markers/m13.jpg'), 0)
    corners_src13, ids_src13, rejected_src13 = aruco.detectMarkers(ref_13, aruco_dict,
                                            parameters = params)
    ref_14 = cv2.imread(os.path.join(dir_name, 'markers/m14.jpg'), 0)
    corners_src14, ids_src14, rejected_src14 = aruco.detectMarkers(ref_14, aruco_dict,
                                            parameters = params)
    ref_15 = cv2.imread(os.path.join(dir_name, 'markers/m15.jpg'), 0)
    corners_src15, ids_src15, rejected_src15 = aruco.detectMarkers(ref_15, aruco_dict,
                                            parameters = params)
    ref_16 = cv2.imread(os.path.join(dir_name, 'markers/m16.jpg'), 0)
    corners_src16, ids_src16, rejected_src16 = aruco.detectMarkers(ref_16, aruco_dict,
                                            parameters = params)
    ref_17 = cv2.imread(os.path.join(dir_name, 'markers/m17.jpg'), 0)
    corners_src17, ids_src17, rejected_src17 = aruco.detectMarkers(ref_17, aruco_dict,
                                            parameters = params)
    ref_18 = cv2.imread(os.path.join(dir_name, 'markers/m18.jpg'), 0)
    corners_src18, ids_src18, rejected_src18 = aruco.detectMarkers(ref_18, aruco_dict,
                                            parameters = params)
    ref_19 = cv2.imread(os.path.join(dir_name, 'markers/m19.jpg'), 0)
    corners_src19, ids_src19, rejected_src19 = aruco.detectMarkers(ref_19, aruco_dict,
                                            parameters = params)
    ref_20 = cv2.imread(os.path.join(dir_name, 'markers/m20.jpg'), 0)
    corners_src20, ids_src20, rejected_src20 = aruco.detectMarkers(ref_20, aruco_dict,
                                            parameters = params)
    count = 0
    while(1):

        _, frame = cap1.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict,
                                                    parameters = params)
        if np.all(ids != None):
            rvec_list = []
            tvec_list = []
            aruco.drawDetectedMarkers(frame, corners)
            for i in range(len(ids)):
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners[i], 0.05, mtx, dist)
                #aruco.drawAxis(frame, mtx, dist, rvecs[0], tvecs[0], 0.05)
                cv2.putText(frame, str(ids[i][0]),
                                    tuple(corners[i][0][2]),
                                    font, 0.5, (0, 0, 255), 1, 4)
                rvec_list.append(rvecs)
                #print(tvecs[0][0])
                tvec_list.append(tvecs)
            if ids == ids_src1:
                homography, mask = cv2.findHomography(corners_src1[0], corners[0], cv2.RANSAC, 5.0)
                if homography is not None:
                    try:
                        # obtain 3D projection matrix from homography matrix and camera parameters
                        projection = projection_matrix(mtx, homography)
                        # project cube or model
                        frame = render(frame, obj_1, projection, ref_1, False)
                    except:
                        pass
            elif ids == ids_src2:
                homography, mask = cv2.findHomography(corners_src2[0], corners[0], cv2.RANSAC, 5.0)
                if homography is not None:
                    try:
                        # obtain 3D projection matrix from homography matrix and camera parameters
                        projection = projection_matrix(mtx, homography)
                        # project cube or model
                        frame = render(frame, obj_2, projection, ref_2, False)
                    except:
                        pass
            elif ids == ids_src3:
                homography, mask = cv2.findHomography(corners_src3[0], corners[0], cv2.RANSAC, 5.0)
                if homography is not None:
                    try:
                        # obtain 3D projection matrix from homography matrix and camera parameters
                        projection = projection_matrix(mtx, homography)
                        # project cube or model
                        frame = render(frame, obj_3, projection, ref_3, False)
                    except:
                        pass
            elif ids == ids_src4:
                homography, mask = cv2.findHomography(corners_src4[0], corners[0], cv2.RANSAC, 5.0)
                if homography is not None:
                    try:
                        # obtain 3D projection matrix from homography matrix and camera parameters
                        projection = projection_matrix(mtx, homography)
                        # project cube or model
                        frame = render(frame, obj_4, projection, ref_4, False)
                    except:
                        pass
            elif ids == ids_src5:
                homography, mask = cv2.findHomography(corners_src5[0], corners[0], cv2.RANSAC, 5.0)
                if homography is not None:
                    try:
                        # obtain 3D projection matrix from homography matrix and camera parameters
                        projection = projection_matrix(mtx, homography)
                        # project cube or model
                        frame = render(frame, obj_5, projection, ref_5, False)
                    except:
                        pass                        
            elif ids == ids_src6:
                homography, mask = cv2.findHomography(corners_src6[0], corners[0], cv2.RANSAC, 5.0)
                if homography is not None:
                    try:
                        # obtain 3D projection matrix from homography matrix and camera parameters
                        projection = projection_matrix(mtx, homography)
                        # project cube or model
                        frame = render(frame, obj_6, projection, ref_6, False)
                    except:
                        pass
            elif ids == ids_src7:
                homography, mask = cv2.findHomography(corners_src7[0], corners[0], cv2.RANSAC, 5.0)
                if homography is not None:
                    try:
                        # obtain 3D projection matrix from homography matrix and camera parameters
                        projection = projection_matrix(mtx, homography)
                        # project cube or model
                        frame = render(frame, obj_7, projection, ref_7, False)
                    except:
                        pass
            elif ids == ids_src8:
                homography, mask = cv2.findHomography(corners_src8[0], corners[0], cv2.RANSAC, 5.0)
                if homography is not None:
                    try:
                        # obtain 3D projection matrix from homography matrix and camera parameters
                        projection = projection_matrix(mtx, homography)
                        # project cube or model
                        frame = render(frame, obj_8, projection, ref_8, False)
                    except:
                        pass
            elif ids == ids_src9:
                homography, mask = cv2.findHomography(corners_src9[0], corners[0], cv2.RANSAC, 5.0)
                if homography is not None:
                    try:
                        # obtain 3D projection matrix from homography matrix and camera parameters
                        projection = projection_matrix(mtx, homography)
                        # project cube or model
                        frame = render(frame, obj_9, projection, ref_9, False)
                    except:
                        pass
            elif ids == ids_src10:
                homography, mask = cv2.findHomography(corners_src10[0], corners[0], cv2.RANSAC, 5.0)
                if homography is not None:
                    try:
                        # obtain 3D projection matrix from homography matrix and camera parameters
                        projection = projection_matrix(mtx, homography)
                        # project cube or model
                        frame = render(frame, obj_10, projection, ref_10, False)
                    except:
                        pass
            elif ids == ids_src11:
                homography, mask = cv2.findHomography(corners_src11[0], corners[0], cv2.RANSAC, 5.0)
                if homography is not None:
                    try:
                        # obtain 3D projection matrix from homography matrix and camera parameters
                        projection = projection_matrix(mtx, homography)
                        # project cube or model
                        frame = render(frame, obj_11, projection, ref_11, False)
                    except:
                        pass
            elif ids == ids_src12:
                homography, mask = cv2.findHomography(corners_src12[0], corners[0], cv2.RANSAC, 5.0)
                if homography is not None:
                    try:
                        # obtain 3D projection matrix from homography matrix and camera parameters
                        projection = projection_matrix(mtx, homography)
                        # project cube or model
                        frame = render(frame, obj_12, projection, ref_12, False)
                    except:
                        pass
            elif ids == ids_src13:
                homography, mask = cv2.findHomography(corners_src13[0], corners[0], cv2.RANSAC, 5.0)
                if homography is not None:
                    try:
                        # obtain 3D projection matrix from homography matrix and camera parameters
                        projection = projection_matrix(mtx, homography)
                        # project cube or model
                        frame = render(frame, obj_13, projection, ref_13, False)
                    except:
                        pass
            elif ids == ids_src14:
                homography, mask = cv2.findHomography(corners_src14[0], corners[0], cv2.RANSAC, 5.0)
                if homography is not None:
                    try:
                        # obtain 3D projection matrix from homography matrix and camera parameters
                        projection = projection_matrix(mtx, homography)
                        # project cube or model
                        frame = render(frame, obj_14, projection, ref_14, False)
                    except:
                        pass
            elif ids == ids_src15:
                homography, mask = cv2.findHomography(corners_src15[0], corners[0], cv2.RANSAC, 5.0)
                if homography is not None:
                    try:
                        # obtain 3D projection matrix from homography matrix and camera parameters
                        projection = projection_matrix(mtx, homography)
                        # project cube or model
                        frame = render(frame, obj_15, projection, ref_15, False)
                    except:
                        pass
            elif ids == ids_src16:
                homography, mask = cv2.findHomography(corners_src16[0], corners[0], cv2.RANSAC, 5.0)
                if homography is not None:
                    try:
                        # obtain 3D projection matrix from homography matrix and camera parameters
                        projection = projection_matrix(mtx, homography)
                        # project cube or model
                        frame = render(frame, obj_16, projection, ref_16, False)
                    except:
                        pass
            elif ids == ids_src17:
                homography, mask = cv2.findHomography(corners_src17[0], corners[0], cv2.RANSAC, 5.0)
                if homography is not None:
                    try:
                        # obtain 3D projection matrix from homography matrix and camera parameters
                        projection = projection_matrix(mtx, homography)
                        # project cube or model
                        frame = render(frame, obj_17, projection, ref_17, False)
                    except:
                        pass
            elif ids == ids_src18:
                homography, mask = cv2.findHomography(corners_src18[0], corners[0], cv2.RANSAC, 5.0)
                if homography is not None:
                    try:
                        # obtain 3D projection matrix from homography matrix and camera parameters
                        projection = projection_matrix(mtx, homography)
                        # project cube or model
                        frame = render(frame, obj_18, projection, ref_18, False)
                    except:
                        pass
            elif ids == ids_src19:
                homography, mask = cv2.findHomography(corners_src19[0], corners[0], cv2.RANSAC, 5.0)
                if homography is not None:
                    try:
                        # obtain 3D projection matrix from homography matrix and camera parameters
                        projection = projection_matrix(mtx, homography)
                        # project cube or model
                        frame = render(frame, obj_19, projection, ref_19, False)
                    except:
                        pass
            elif ids == ids_src20:
                homography, mask = cv2.findHomography(corners_src20[0], corners[0], cv2.RANSAC, 5.0)
                if homography is not None:
                    try:
                        # obtain 3D projection matrix from homography matrix and camera parameters
                        projection = projection_matrix(mtx, homography)
                        # project cube or model
                        frame = render(frame, obj_20, projection, ref_20, False)
                    except:
                        pass

        img = cv2.imread('clk4.png', 1)
        img1 = cv2.imread('calendar3.png', 1)
        img2 = cv2.imread('tv.png', 1)
        rows2, cols2, _ = img2.shape
        rows, cols, xyz = img.shape
        row, col, xsd = img1.shape
        date = time.ctime()
        if (count % 400==0):
            info,c_t_abs,c_t,c_scores = showmatch()
        img = cv2.putText(img, str(str(date).split(" ")[3]), (15,40), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX , 0.7, (0,0,0),1)
        img2 = cv2.putText(img2, "LIVE" , (20,22), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,255), 1)
        img2 = cv2.putText(img2, c_t[0].text+' VS '+c_t[1].text, (70,22), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1)
        img2 = cv2.putText(img2, c_t_abs[0].text+'      '+c_scores[0].text, (20,44), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0,0,0), 1)
        img2 = cv2.putText(img2, c_t_abs[1].text+'      '+c_scores[1].text, (20,60), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0,0,0), 1)
        # img1 = cv2.putText(img1, str(str(date).split(" ")[4]), (30,85), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX , 0.7, (255,0,0),1)
        # img1 = cv2.putText(img1, str(str(date).split(" ")[1]), (15,60), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX , 0.7, (255,0,0),1)
        # img1 = cv2.putText(img1, str(str(date).split(" ")[2]), (70,60), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX , 0.7, (255,0,0),1)
        frame = cv2.resize(frame,(950,980),interpolation = cv2.INTER_AREA)
        #frame = cv2.putText(frame,voice,(600,400), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX , 0.7, (255,0,0),1)
        modification2 = cv2.addWeighted(img2, 0.8, frame[600:600+rows2, 40:40+cols2], 0.7, 2)
        modification = cv2.addWeighted(img, 0.3, frame[60:60+rows, 40:40+cols], 0.7, 2)
        frame[60:60+rows, 40:40+cols] = modification
        frame[600:600+rows2, 40:40+cols2] = modification2
        count = count+1

        # modification1 = cv2.addWeighted(img1, 0.6, frame[55:55+row,800:800+col], 0.4, 2)
        # frame[40:40+row, 800:800+col] = modification1
        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap1.release()
    cv2.destroyAllWindows()
    return rvec_list, tvec_list

if __name__ == '__main__':
    multi_marker_pose()
