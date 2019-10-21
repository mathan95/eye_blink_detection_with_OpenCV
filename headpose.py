# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:43:41 2019

@author: Mathan
"""

import dlib
import cv2
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

################################
################################
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                         
                        ])
##################################
#################################

cam = cv2.VideoCapture(0)
color_green = (0,255,0)
line_width = 3
win = dlib.image_window()
det_no=1
dets=[]
while True:
    ret_val, img = cam.read()
    
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #rgb_image=cv2.imread("untitled.png")
    #####################################
    ####################################
    size=rgb_image.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype = "double"
                            )
    ####################################
    ###################################
    
    if (det_no%3==0):
        dets = detector(rgb_image)
        det_no=1
    det_no+=1
    for k, det in enumerate(dets):
        cv2.rectangle(rgb_image,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
        shape = predictor(rgb_image, det)
        ################################
        #################################
#        image_points = np.array([
#                            (shape.part(30).x, shape.part(30).y),     # Nose tip
#                            (shape.part(8).x, shape.part(8).y),     # Chin
#                            (shape.part(36).x, shape.part(36).y),     # Left eye left corner
#                            (shape.part(45).x, shape.part(45).y),     # Right eye right corne
#                            (shape.part(48).x, shape.part(48).y),     # Left Mouth corner
#                            (shape.part(54).x, shape.part(54).y)      # Right mouth corner
#                        ], dtype="double")
#        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
#        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
#        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
#        
#        
#        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
#        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
#        
#        
#        cv2.line(rgb_image, p1, p2, (255,0,0), 2)
#        ####################################
#        ###################################
##        for i in range(0,5):
##            cv2.circle(img,(shape.part(i).x, shape.part(i).y),5, (0,0,255), -1)
#        
#        
        ##############################
        #############################
        ###pupil detection
        ##############################
        roi = rgb_image[shape.part(38).y: shape.part(40).y,shape.part(36).x: shape.part(39).x]
        rows, cols, _ = roi.shape
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

        _, threshold = cv2.threshold(gray_roi, 40, 255, cv2.THRESH_BINARY_INV)
        _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
    
            #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
#            cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
#            cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
            break
        
        rgb_image[shape.part(38).y: shape.part(40).y,shape.part(36).x: shape.part(39).x]=roi
        ##############################
        #############################
        #############################
        
        roi2 = rgb_image[shape.part(44).y: shape.part(46).y,shape.part(42).x: shape.part(45).x]
        rows2, cols2, _ = roi2.shape
        gray_roi2 = cv2.cvtColor(roi2, cv2.COLOR_RGB2GRAY)
        gray_roi2 = cv2.GaussianBlur(gray_roi2, (7, 7), 0)

        _, threshold2 = cv2.threshold(gray_roi2, 40, 255, cv2.THRESH_BINARY_INV)
        _, contours2, _ = cv2.findContours(threshold2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours2 = sorted(contours2, key=lambda x: cv2.contourArea(x), reverse=True)

        for cnt in contours2:
            (x, y, w, h) = cv2.boundingRect(cnt)
    
            #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
            cv2.rectangle(roi2, (x, y), (x + w, y + h), (255, 0, 0), 2)
#            cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
#            cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
            break
        
        rgb_image[shape.part(44).y: shape.part(46).y,shape.part(42).x: shape.part(45).x]=roi2
        ###########################
        ##########################
        ########################
        
        
        win.clear_overlay()
        win.set_image(rgb_image)
        #win.add_overlay(shape)
        
#    win.add_overlay(dets)
#    cv2.imshow('my webcam', img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cam.release()
cv2.destroyAllWindows()
