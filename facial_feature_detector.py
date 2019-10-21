# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 19:22:00 2019

@author: Mathan
"""

import sys
import dlib
import cv2
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
cam = cv2.VideoCapture(0)
color_green = (0,255,0)
line_width = 3

while True:
    ret_val, img = cam.read()
    
    
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(np.shape(rgb_image))
    #rgb_image=cv2.imread("untitled.png")
    dets = detector(rgb_image)
    
    for k, det in enumerate(dets):
        cv2.rectangle(img,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
        shape = predictor(rgb_image, det)
        for i in range(0,5):
            cv2.circle(img,(shape.part(i).x, shape.part(i).y),5, (0,0,255), -1)
        #win.add_overlay(shape)

    cv2.imshow('my webcam', img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()