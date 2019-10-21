import dlib
import cv2
import numpy as np
import time
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

def eye_aspect_ratio(EYE):
    A=np.sqrt((EYE[0][0]-EYE[1][0])**2+(EYE[1][1]-EYE[0][1])**2)
    B=np.sqrt((EYE[2][0]-EYE[3][0])**2+(EYE[2][1]-EYE[3][1])**2)

    C=np.sqrt((EYE[4][0]-EYE[5][0])**2+(EYE[4][1]-EYE[5][1])**2)

    ear = (A + B) / (2.0 * C)
    amp = (A + B) / 2.0
	# return the eye aspect ratio
    return ear, amp
COUNTER=0
Frame=0
time_s=time.time()
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
    
    if (det_no%2==0):
        dets = detector(rgb_image)
        det_no=1
    det_no+=1
    Frame+=1
    
    
    for k, det in enumerate(dets):
        cv2.rectangle(rgb_image,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
        shape = predictor(rgb_image, det)
        #################################
        ##################################
#        time_start=time.time()
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
#        time_finish=time.time()
        #print(time_finish-time_start)
        
        #######################################
        time_s1=time.time()

        i_1=[[shape.part(37).x,shape.part(37).y],[shape.part(41).x,shape.part(41).y],[shape.part(38).x,shape.part(38).y],[shape.part(40).x,shape.part(40).y],[shape.part(36).x,shape.part(36).y],[shape.part(39).x,shape.part(39).y]]
        ear_i1,amp_i1=eye_aspect_ratio(i_1)

        i_2=[[shape.part(43).x,shape.part(43).y],[shape.part(47).x,shape.part(47).y],[shape.part(44).x,shape.part(44).y],[shape.part(46).x,shape.part(46).y],[shape.part(42).x,shape.part(42).y],[shape.part(45).x,shape.part(45).y]]
        ear_i2,amp_i2=eye_aspect_ratio(i_2)
        time_s2=time.time()

        ear=(ear_i1+ear_i2)/2.0
        amp=(amp_i1+amp_i2)/2.0
        
        time_s3=time.time()
        #print('data'+str(time_s2-time_s1))
        #print('analysis'+str(time_s3-time_s1))
        
        if ear<0.25:
            COUNTER+=1
            
        else:
            if COUNTER>=1:
                print(str(COUNTER)+' - blink_detected')
                time_f=time.time()
                print(Frame/(time_f-time_s))
                time_s=time.time()
                Frame=0
            COUNTER = 0
        
    if cv2.waitKey(1) == 27:
        break  # esc to quit
    
cam.release()
cv2.destroyAllWindows()
    
