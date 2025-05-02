
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
from pygame import mixer
import argparse
import imutils
from time import sleep
import dlib
import cv2
import winsound
import pyfirmata
# from pyfirmata import pyfirmata, util
# from pyfirmata import Arduino


port = 'COM3'
board = pyfirmata.Arduino(port)
buzzer=7
motor=5
c=pyfirmata.util.Iterator(board)
c.start()

mixer.init()
sound = mixer.Sound("alarm.wav")


# Calculation of EAR
def eye_aspect_ratio(eye):

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear
def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

# live landmark detection
def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


# calculation of MAR
def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])

def mouth_open(image):
    landmarks = get_landmarks(image)
    
    if landmarks == "error":
        return image, 0
    
    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance





# Declaration of threshold value and the number of frames eyes should be below threshold to turn off the alarm
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20

# Same for yawn count
yawns = 0
count=0
yawn_thresh=3
yawn_status = False
MOUTH_AR_CONSEC_FRAMES = 5


# initialize the frame counter
COUNTER = 0
ALARM_ON = False

# initialize dlib's face detector 
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Extract indexes of facial landmarks for both eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
cap = cv2.VideoCapture(0)


while True:
   # Convert the frame into grayscale
   ret, frame = cap.read()
   frame = cv2.resize(frame, (0,0), fx=0.80, fy=0.80)
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
   rects = detector(gray, 0)    

   image_landmarks, lip_distance = mouth_open(frame)  
   prev_yawn_status = yawn_status    
 
  
   for rect in rects:
       # determine the facial landmarks for the face region
       shape = predictor(gray, rect)
       shape = face_utils.shape_to_np(shape)

       # extracting the left and right eye coordinates
       leftEye = shape[lStart:lEnd]
       rightEye = shape[rStart:rEnd]
       leftEAR = eye_aspect_ratio(leftEye)
       rightEAR = eye_aspect_ratio(rightEye)

       # average the eye aspect ratio together for both eyes
       ear = (leftEAR + rightEAR) / 2.0

       leftEyeHull = cv2.convexHull(leftEye)
       rightEyeHull = cv2.convexHull(rightEye)
       cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
       cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
#-----------------------------------------------------------------------------------------------------------        
#MAR
       if lip_distance > 30:
           count += 1
           if count >= MOUTH_AR_CONSEC_FRAMES:
            yawn_status = True 
            output_text = " Yawn Count: " + str(yawns + 1)
            cv2.putText(frame, output_text, (0,350),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,127),2)
       else:
           yawn_status = False 
           count=0
        
       if prev_yawn_status == True and yawn_status == False:
           yawns += 1
#------------------------------------------------------------------------------------------------------------
# EAR
       # Check if Eye Aspect Ratio is below threshold value
       if ear < EYE_AR_THRESH:
           COUNTER += 1

           # Play alarm if eyes were closed for sufficient amount of time
           if COUNTER >= EYE_AR_CONSEC_FRAMES:
               
             board.digital[buzzer].write(1)
             board.digital[motor].write(1)
             if not ALARM_ON:
                ALARM_ON = True
                sound.play()

           else:
               board.digital[buzzer].write(0)
               board.digital[motor].write(0)

            
               # Display Alert
               cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

       else:
           COUNTER = 0
           ALARM_ON = False

       if yawns>=yawn_thresh:
           cv2.putText(frame, "Driver is Yawning a lot", (0,150),cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)

       # Display Eye Aspect Ratio
       cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    



   # show the frame
   #cv2.imshow('Live Landmarks', image_landmarks )
   cv2.imshow("Frame", frame)
   key = cv2.waitKey(1) & 0xFF

   # Press q to exit
   if key == ord("q"):
       break

cap.release()
cv2.destroyAllWindows()







