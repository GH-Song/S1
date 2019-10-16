# USAGE
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import datetime
import argparse
import imutils
import time
import dlib
import cv2
 
 
# PATH_predictor = "C:/SongsProjectfiles/REALTIME_facelandmark/Realtime_facial_landmarks/shape_predictor_68_face_landmarks.dat"
PATH_predictor = "Realtime_facial_landmarks/shape_predictor_68_face_landmarks.dat"


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PATH_predictor)

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream().start()
time.sleep(1.0)

# print(face_utils.FACIAL_LANDMARKS_IDXS.items())
# print(face_utils.FACIAL_LANDMARKS_IDXS["mouth"])

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Frame", frame)
    # detect faces in the grayscale frame
    rects = detector(gray, 1)
 
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
       
        # loop over the face parts individually
        i,j = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        for (x, y) in shape[i:j]:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        for (x, y) in shape[28:29]:
            cv2.rectangle(frame,(x-160, y-150), (x+160,y-70), (0, 0, 255), 1 )
        cv2.imshow("Frame", frame)
    
    # show the frame
    # cv2.imshow("Frame", frame)
    # cv2.imshow("Frame", face_utils.visualize_facial_landmarks(frame, shape))
    key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
