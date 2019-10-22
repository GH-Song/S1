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

def mouth_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[7])
	B = dist.euclidean(eye[3], eye[5])
 
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[4])
 
	# compute the eye aspect ratio
    mar = (A + B) / (2.0 * C)
 
	# return the eye aspect ratio
	return mar

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(outerM_s, outerM_e) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
inM_s, inM_e) = face_utils.FACIAL_LANDMARKS_IDXS["innermouth"]


		leftEye = shape[outerM_s:outerM_e]
		Innermouth = shape[inM_s:inM_e]

        Inner_mar = mouth_aspect_ratio(Innermouth)