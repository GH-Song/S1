# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


# check the time
def getCurrentTime(s):
	ss=s/1
	return ss

First_time = getCurrentTime(time.time())

 
# PATH_predictor = "C:/SongsProjectfiles/REALTIME_facelandmark/Realtime_facial_landmarks/shape_predictor_68_face_landmarks.dat"
PATH_predictor = "Realtime_facial_landmarks/shape_predictor_68_face_landmarks.dat"



def mouth_aspect_ratio(mouth):
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6])
    C = dist.euclidean(mouth[3], mouth[5])
 
	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
    D = dist.euclidean(mouth[0], mouth[4])
 
	# compute the mouth aspect ratio
    mar = (A + B + C) / (3.0*D)
 
	# return the mouth aspect ratio
    return mar

# grab the indexes of the facial landmarks for the left and
# right mouth, respectively
(outerM_s, outerM_e) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
inM_s, inM_e = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]
#leftmouth = shape[outerM_s:outerM_e]


threshold = 0.15
FRAMES = 1
# initialize the frame COUNTers and the total number of blinks
COUNTER = [0, 0, 0]
TOTAL = [0,0,0]
TOTAL_SUB = [0,0,0]
color_a = [255,255,255]
color_b = [255,255,255]

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

First_time = getCurrentTime(time.time())
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Frame", frame)
    # detect faces in the grayscale frame
    rects = detector(gray, 1) # 직사각형으로 얼굴(들) 검출
    print(dir(rects))
    print(help(rects))
    print(help(rects.__getitem__))
    import inspect
    inspect.getsource(dlib.dlib.rectangles)
    break
    FACE = [ rects.iloc[0], rects.iloc[1] ]
    
    # loop over the face detections
    for (i, rect) in enumerate(FACE):
        # rect에는 얼굴 하나의 왼쪽 위 오른쪽 아래 좌표
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        # print(rect)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        Innermouth = shape[inM_s:inM_e]
        Inner_mar = mouth_aspect_ratio(Innermouth)
        
        time1 = getCurrentTime(time.time())
        mar1= mouth_aspect_ratio(Innermouth)
        if(getCurrentTime(time.time()) > 0.1):
            mar2= mouth_aspect_ratio(Innermouth)
            dev_mal = abs((mar2-mar1)/0.1)

        print(dev_mal)
        cv2.putText(frame, "devMAR: {:.2f}".format(dev_mal), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if Inner_mar > threshold:
            COUNTER[i] += 1
        else:
            if COUNTER[i] >= FRAMES:
                #입을 열었다가 닫기까지, 연 것으로 인식된 프레임 수가 기준치 이상일 때  
                # TOTAL[i] += 1
                TOTAL_SUB[i] += 1
                COUNTER = [0,0,0]

        Last_time = getCurrentTime(time.time())
            # 2초동안 들린 소리가 누구 것에 가까운지 판단하는 것이 좋다.
        if(Last_time - First_time) > 1:
            for Others in TOTAL_SUB:
                if TOTAL_SUB[i] >= Others:
                    if TOTAL_SUB[i] >= 1:
                        color_a[i]=0
                        color_b[i]=0
                    else:
                        color_a[i]=255
                        color_b[i]=255
                else:
                    color_a[i]=255
                    color_b[i]=255
            First_time = getCurrentTime(time.time())
            TOTAL_SUB = [0,0,0]
        # cv2.putText(frame, "First_time: {:.2f}".format(First_time), (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "Last_time: {:.2f}".format(Last_time), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # loop over the face parts individually
        # cv2.putText(frame, "OPEN: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for (x, y) in shape[inM_s:inM_e]:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        for (x, y) in shape[28:29]:
            cv2.rectangle(frame,(x-160, y-150), (x+160,y-70), (255, 255, 255), -1 )
            cv2.rectangle(frame,(x-160, y-150), (x+160,y-70), (color_a[i], 255, color_b[i]), 3 )
        cv2.imshow("Frame", frame)
        # print("frame is printed")
    
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
