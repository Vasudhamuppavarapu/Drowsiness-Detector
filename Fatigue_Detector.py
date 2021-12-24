#python drowniness_yawn.py --webcam webcam_index

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

def alarm(msg):
    global alert1
    global al_top
    global drowsi

    while alert1:
        print('call')
        varS = 'espeak "'+msg+'"'
        os.system(varS)

    if al_top:
        print('call')
        drowsi = True
        varS = 'espeak "' + msg + '"'
        os.system(varS)
        drowsi = False

def eye_dist_function(varEyeDistance):
    varA = dist.euclidean(varEyeDistance[1], varEyeDistance[5])
    varB = dist.euclidean(varEyeDistance[2], varEyeDistance[4])

    varC = dist.euclidean(varEyeDistance[0], varEyeDistance[3])

    varEyeDistance = (varA + varB) / (2.0 * varC)

    return varEyeDistance

def dif_eye_function(varEyeOpen):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    eyeL = varEyeOpen[lStart:lEnd]
    eyeR = varEyeOpen[rStart:rEnd]

    eyeL = eye_dist_function(eyeL)
    eyeR = eye_dist_function(eyeR)

    varEyeDistance = (eyeL + eyeR) / 2.0
    return (varEyeDistance, eyeL, eyeR)

def mouth_function(varEyeOpen):
    upper_mou = varEyeOpen[50:53]
    upper_mou = np.concatenate((upper_mou, varEyeOpen[61:64]))

    btm_mou = varEyeOpen[56:59]
    btm_mou = np.concatenate((btm_mou, varEyeOpen[65:68]))

    btm_m = np.mean(upper_mou, axis=0)
    btm_mo = np.mean(btm_mou, axis=0)

    distance = abs(btm_m[1] - btm_mo[1])
    return distance


ap = argparse.ArgumentParser()
ap.add_argument("-k", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
alert1 = False
al_top = False
drowsi = False
COUNTER = 0

print("-> Loading the eyePredict and eyeDetect...")

eyeDetect = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")    #Faster but less accurate
eyePredict = dlib.shape_predictor('haarcascades/shape_predictor_68_face_landmarks.dat')


print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:

    drowsi_framing = vs.read()
    drowsi_framing = imutils.resize(drowsi_framing, width=450)
    gray = cv2.cvtColor(drowsi_framing, cv2.COLOR_BGR2GRAY)

 
    rects = eyeDetect.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)


    for (j, y, k, h) in rects:
        rect = dlib.rectangle(int(j), int(y), int(j + k),int(y + h))
        
        varEyeOpen = eyePredict(gray, rect)
        varEyeOpen = face_utils.shape_to_np(varEyeOpen)

        varEyeDistance = dif_eye_function(varEyeOpen)
        varEyeDistance = varEyeDistance[0]
        eyeL = varEyeDistance [1]
        eyeR = varEyeDistance[2]

        distance = mouth_function(varEyeOpen)

        hullL = cv2.convexHull(eyeL)
        hullR = cv2.convexHull(eyeR)
        cv2.drawContours(drowsi_framing, [hullL], -1, (0, 255, 0), 1)
        cv2.drawContours(drowsi_framing, [hullR], -1, (0, 255, 0), 1)

        lip = varEyeOpen[48:60]
        cv2.drawContours(drowsi_framing, [lip], -1, (0, 255, 0), 1)

        if varEyeDistance < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if alert1 == False:
                    alert1 = True
                    t = Thread(target=alarm, args=('Fatigue Detection, please be awake to prevent accident',))
                    t.deamon = True
                    t.start()

                cv2.putText(drowsi_framing, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            alert1 = False

        if (distance > YAWN_THRESH):
                cv2.putText(drowsi_framing, "Yawn Alert", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if al_top == False and drowsi == False:
                    al_top = True
                    t = Thread(target=alarm, args=('Fatigue Detected, take some fresh air please',))
                    t.deamon = True
                    t.start()
        else:
            al_top = False

        cv2.putText(drowsi_framing, "varEyeDistance: {:.2f}".format(varEyeDistance), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(drowsi_framing, "YAWN: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.imshow("Frame", drowsi_framing)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
