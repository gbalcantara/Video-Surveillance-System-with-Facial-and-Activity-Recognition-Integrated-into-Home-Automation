from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import cv2 
import math
import datetime
import os
import argparse
import imutils
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
#ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
#ap.add_argument("-e", "--encodings", required=True,
#	help="path to serialized db of facial encodings")
#ap.add_argument("-d", "--detection-method", type=str, default="cnn",
#	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

if args.get("video", None) is None:
	cap = cv2.VideoCapture('rtsp://192.168.0.143:8554/unicast')

else:
	cap = cv2.VideoCapture(args["video"])


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


frameRate = cap.get(5) #frame rate
#frameRate = 25
static_back = None
savepath = 'images'
try:
    os.mkdir(savepath)
except OSError:
    pass
x=1
while(cap.isOpened()):
	frameId = cap.get(1) #current frame number
	ret, frame = cap.read()
	if (ret != True):
		break
	if (frameId % math.floor(frameRate) == 0):
        #currentDT = datetime.datetime.now()
        #timetime = currentDT.strftime("%H:%M:%S")
        #filename = timetime + ".jpg";x+=1
        #os.chdir(savepath)
        #cv2.imwrite(filename, frame)
        #os.chdir('..')

        # load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
	#image = cv2.imread(imagePath)
		frame = imutils.resize(frame, width=min(400, frame.shape[1]))
		orig = frame.copy()
 
	# detect people in the image
		(rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
			padding=(8, 8), scale=1.05)
 
	# draw the original bounding boxes
		for (x, y, w, h) in rects:
			cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
 
	# draw the final bounding boxes
		for (xA, yA, xB, yB) in pick:
			cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
 
	# show some information on the number of bounding boxes
	#filename = imagePath[imagePath.rfind("/") + 1:]
		print("[INFO]: {} original boxes, {} after suppression".format(
			len(rects), len(pick)))
 
	# show the output images
		#cv2.imshow("Before NMS", orig)
		cv2.imshow("After NMS", frame)
		#cv2.waitKey(0)
        # if q entered whole process will stop 
	key = cv2.waitKey(1) 
	if key == ord('q'): 
             # if something is movingthen it append the end time of movement 
		break

cap.release()
print ("Done!")
