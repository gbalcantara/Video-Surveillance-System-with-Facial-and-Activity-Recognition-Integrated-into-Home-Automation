"""Access IP Camera in Python OpenCV"""

import cv2
import argparse
import time

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
	stream = cv2.VideoCapture('rtsp://192.168.0.143:8554/unicast')
	#time.sleep(2)

else:
	stream = cv2.VideoCapture(args["video"])
#stream.set(cv2.CAP_PROP_FPS, 2)

# Use the next line if your camera has a username and password
# stream = cv2.VideoCapture('protocol://username:password@IP:port/1')  

if stream.isOpened() == False:
    print ("VideoCapture failed")

while True:

    r, f = stream.read()
    cv2.imshow('IP Camera stream',f)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
