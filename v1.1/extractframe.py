import cv2 
import os

cap = cv2.VideoCapture('rtsp://192.168.0.143:8554/unicast')
#frameRate = cap.get(5) #frame rate
frameRate = 25
x=1
savepath = 'cframe'

if cap.isOpened() == False:
    print ("VideoCapture failed")

while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % frameRate == 0):
        os.chdir(savepath)
        cv2.imwrite('cframe'+'.jpg', frame)
        os.chdir('..')
        print("image saved ",x)
        x=x+1

cap.release()
print ("Done!")
