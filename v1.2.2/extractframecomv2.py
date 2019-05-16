import cv2 
import os

cap1 = cv2.VideoCapture('rtsp://192.168.0.143:8554/unicast')
cap2 = cv2.VideoCapture('rtsp://192.168.0.144:8554/unicast')
#frameRate = cap.get(5) #frame rate
frameRate = 25
x=1
y=1
savepath = 'cframe'
savepathh = 'cframe2'
while(cap1.isOpened() and cap2.isOpened()):
	frameId1 = cap1.get(1) #current frame number
	ret1, frame1 = cap1.read()
	
	frameId2 = cap2.get(1) #current frame number
	ret2, frame2 = cap2.read()

	if (ret1 != True):
  		break
	if (frameId1 % frameRate == 0):
		#cwd = os.getcwd()
		#print(cwd)
		os.chdir(savepath)
		cv2.imwrite('cframe'+'.jpg', frame1)
		os.chdir('..')
		print("image143",x)
		x=x+1

	if (ret2 != True):
  		break
	if (frameId2 % frameRate == 0):
		#cwd = os.getcwd()
		#print(cwd)
		os.chdir(savepathh)
		cv2.imwrite('cframe'+'.jpg', frame2)
		os.chdir('..')
		print("image144",y)
		y=y+1

cap2.release()
cap1.release()
print ("Done!")
