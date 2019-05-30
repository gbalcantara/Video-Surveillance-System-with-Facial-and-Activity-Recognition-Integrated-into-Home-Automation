import cv2 
import os

def my_main_function():
	while(True):
		cap = cv2.VideoCapture('rtsp://192.168.0.143:8554/unicast')
		#frameRate = cap.get(5) #frame rate
		frameRate = 25
		x=1
		savepath = 'cframe'
		while(cap.isOpened()):
			frameId = cap.get(1) #current frame number
			ret, frame = cap.read()
			if (ret != True):
		  		break
			os.chdir(savepath)
			cv2.imwrite('cframe'+'.jpg', frame)
			os.chdir('..')
			print("image",x)
			x=x+1
		cap.release()
		print ("Done!")

if __name__=='__main__':
	try:
		my_main_function()
	except:
		my_main_function()
	else:
		my_main_function()
