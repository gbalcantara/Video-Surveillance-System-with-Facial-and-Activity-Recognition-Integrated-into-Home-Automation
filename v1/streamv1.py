import cv2 
import math
import datetime
import os
import argparse
import psycopg2

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

#frameRate = cap.get(5) #frame rate
#print(frameRate)
frameRate = 25
static_back = None
savepath = 'images'
try:
    os.mkdir(savepath)
except OSError:
    pass
x=1
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    #print(frameId)
    ret, frame = cap.read()
    if (ret != True):
        break
    #if (frameId % math.floor(frameRate) == 0):
    if (frameId % frameRate == 0):
        #currentDT = datetime.datetime.now()
        #timetime = currentDT.strftime("%H:%M:%S")
        #filename = timetime + ".jpg";x+=1
        #os.chdir(savepath)
        #cv2.imwrite(filename, frame)
        #os.chdir('..')

        os.chdir(savepath)
        cv2.imwrite('cframe'+'.jpg', frame)
        os.chdir('..')

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0) 
        if static_back is None:

                #static_back = cv2.imread("firstframe.png") 
                #static_back = cv2.cvtColor(static_back, cv2.COLOR_BGR2GRAY)
                #static_back = cv2.GaussianBlur(static_back, (21, 21), 0) 
                static_back = gray 
                continue
        diff_frame = cv2.absdiff(static_back, gray) 
        thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1] 
        thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2) 
        (_, cnts, _) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        i=0
        for contour in cnts: 
		
                if cv2.contourArea(contour) < 10000: 
                        continue

                (x, y, w, h) = cv2.boundingRect(contour) 
                #cv2.rectangle(frame, (x - 50, y - 50), (x + w + 50, y + h + 50), (0, 255, 0), 3)
                if w>10 and h>10:
                        i+=1
                        new_img=frame[y:y+h,x:x+w]
                        currentDT = datetime.datetime.now()
                        timetime = currentDT.strftime("%H:%M:%S")
                        os.chdir(savepath)
                        cv2.imwrite(timetime+'_'+str(i)+'.jpg', new_img)
                        os.chdir('..')

                        conn = psycopg2.connect(database="dateandtime", user = "gab", password = "admin1234", host = "127.0.0.1", port = "5432")

                        #print ("Opened database successfully")

                        cur = conn.cursor()

                        cur.execute("SELECT id, fn, mdytm, zn, ps, ac from db11")

                        id1 = cur.rowcount
                        id2 = id1 + 1
                        id2 = str(id2)
                        xx = ((x + w + 50)+(x - 50))/2
                        yy = ((y - 50)+(y + h + 50))/2
                        if (xx in range(0, 240)) and (yy in range(0, 720)):
                                dd = 1
                                #print ("Zone",dd)
                        elif (xx in range(241, 500)) and (yy in range(0, 720)):
                                dd = 2
                                #print ("Zone",dd)
                        elif (xx in range(501, 1280)) and (yy in range(0, 260)):
                                dd = 3
                                #print ("Zone",dd)
                        else:
                                dd = 4
                                #print ("Zone",dd)

                        fn = str(timetime) + "_" + str(i)

                        #cur.execute("INSERT INTO DATE_TIME (ID,MDY,DY,TM) VALUES (1,aa,bb,cc)")
                        cur.execute("INSERT INTO db11 (ID,FN,MDYTM,ZN) VALUES (%s,%s,current_timestamp,%s)", (id2,fn,dd))

                        conn.commit()
                        #print ("Records created successfully")
                        conn.close()
                        


        # Displaying image in gray_scale 
        #cv2.imshow("Gray Frame", gray) 

	# Displaying the difference in currentframe to 
	# the staticframe(very first_frame) 
        #cv2.imshow("Difference Frame", diff_frame) 

	# Displaying the black and white image in which if 
	# intencity difference greater than 30 it will appear white 
        #cv2.imshow("Threshold Frame", thresh_frame) 
        cv2.imshow("Color Frame", frame) 
        #key = cv2.waitKey(1) 
        #print("Frame shown")
        # if q entered whole process will stop 
    key = cv2.waitKey(1) 
    if key == ord('q'): 
             # if something is movingthen it append the end time of movement 
             break

cap.release()
print ("Done!")
