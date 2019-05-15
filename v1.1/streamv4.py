import cv2 
import math
import datetime
import os
import argparse
import psycopg2
import glob
import os.path
import operator
import time
import face_recognition
import pickle
import numpy as np
from queue import Queue
from threading import Thread
from data import DataSet
from processor import process_image
from keras.models import load_model


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
#ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
#ap.add_argument("-e", "--encodings", required=True,
#	help="path to serialized db of facial encodings")
#ap.add_argument("-d", "--detection-method", type=str, default="cnn",
#	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())



def detect_objects(data, model, image, datafaces1):
    action=None
    for _ in range(1):
        images = glob.glob(os.path.join('images','*.jpg'))
        #print('-'*80)
        # Get a random row.
        sample = 0
        #print(sample)
        #print(len(images))
        image = images[sample]
        print(image)
        # Turn the image into an array.
        image2 = cv2.imread(image)
        image_arr = process_image(image, (299, 299, 3))
        image_arr = np.expand_dims(image_arr, axis=0)

        # Predict.
        predictions = None
        predictions = model.predict(image_arr)
        #print(predictions)
        # Show how much we think it's each one.
        #label_predictions = {}
        label_predictions = predictions[0]
        #run, sit, stand = label_predictions.partition(' ')
        run = label_predictions[0]
        print(run)
        sit = label_predictions[1]
        print(sit)
        stand = label_predictions[2] 
        print(stand)
        #for m, label in enumerate(data.classes):
        #    label_predictions[label] = predictions[0][m]

        #sorted_lps = sorted(label_predictions.items(), key=operator.itemgetter(1), reverse=True)
        
        #for n, class_prediction in enumerate(sorted_lps):
            # Just get the top five.
            #if i > 0:
               # break

            #print(image)
            #print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
            #action = class_prediction[0]
        if (run > sit):
            #action = stand
            if (run > stand):
                action = "stand"
            else:
                action = "stand"
        elif (run < sit):
            if (sit > stand):
                action = "sit"
            else:
                action = "stand"
        print("aciton=%s",action)
         
        person = None
        imagepathhh=str(image)
        #print(imagepathhh)
        imagepathh = imagepathhh.partition('/')
        savepath = 'images'
        os.chdir(savepath)
        print(imagepathh)
        imageface=cv2.imread(imagepathh[2])
        os.chdir('..')
        rgb = cv2.cvtColor(imageface, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes corresponding
        # to each face in the input image, then compute the facial embeddings
        # for each face
        print("[INFO] recognizing faces...")
        boxes = face_recognition.face_locations(rgb,
                model='cnn')
        encodings = face_recognition.face_encodings(rgb, boxes)
	
        # initialize the list of names for each face detected
        names = []
	
        # loop over the facial embeddings
        for encoding in encodings:
                # attempt to match each face in the input image to our known
                # encodings
                matches = face_recognition.compare_faces(data["encodings"],
                        encoding)
                name = "Unknown"
	
                # check to see if we have found a match
                if True in matches:
                        # find the indexes of all matched faces then initialize a
                        # dictionary to count the total number of times each face
                        # was matched
                        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                        counts = {}
	
                        # loop over the matched indexes and maintain a count for
                        # each recognized face face
                        for i in matchedIdxs:
                                name = data["names"][i]
                                counts[name] = counts.get(name, 0) + 1
	
                        # determine the recognized face with the largest number of
                        # votes (note: in the event of an unlikely tie Python will
                        # select first entry in the dictionary)
                        name = max(counts, key=counts.get)
	
                # update the list of names
                names.append(name)
                person = name

        # loop over the recognized faces
        #for ((top, right, bottom, left), name) in zip(boxes, names):
                # draw the predicted face name on the image
                #cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                #y = top - 15 if top - 15 > 15 else top + 15
                #cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                       # 0.75, (0, 255, 0), 2)        

    return action, person


def worker(input_q, output_q):
    data = DataSet()
    model = load_model('3motion.hdf5')
    datafaces = pickle.loads(open('thesesencodings1.pickle', "rb").read())

    #fps = FPS().start()
    while True:
        #fps.update()
        frame12 = input_q.get()
        output_q.put(detect_objects(data, model, frame12, datafaces))

    #fps.stop()
    sess.close()

input_q = Queue(1)  # fps is better if queue is higher but then more lags
output_q = Queue()
for i in range(1):
    t = Thread(target=worker, args=(input_q, output_q))
    t.daemon = True
    t.start()


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

#fps = FPS().start()

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
                        x2 = x - 50
                        y2 = y - 50
                        h2 = y + h + 50
                        w2 = x + w + 50
                        new_img=frame[y2:h2,x2:w2]
                        currentDT = datetime.datetime.now()
                        timetime = currentDT.strftime("%H:%M:%S")
                        os.chdir(savepath)
                        cv2.imwrite(timetime+'_'+str(i)+'.jpg', new_img)
                        os.chdir('..')

                        framescan1 = timetime+'_'+str(i)+'.jpg'
                        time.sleep(0.050)
                        input_q.put(framescan1)
                        time.sleep(0.050)
                        #t = time.time()

                        action1 = output_q.get()
                        time.sleep(0.050)
                        print("aciton1=%s",action1)
                        os.chdir(savepath)
                        try: 
                            os.remove(framescan1)
                        except: pass
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
                        cur.execute("INSERT INTO db11 (ID,FN,MDYTM,ZN,AC,PS) VALUES (%s,%s,current_timestamp,%s,%s,%s)", (id2,fn,dd,str(action1[0]),str(action1[1])))

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
#fps.stop()
cap.release()
print ("Done!")
