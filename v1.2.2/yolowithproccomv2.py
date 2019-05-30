# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
from queue import Queue
from threading import Thread
from data import DataSet
from processor import process_image
from keras.models import load_model
from keras.models import model_from_json
from imutils import paths
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.preprocessing import image as imageker
from keras.models import model_from_json
import cv2 
import math
import datetime
import os
import argparse
import psycopg2
import glob
import imutils
import os.path
import operator
import time
import pickle
import paho.mqtt.client as paho
import numpy as np


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="cframe/cframe.jpg",
	help="path to input image")
ap.add_argument("-ii", "--imageimage", default="cframe2/cframe.jpg",
	help="path to input image")
ap.add_argument("-y", "--yolo", default="yolo-coco",
	help="base path to YOLO directory")
# ap.add_argument("-m", "--model", default="method2actionmix.model",
# 	help="path to trained model model")
# ap.add_argument("-l", "--labelbin", default="method2actionmix.pickle",
# 	help="path to label binarizer")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


broker="192.168.0.108"
port=1883

def on_connect(client, userdata, flags, rc):
 logging.debug("Connected flags"+str(flags)+"result code "\
 +str(rc)+"client1_id")
 if rc==0:
 	client1.connected_flag=True

def on_publish(client,userdata,result):			 #create function for callback
	print("data published")
	pass

def detect_objects(model, image1):
	action=None
	for _ in range(1):
		# load the image
		#print(image1)
		os.chdir(savepath)
		if os.path.getsize(image1):
		# Execute!
			print('image found')
			print(image1)
			test_image = imageker.load_img(image1, target_size = (64,64))
			test_image = imageker.img_to_array(test_image)
			test_image = np.expand_dims(test_image,axis=0)
		else:
			os.remove(image1)
			break
		os.chdir('..')
 
		# test_image = image.load_img(imagePath, target_size = (64,64))
		# test_image = image.img_to_array(test_image)
		# test_image = np.expand_dims(test_image,axis=0)
		result = model.predict(test_image)
		print(result)
	
		maxElement = np.amax(result)
		#print (maxElement)
		result1 = np.where(result == np.amax(result))
		#print(result1)
		print(result1[1])

		# y_prob = loaded_model.predict(test_image)
		# y_classes = y_prob.argmax(axis=-1)
		# print(y_classes)
		#training_set.class_indices
		# if result1[1] == 0:
		# 	prediction = 'sitting'
		# elif result1[1] == 1:
		# 	prediction = 'sleeping'
		# elif result1[1] == 2:
		# 	prediction = 'standing'
		# elif result1[1] == 3:
		# 	prediction = 'waving'
		# elif result1[1] == 4:
		# 	prediction = 'writing'

		if result1[1] == 0:
			prediction = 'sitting'
		else:
			prediction = 'standing'
		action = prediction

		os.chdir(savepath)
		try: 
			os.remove(framescan1)
		except: 
			pass
		os.chdir('..')

	return action


def worker(input_q, output_q):
	# model = load_model(args["model"])
	# lb = pickle.loads(open(args["labelbin"], "rb").read())
	json_file = open('actionmodel4_mix.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("actionmodel4_mix.h5")

	#fps = FPS().start()
	while True:
		#fps.update()
		frame12 = input_q.get()
		output_q.put(detect_objects(loaded_model, frame12))

	#fps.stop()
	sess.close()

input_q = Queue(1)  # fps is better if queue is higher but then more lags
output_q = Queue()
for i in range(1):
	t = Thread(target=worker, args=(input_q, output_q))
	t.daemon = True
	t.start()

client1= paho.Client("control1")						   #create client object
client1.on_publish = on_publish						  #assign function to callback
client1.on_connect = on_connect

client1.username_pw_set(username="admin",password="RxVm1KSVo0rq")
client1.connect(broker,port)								 #establish connection

savepath = 'action'
savepath1 = 'faces'

cntdwn = 0

while(cntdwn<60):

	# load our input image and grab its spatial dimensions
	image = cv2.imread(args["image"])
	(H, W) = image.shape[:2]

	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# show timing information on YOLO
	#print("[INFO] YOLO took {:.6f} seconds".format(end - start))

	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

	# ensure at least one detection exists
	j=0
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			classs = LABELS[classIDs[i]]
			#print(classs)
			if (classs == 'person'):
				# draw a bounding box rectangle and label on the image
				#color = [int(c) for c in COLORS[classIDs[i]]]
				#cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
				#text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
				#cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
				#	0.5, color, 2)
				#print('went here')
				j+=1
				x2 = x - 50
				y2 = y - 50
				h2 = y + h + 50
				w2 = x + w + 50
				new_img=image[y2:h2,x2:w2]
				currentDT = datetime.datetime.now()
				timetime = currentDT.strftime("%H:%M:%S")
				datedate = currentDT.strftime("%Y:%m:%d")
				os.chdir(savepath)
				#time.sleep(0.250)
				cv2.imwrite(datedate + "_" + timetime+'_'+str(j)+'.jpg', new_img)
				os.chdir('..')
				os.chdir(savepath1)
				#time.sleep(0.250)
				cv2.imwrite(datedate + "_" + timetime+'_'+str(j)+'.jpg', new_img)
				os.chdir('..')

				framescan1 = datedate + "_" + timetime+'_'+str(j)+'.jpg'
				#time.sleep(0.050)
				input_q.put(framescan1)
				#time.sleep(0.050)
				#t = time.time()

				action1 = output_q.get()
				#time.sleep(0.050)
				print(action1)
				

				conn = psycopg2.connect(database="dateandtime", user = "gab", password = "admin1234", host = "127.0.0.1", port = "5432")

				#print ("Opened database successfully")

				cur = conn.cursor()
				cur.execute("SELECT id, fn, mdytm, zn, ps, ac from db11")

				id1 = cur.rowcount
				id2 = id1 + 1
				id2 = str(id2)
				xx = ((x + w)+(x))/2
				yy = ((y)+(y + h))/2

				print(xx)
				print(yy)

				per = ""

				if ((xx <= 210) and (yy < 720)):
					dd = 1
					per = 'jom'
					print ("Zone",dd)
				elif ((xx > 210) and (xx <= 450) and (yy < 720)):
					dd = 2
					per = 'mnill'
					print ("Zone",dd)
				elif ((xx > 815) and (xx <= 1100) and (yy < 240)):
					dd = 5
					per = 'cdz'
					print ("Zone",dd)
				else:
					dd = 0
					print ("Zone",dd)


				if (dd != 0):
					fn = str(datedate) + "_" + str(timetime) + "_" + str(j)
					#cur.execute("INSERT INTO DATE_TIME (ID,MDY,DY,TM) VALUES (1,aa,bb,cc)")
					cur.execute("INSERT INTO db11 (ID,FN,MDYTM,ZN,PS,AC) VALUES (%s,%s,current_timestamp,%s,%s,%s)", (id2,fn,dd,per,str(action1)))
 
					cur.execute("SELECT mdytm from db11 WHERE id = %s",[id2])
					mtime = cur.fetchone()
					#print(mtime)
					conn.commit()
					#print ("Records created successfully")
					conn.close()

					payload1 = '{"action" : "' +str(action1)+ '", "person" : ' + '"' + per + '"' + ', "motion" : "true", "timestamp" : "' +str(mtime[0])+'"}'
					#print(payload1)
					payload = payload1
					zoneee='ucl/zone/'+str(dd)
					ret= client1.publish(zoneee,payload)				   #publish

					print("%s \n", payload)

	# show the output image
	#cv2.imshow("Image", image)
	#cv2.waitKey(1) 
	#time.sleep(0.9)


	# load our input image and grab its spatial dimensions
	image = cv2.imread(args["imageimage"])
	(H, W) = image.shape[:2]

	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# show timing information on YOLO
	#print("[INFO] YOLO took {:.6f} seconds".format(end - start))

	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

	# ensure at least one detection exists
	j=0
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			classs = LABELS[classIDs[i]]
			#print(classs)
			if (classs == 'person'):
				# draw a bounding box rectangle and label on the image
				#color = [int(c) for c in COLORS[classIDs[i]]]
				#cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
				#text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
				#cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
				#	0.5, color, 2)
				#print('went here')
				j+=1
				x2 = x - 50
				y2 = y - 50
				h2 = y + h + 50
				w2 = x + w + 50
				new_img=image[y2:h2,x2:w2]
				currentDT = datetime.datetime.now()
				timetime = currentDT.strftime("%H:%M:%S")
				datedate = currentDT.strftime("%Y:%m:%d")
				os.chdir(savepath)
				#time.sleep(0.250)
				cv2.imwrite(datedate + "_" + timetime+'_'+str(j)+'.jpg', new_img)
				os.chdir('..')
				os.chdir(savepath1)
				#time.sleep(0.250)
				cv2.imwrite(datedate + "_" + timetime+'_'+str(j)+'.jpg', new_img)
				os.chdir('..')

				framescan1 = datedate + "_" + timetime+'_'+str(j)+'.jpg'
				#time.sleep(0.050)
				input_q.put(framescan1)
				#time.sleep(0.050)
				#t = time.time()

				action1 = output_q.get()
				#time.sleep(0.050)
				print(action1)
				#os.chdir(savepath)
				#try: 
				#	os.remove(framescan1)
				#except: pass
				#os.chdir('..')

				conn = psycopg2.connect(database="dateandtime", user = "gab", password = "admin1234", host = "127.0.0.1", port = "5432")

				#print ("Opened database successfully")

				cur = conn.cursor()
				cur.execute("SELECT id, fn, mdytm, zn, ps, ac from db11")

				id1 = cur.rowcount
				id2 = id1 + 1
				id2 = str(id2)
				xx = ((x + w)+(x))/2
				yy = ((y)+(y + h))/2

				print(xx)
				print(yy)
				per = ""

				if ((xx <= 380) and (yy <= 720)):
					dd = 3
					per = 'ntig'
					print ("Zone",dd)
				elif ((xx > 380) and (xx <= 580) and (yy <= 720)):
					dd = 4
					per = 'mty'
					print ("Zone",dd)
				elif ((xx > 975) and (yy <= 720)):
					dd = 2
					per = 'mnill'
					print ("Zone",dd)
				else:
					dd = 0
					print ("Zone",dd)

				if (dd != 0):
					fn = str(datedate) + "_" + str(timetime) + "_" + str(j)
					#cur.execute("INSERT INTO DATE_TIME (ID,MDY,DY,TM) VALUES (1,aa,bb,cc)")
					cur.execute("INSERT INTO db11 (ID,FN,MDYTM,ZN,PS,AC) VALUES (%s,%s,current_timestamp,%s,%s,%s)", (id2,fn,dd,per,str(action1)))
 
					cur.execute("SELECT mdytm from db11 WHERE id = %s",[id2])
					mtime = cur.fetchone()
					#print(mtime)
					conn.commit()
					#print ("Records created successfully")
					conn.close()

					payload1 = '{"action" : "' +str(action1)+'", "person" : ' + '"' + per + '"' + ', "motion" : "true", "timestamp" : "' +str(mtime[0])+'"}'
					#print(payload1)
					payload = payload1
					zoneee='ucl/zone/'+str(dd)
					ret= client1.publish(zoneee,payload)				   #publish

					#print(payload)
					print("%s \n", payload)

	# show the output image
	#cv2.imshow("Image", image)
	#cv2.waitKey(1) 
	#time.sleep(0.9)
	cntdwn = cntdwn + 1
	print(cntdwn)
