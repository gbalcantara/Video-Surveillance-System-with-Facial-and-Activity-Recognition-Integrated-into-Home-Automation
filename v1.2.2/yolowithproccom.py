# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
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
import paho.mqtt.client as paho
import numpy as np
from queue import Queue
from threading import Thread
from data import DataSet
from processor import process_image
from keras.models import load_model

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="cframe/cframe.jpg",
	help="path to input image")
ap.add_argument("-ii", "--imageimage", default="cframe2/cframe.jpg",
	help="path to input image")
ap.add_argument("-y", "--yolo", default="yolo-coco",
	help="base path to YOLO directory")
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
	print("data published \n")
	pass

def detect_objects(data, model, image):
	action=None
	for _ in range(1):
		images = glob.glob(os.path.join('action','*.jpg'))
		#print('-'*80)
		# Get a random row.
		sample = 0
		#print(sample)
		#print(len(images))
		image = images[sample]
		print(image)
		imagepathh = image.partition('/')
		savepath = 'action'
		os.chdir(savepath)
		#print(imagepathh)
		#imageface=cv2.imread(imagepathh[2])
		# Turn the image into an array.
		if os.path.getsize(imagepathh[2]):
		# Execute!
			print('image found')
		else:
			os.remove(imagepathh[2])
			image = images[2]
		os.chdir('..')
		
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
		sit = label_predictions[0]
		print(sit)
		stand = label_predictions[1]
		print(stand)
		#stand = label_predictions[2] 
		#print(stand)
		#for m, label in enumerate(data.classes):
		#	label_predictions[label] = predictions[0][m]

		#sorted_lps = sorted(label_predictions.items(), key=operator.itemgetter(1), reverse=True)
		
		#for n, class_prediction in enumerate(sorted_lps):
			# Just get the top five.
			#if i > 0:
			   # break

			#print(image)
			#print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
			#action = class_prediction[0]
		if (stand > sit):
			#action = stand
			action = "standing"
		else:
			action = "sitting"
			
		print("aciton=",action)
		#framescan1 = timetime+'_'+str(i)+'.jpg'
		#os.chdir(savepath)
		#try: 
			#os.remove(framescan1)
		#except: pass
		#os.chdir('..')

	return action


def worker(input_q, output_q):
	data = DataSet()
	model = load_model('2motion.hdf5')

	#fps = FPS().start()
	while True:
		#fps.update()
		frame12 = input_q.get()
		output_q.put(detect_objects(data, model, frame12))

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

while(1):

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
	print("[INFO] YOLO took {:.6f} seconds".format(end - start))

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
				os.chdir(savepath)
				#time.sleep(0.250)
				cv2.imwrite(timetime+'_'+str(j)+'.jpg', new_img)
				os.chdir('..')
				os.chdir(savepath1)
				#time.sleep(0.250)
				cv2.imwrite(timetime+'_'+str(j)+'.jpg', new_img)
				os.chdir('..')

				framescan1 = timetime+'_'+str(j)+'.jpg'
				#time.sleep(0.050)
				input_q.put(framescan1)
				#time.sleep(0.050)
				#t = time.time()

				action1 = output_q.get()
				#time.sleep(0.050)
				print("aciton1=",action1)
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
				xx = ((x2 + w2)+(x2))/2
				yy = ((y2)+(y2 + h2))/2
				if (xx in range(0, 240)) and (yy in range(0, 720)):
						dd = 1
						print ("Zone",dd)
				elif (xx in range(241, 500)) and (yy in range(0, 720)):
						dd = 2
						print ("Zone",dd)
				elif (xx in range(501, 1280)) and (yy in range(0, 260)):
						dd = 3
						print ("Zone",dd)
				else:
						dd = 4
						print ("Zone",dd)

				fn = str(timetime) + "_" + str(j)
				#cur.execute("INSERT INTO DATE_TIME (ID,MDY,DY,TM) VALUES (1,aa,bb,cc)")
				cur.execute("INSERT INTO db11 (ID,FN,MDYTM,ZN,AC) VALUES (%s,%s,current_timestamp,%s,%s)", (id2,fn,dd,str(action1)))
 
				cur.execute("SELECT mdytm from db11 WHERE id = %s",[id2])
				mtime = cur.fetchone()
				#print(mtime)
				conn.commit()
				#print ("Records created successfully")
				conn.close()

				payload1 = '{"action" : "' +str(action1)+'", "person" : "None", "motion" : "true", "timestamp" : "' +str(mtime[0])+'"}'
				#print(payload1)
				payload = payload1
				zoneee='ucl/zone/'+str(dd)
				ret= client1.publish(zoneee,payload)				   #publish

				print(payload)

	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(1) 
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
	print("[INFO] YOLO took {:.6f} seconds".format(end - start))

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
				os.chdir(savepath)
				#time.sleep(0.250)
				cv2.imwrite(timetime+'_'+str(j)+'.jpg', new_img)
				os.chdir('..')
				os.chdir(savepath1)
				#time.sleep(0.250)
				cv2.imwrite(timetime+'_'+str(j)+'.jpg', new_img)
				os.chdir('..')

				framescan1 = timetime+'_'+str(j)+'.jpg'
				#time.sleep(0.050)
				input_q.put(framescan1)
				#time.sleep(0.050)
				#t = time.time()

				action1 = output_q.get()
				#time.sleep(0.050)
				print("aciton1=",action1)
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
				xx = ((x + w)+(x))/2
				yy = ((y)+(y + h))/2
				if (xx in range(0, 400)) and (yy in range(0, 720)):
					dd = 3
					print ("Zone",dd)
				elif (xx in range(401, 980)) and (yy in range(0, 414)):
					dd = 4
					print ("Zone",dd)
				elif (xx in range(401, 1000)) and (yy in range(415, 720)):
					dd = 2
					print ("Zone",dd)
				else:
					dd = 1
					print ("Zone",dd)

				fn = str(timetime) + "_" + str(j)
				#cur.execute("INSERT INTO DATE_TIME (ID,MDY,DY,TM) VALUES (1,aa,bb,cc)")
				cur.execute("INSERT INTO db11 (ID,FN,MDYTM,ZN,AC) VALUES (%s,%s,current_timestamp,%s,%s)", (id2,fn,dd,str(action1)))
 
				cur.execute("SELECT mdytm from db11 WHERE id = %s",[id2])
				mtime = cur.fetchone()
				#print(mtime)
				conn.commit()
				#print ("Records created successfully")
				conn.close()

				payload1 = '{"action" : "' +str(action1)+'", "person" : "None", "motion" : "true", "timestamp" : "' +str(mtime[0])+'"}'
				#print(payload1)
				payload = payload1
				zoneee='ucl/zone/'+str(dd)
				ret= client1.publish(zoneee,payload)				   #publish

				print(payload)

	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(1) 
	#time.sleep(0.9)
