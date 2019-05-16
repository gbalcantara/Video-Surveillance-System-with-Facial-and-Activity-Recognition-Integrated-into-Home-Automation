# USAGE
# python3 recognize_faces_image1.py --encodings encodings.pickle --image examples 

# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
import psycopg2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from imutils import paths

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="method2facesmix.model",
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", default="method2facesmix.pickle",
	help="path to label binarizer")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["image"]))
#print (imagePaths)

print("[INFO] loading network...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())
#print (imagePaths)
savepath1 = 'faces'
savepath2 = 'facescopy'

for (i, imagePath) in enumerate(imagePaths):
	# load the input image and convert it from BGR to RGB
	print (imagePath)
	#image = cv2.imread(imagePath)
	#os.chdir(savepath1)
	if os.path.getsize(imagePath):
	# Execute!
		print('image found')
		image = cv2.imread(imagePath)
	else:
		os.remove(imagePath)
		#os.chdir('..')
		break
	#os.chdir('..')
	output = image.copy()
	# pre-process the image for classification
	image = cv2.resize(image, (96, 96))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# load the trained convolutional neural network and the label
	# binarizer
	

	# classify the input image
	print("[INFO] classifying image...")
	proba = model.predict(image)[0]
	idx = np.argmax(proba)
	label = lb.classes_[idx]

	# we'll mark our prediction as "correct" of the input image filename
	# contains the predicted label text (obviously this makes the
	# assumption that you have named your testing image files this way)
	#filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
	#Zcorrect = "correct" if filename.rfind(label) != -1 else "incorrect"
	print(label)
	#print(image)
	fol, fn1, t1 = imagePath.partition('/')
	#print(fol)
	#print(t1)
	#print(fn1)
	fn,do,t2 = t1.partition('.')
	#print(fn)
	#print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
	i += 1
	conn = psycopg2.connect(database="dateandtime", user = "gab", password = "admin1234", host = "127.0.0.1", port = "5432")

	#print ("Opened database successfully")

	cur = conn.cursor()
	#cur.execute("SELECT id, fn, mdy, dy, tm, zn, ps, ac from db1")


	#cur.execute("INSERT INTO DATE_TIME (ID,MDY,DY,TM) VALUES (1,aa,bb,cc)")
	cur.execute("UPDATE db11 SET ps = %s WHERE fn = %s", (name,fn))
	conn.commit()
	#print ("Records created successfully")
	conn.close()

	fol, fn1, t1 = imagePath.partition('/')
	fn,do,t2 = t1.partition('.')
	os.chdir(savepath2)
	cv2.imwrite(t1, image)
	os.chdir('..')
	os.chdir(savepath1)
	#try: 
	os.remove(t1)
	#except: pass
	os.chdir('..')

	# loop over the recognized faces
	#for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image
		#cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
		#y = top - 15 if top - 15 > 15 else top + 15
		#cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			#0.75, (0, 255, 0), 2)

	# show the output image
	#cv2.imshow("Image", image)
	#cv2.waitKey(0)
