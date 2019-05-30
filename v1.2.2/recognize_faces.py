# USAGE
# python3 recognize_faces_image1.py --encodings encodings.pickle --image examples 

# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import psycopg2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
	help="path to input image folder")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

imagePaths = list(paths.list_images(args["image"]))
#print (imagePaths)

for (i, imagePath) in enumerate(imagePaths):
	# load the input image and convert it from BGR to RGB
	print (imagePath)
	image = cv2.imread(imagePath)
	#if image == None: 
    	#	raise Exception("could not load image !")
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes corresponding
	# to each face in the input image, then compute the facial embeddings
	# for each face
	print("[INFO] recognizing faces...")
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
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
		print(name)
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
