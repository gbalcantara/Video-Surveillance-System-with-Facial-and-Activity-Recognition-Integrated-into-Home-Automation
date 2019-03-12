"""
Classify a few images through our CNN.
"""
import numpy as np
import operator
import random
import glob
import os.path
from data import DataSet
from processor import process_image
from keras.models import load_model
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2

def main():
    """Spot-check `nb_images` images."""
    data = DataSet()
    model = load_model('inception.038-0.85.hdf5')

    # Get all our test images.
    images = glob.glob(os.path.join('images','*.jpg'))
    nb_images=len(images)
    for _ in range(nb_images):
        print('-'*80)
        # Get a random row.
        sample = random.randint(0, len(images) - 1)
        #print(sample)
        #print(len(images))
        image = images[sample]

        # Turn the image into an array.
        print(image)
        image_arr = process_image(image, (299, 299, 3))
        image_arr = np.expand_dims(image_arr, axis=0)

        # Predict.
        predictions = model.predict(image_arr)

        # Show how much we think it's each one.
        label_predictions = {}
        for i, label in enumerate(data.classes):
            label_predictions[label] = predictions[0][i]

        sorted_lps = sorted(label_predictions.items(), key=operator.itemgetter(1), reverse=True)
        
        for i, class_prediction in enumerate(sorted_lps):
            # Just get the top five.
            if i > 4:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
            i += 1

def main2():
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

if __name__ == '__main__':
    main()
    main2()
