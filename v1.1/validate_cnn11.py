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
import psycopg2

def main(nb_images=5):
    """Spot-check `nb_images` images."""
    data = DataSet()
    model = load_model('3motion.hdf5')

    # Get all our test images.
    images = glob.glob(os.path.join('images','*.jpg'))
    nb_images=len(images)
    for _ in range(nb_images):
        #print('-'*80)
        # Get a random row.
        sample = random.randint(0, len(images) - 1)
        #print(sample)
        #print(len(images))
        image = images[sample]

        # Turn the image into an array.
        
        image_arr = process_image(image, (299, 299, 3))
        image_arr = np.expand_dims(image_arr, axis=0)

        # Predict.
        predictions = model.predict(image_arr)

        # Show how much we think it's each one.
        label_predictions = {}
        cars = [1, 2, 3] 
        for i, label in enumerate(cars):
            label_predictions[label] = predictions[0][i]
        #print(label_predictions)
        sorted_lps = sorted(label_predictions.items(), key=operator.itemgetter(1), reverse=True)
        
        for i, class_prediction in enumerate(sorted_lps):
            # Just get the top five.
            if i > 0:
                break

            #print(image)
            fol, fn1, t1 = image.partition('/')
            #print(fol)
            #print(t1)
            #print(fn1)
            fn,do,t2 = t1.partition('.')
            #print(fn)
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
            i += 1
            conn = psycopg2.connect(database="dateandtime", user = "gab", password = "admin1234", host = "127.0.0.1", port = "5432")

            #print ("Opened database successfully")

            cur = conn.cursor()

            #cur.execute("SELECT id, fn, mdy, dy, tm, zn, ps, ac from db1")


            #cur.execute("INSERT INTO DATE_TIME (ID,MDY,DY,TM) VALUES (1,aa,bb,cc)")
            cur.execute("UPDATE db11 SET ac = %s WHERE fn = %s", (class_prediction[0],fn))

            conn.commit()
            #print ("Records created successfully")
            conn.close()

if __name__ == '__main__':
    main()
