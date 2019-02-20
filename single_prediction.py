# Importing neccessary packages and libraries
import cv2 
import numpy as np 
import argparse 
from keras.models import load_model

# Function to make a signal prediction
def make_single_prediction(imagePath):
    # Defining height and width of image expected by model
    height = 50
    width = 50

    # Loading Image
    img = cv2.imread(imagePath)
    print("Image Shape - {}".format(img.shape))
    
    # Resizing Image
    img = (cv2.resize(img, (width,height), interpolation=cv2.INTER_CUBIC))
    
    # Expanding dimensions to be a 4-rank tensor needed by keras function 
    img = np.expand_dims(img, axis=0)
    print("[INFO]Loading model...")
    model = load_model('idc_model.h5')

    # Making prediction
    single_pred = np.argmax(model.predict(img), axis = 1)

    if single_pred[0] == 0:
        print("[INFO]Image is IDC Negative.")
    else:
        print("[INFO]Image is IDC Positive.")