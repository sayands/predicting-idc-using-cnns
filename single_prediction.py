# Importing neccessary packages and libraries
import cv2 
import numpy as np 
import argparse 
from keras.models import load_model

# Function to make a signal prediction
def make_single_prediction(imagePath):
    '''
    Arguments : 
    imagePath : path of image to be read for prediction by the trained model

    Tasks:
    1. Resize and normalise image to match model input shape
    2. Load model and make a prediction
    3. Display the result
    '''

    # Defining height and width of image as expected by model
    height = 50
    width = 50

    # Loading Image
    img = cv2.imread(imagePath)
    print("Image Shape - {}".format(img.shape))
    
    # Resizing Image
    img = (cv2.resize(img, (width,height), interpolation=cv2.INTER_CUBIC))
    img /= 255.0
    # Expanding dimensions to be a 4-rank tensor needed by keras function 
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    print("[INFO]Loading model...")
    model = load_model('idc_model.h5')

    # Making prediction
    single_pred = np.argmax(model.predict(img), axis = 1)

    if single_pred[0] == 0:
        print("[INFO]Image is IDC Negative.")
    else:
        print("[INFO]Image is IDC Positive.")