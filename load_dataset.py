# Importing neccessary packages and libraries
from glob import glob
import fnmatch
import cv2
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

# Function to process images and return corresponding tensor
def process_images(imagePatches, classZero, classOne):
    """
    Returns two arrays: 
        x is an array of resized images
        y is an array of labels
    """ 
    height = 50
    width = 50
    channels = 3
    

    # Lists to store image data and corresponding class labels
    x = [] 
    y = []
    
    # looping over all the imagePatches to process each image
    for img in imagePatches:
        # Reading in the image
        full_size_image = cv2.imread(img)

        # Resizing image
        image = (cv2.resize(full_size_image, (width,height), interpolation=cv2.INTER_CUBIC))
        
        # Appending image to the list 
        x.append(image)

        # Checking label of image to be IDC Positive/ Negative
        if img in classZero:
            y.append(0)
        elif img in classOne:
            y.append(1)
        else:
            return
    
    # returning the processed list of images and labels
    return x,y

# Function to load images from disk
def load_images():

    # Loading the path of all image Patches 
    path = './data/'
    imagePatches = glob(path + 'images/**/*.png', recursive=True)
    print("[INFO]Total No.of Images in our Dataset - {}".format(len(imagePatches)))

    # Defining the two classes of data for further loading and processing
    patternZero = '*class0.png'
    patternOne = '*class1.png'

    # Saving the file location of all images with file name class0
    classZero = fnmatch.filter(imagePatches, patternZero) 
    classOne = fnmatch.filter(imagePatches, patternOne)


    print("[INFO]Started Processing Data...")
    
    # Timer to note down pocessing time
    start = time.time()
    
    # Invoking function to process images
    X, Y = process_images(imagePatches, classZero, classOne)
    
    # Converting to a numpy array for ease in usage
    X = np.array(X)
    Y = np.array(Y)

    # Displaying information
    print("[INFO]Processed Image Data Shape - {}".format(X.shape))
    print("[INFO]Number of Labels - {}".format(Y.shape))

    # Saving data to a npy file on disc
    np.save(path + 'NPY-Files/Images.npy', X)
    np.save(path + 'NPY-Files/Labels.npy', Y)
    end = time.time()

    # Printing timeer information
    print("[INFO]Saved Files Successfully.")
    print("[INFO]Time Taken - {}s".format(end - start))
