# ------------------------------------------------------------------------------------------------------
# Main Program to run the codes according to the pipeline
# -------------------------------------------------------------------------------------------------------

# Importing neccessary packages and libraries
from glob import glob
import fnmatch
import cv2
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import keras
from imblearn.under_sampling import RandomUnderSampler
from keras.models import load_model
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation
from keras.optimizers import Adadelta 
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score 
from keras.utils import to_categorical


# Importing user-defined functions
from load_dataset import load_images
from split_dataset import split_data
from train_model import train_model
from evaluate_model import 
from utils.plot_acc_loss import make_plots

# Function call to load images and store them as npy files
load_images()

# Function call to split data into train and test
split_data()

# Function call to train model 
train_model()

# Function call to evaluate model
evaluate_model()

# Function call to plot accuracy and loss
make_plots()