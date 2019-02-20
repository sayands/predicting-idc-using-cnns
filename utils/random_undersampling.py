# Importing neccessary packages and libraries
import numpy as np
import keras
from keras.utils import to_categorical
from imblearn.under_sampling import RandomUnderSampler

def undersample_data():

    # Defining path to load data from    
    path = "../data/NPY-Files/"
    
    # Loading training data from disc
    X_train = np.load(path + 'X_train.npy')
    y_train = np.load(path + 'Y_train.npy')

    print("[INFO]Loaded Dataset successfully.")

    # Reshaping data for sampling
    X_trainShape = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
    X_trainFlat = X_train.reshape(X_train.shape[0], X_trainShape)

    # Defining the random under sampler
    random_under_sampler = RandomUnderSampler(ratio='majority')
    X_trainRus, Y_trainRus = random_under_sampler.fit_sample(X_trainFlat, y_train)

    # One-hot-encoding undersampled train labels
    Y_trainRusHot = to_categorical(Y_trainRus, num_classes = 2)


    print("[INFO]No.of Training Images After UnderSampling - {}".format(Y_trainRusHot.shape))

    # Reshaping sampled data to be passed to train the model
    for i in range(len(X_trainRus)):
        height, width, channels = 50,50,3
        X_trainRusReshaped = X_trainRus.reshape(len(X_trainRus),height,width,channels)

    # return data
    return X_trainRusReshaped, Y_trainRusHot
