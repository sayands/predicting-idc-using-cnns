# Importing libraries and packages
import numpy as np 
from sklearn.model_selection import train_test_split


# Function for loading and splitting into train-test
def split_data():
    # Loading data from the datapath
    path = './data/images/'
    
    X = np.load(path + 'Images.npy')
    Y = np.load(path + 'Labels.npy')

    print("[INFO]Image Data Shape - {}".format(X.shape))
    print("[INFO]No.of Labels - {}".format(Y.shape[0]))

    # Changing datatype for further processing and dividing
    # by 255 to keep pixel values in the range [0..1]
    X = X.astype(np.float32)
    X /= 255.

    # Split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.15)   
    # Saving the split data as npy files for training
    np.save(path + 'X_train.npy', X_train)
    np.save(path + 'X_test.npy', X_test)
    np.save(path + 'Y_train.npy', y_train)
    np.save(path + 'Y_test.npy', y_test)
