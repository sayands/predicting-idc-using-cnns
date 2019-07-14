# Importing libraries and packages
import numpy as np 
from sklearn.model_selection import train_test_split


# Function for loading and splitting into train-test
def split_data(test_size):
    '''
    Task : Loads the .npy files from the 'data/NPY-Files' directory
    of the root folder and makes the train-test split to 
    save the split data back in the same directory
    '''
    # Loading data from the datapath
    path = './data/NPY-Files/'
    
    X = np.load(path + 'Images.npy')
    Y = np.load(path + 'Labels.npy')

    print("[INFO]Image Data Shape - {}".format(X.shape))
    print("[INFO]No.of Labels - {}".format(Y.shape[0]))

    # Changing datatype for further processing and dividing
    # by 255 to keep pixel values in the range [0..1]
    X = X.astype(np.float32)
    X /= 255.

    # Split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = test_size)   
    # Saving the split data as npy files for training
    np.save(path + 'X_train.npy', X_train)
    np.save(path + 'X_test.npy', X_test)
    np.save(path + 'Y_train.npy', y_train)
    np.save(path + 'Y_test.npy', y_test)
