import numpy as np
import keras
from utils import model, random_undersampling
from keras.utils import to_categorical
import pickle

def train_model(batch_size, num_classes, epochs, learning_rate):
    '''
    Tasks : 
    1. Load the training and test dataset from the root folder directory
    2. Randomly Undersample the training data to prevent bias due to
       class imbalance
    3. Define the Model Architecture 
    4. Train the model
    5. Save the model and history to get a visualisation of model 
       performance over the period of training
    '''

    # DataPath
    path = './data/NPY-Files/'

    # Random Undersampling of Training Data
    X_trainRusReshaped, Y_trainRusHot = random_undersampling.undersample_data()
    
    # Loading Test Data For Validation
    X_test = np.load(path + 'X_test.npy')
    y_test = np.load(path + 'Y_test.npy')
    # Creating model
    classifier, annealer = model.init_model(num_classes = num_classes, learning_rate = learning_rate)

    # Starting training of model on undersampled training data
    history = classifier.fit(X_trainRusReshaped,Y_trainRusHot,batch_size=batch_size,
                      epochs=epochs, verbose=1, validation_data = (X_test, y_test), callbacks = [annealer])

    # Saving dictionary as a pickle file
    f = open('./model/Pickle-Files/history.pckl', 'wb')
    pickle.dump(history.history, f)
    f.close()


    # Saving Model for further evaluation 
    classifier.save('./model/idc_model.h5')