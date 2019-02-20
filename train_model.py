import numpy as np
import keras
from utils import model, random_undersampling
from keras.utils import to_categorical
import pickle

def train_model():
    # Hyperparameters
    batch_size = 32
    num_classes = 2
    epochs = 50

    # DataPath
    path = './data/NPY-Files/'

    # Random Undersampling of Training Data
    X_trainRusReshaped, Y_trainRusHot = random_undersampling.undersample_data()
    
    # Loading Test Data For Validation
    X_test = np.load(path + 'X_test.npy')
    y_test = np.load(path + 'Y_test.npy')
    # Creating model
    classifier, annealer = model.init_model(num_classes)

    # Starting training of model on undersampled training data
    history = classifier.fit(X_trainRusReshaped,Y_trainRusHot,batch_size=batch_size,
                      epochs=epochs, verbose=1, validation_data = (X_test, y_test), callbacks = [annealer])

    # Saving dictionary as a pickle file
    f = open('./data/history.pckl', 'wb')
    pickle.dump(history.history, f)
    f.close()


    # Saving Model for further evaluation 
    classifier.save('idc_model.h5')