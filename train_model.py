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

    # Random Undersampling of Training Data
    X_trainRusReshaped, Y_trainRusHot = random_undersampling.undersample_data()

    # Creating model
    classifier = model.init_model(num_classes)

    # Starting training of model on undersampled training data
    history = classifier.fit(X_trainRusReshaped,Y_trainRusHot,batch_size=batch_size,
                      epochs=epochs, verbose=1)

    # Saving dictionary as a pickle file
    f = open('./data/history.pckl', 'wb')
    pickle.dump(history.history, f)
    f.close()


    # Saving Model for further evaluation 
    classifier.save('idc_model.h5')