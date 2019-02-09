# Impoting neccessary packages and libraries
import keras
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation
from keras.optimizers import Adadelta 

# Function to define model architecture
def init_model(num_classes):
    # Defining Model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(50, 50, 3),strides=2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Defining Optimizer
    optimizer = Adadelta(lr = 0.75)

    # Compiling model with optimizer, loss and metrics
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ["accuracy"])

    # return the compiled model
    return model