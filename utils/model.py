# Impoting neccessary packages and libraries
import keras
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation
from keras.optimizers import Adadelta 
from keras.callbacks import ReduceLROnPlateau

# Function to define model architecture
def init_model(num_classes, learning_rate):
    '''
    Arguments : 
    num_classes : No.of classes in the dataset for classification
    learning_rate : learning rate of the model to be used

    Tasks : 
    1. Define the model architecture
    2. Define the callbacks to reduce learning rate on plateau
       in validation accuracy

    Returns : model and callback named 'annealer'
    '''
    # Defining Model
    model = Sequential() # Initialise model 
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(50, 50, 3),strides=2)) # 24 * 24 * 32
    model.add(Conv2D(64, (3, 3), activation='relu')) # 22 * 22 * 64
    model.add(MaxPool2D(pool_size=(2, 2))) # 11 * 11 * 64
    model.add(Dropout(0.25)) # 11 * 64 * 64
    model.add(Flatten()) # 7744
    model.add(Dense(128, activation='relu')) # 128
    model.add(Dropout(0.5)) # 128
    model.add(Dense(num_classes, activation='softmax')) # 2

    # Defining Optimizer
    optimizer = Adadelta(lr = learning_rate)

    # Compiling model with optimizer, loss and metrics
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ["accuracy"])
    
    # Callback to Reduce Learning Rate when Validation Accuracy
    # saturates for certain number of epochs
    annealer = ReduceLROnPlateau(monitor = 'val_acc', patience = 3, verbose = 1, factor = 0.5, min_lr = 0.00001)
    
    # return the compiled model and the respected callback
    return model, annealer