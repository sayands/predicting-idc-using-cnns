# Impoting neccessary packages and libraries
import keras
from keras.utils import to_categorical
from keras.models import Sequential, load_model, Model
from keras.layers import *
from keras.optimizers import Adadelta
from keras.callbacks import ReduceLROnPlateau
from keras import losses
from keras import backend as K

# Function to define model architecture
def init_model(num_classes):
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
    input_size = (50, 50, 3)
    input = Input(input_size)

    # Block 1
    x = Conv2D(32, 3, padding = 'same', activation='relu', kernel_initializer = 'he_normal')(input)
    x = Conv2D(32, 3, padding = 'same', activation='relu', kernel_initializer = 'he_normal')(x)
    x = MaxPool2D(2, strides= 2, name='block1_pool')(x)

    # Block 2
    x = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', activation='relu')(x)
    x = Conv2D(64, 3, padding = 'same', activation='relu', kernel_initializer = 'he_normal')(x)
    x = MaxPool2D(2, strides= 2, name='block2_pool')(x)

    # FCN Block
    x = Conv2D(64, 3, activation = 'relu', kernel_initializer = 'he_normal', padding='same', name='fcn1')(x)
    x = Conv2D(128, 3, kernel_initializer = 'he_normal', activation = 'relu', padding='same', name='fcn2')(x)

    x = Conv2D(128, 3, kernel_initializer = 'he_normal', activation = 'relu', padding='same', name='fcn3')(x)
    x = Conv2D(64, 3, kernel_initializer = 'he_normal', activation = 'relu', padding='same', name='fcn4')(x)

    x = GlobalAveragePooling2D(name='avgpool')(x)
    x = Dense(32, kernel_initializer='he_normal', activation = 'relu')(x)
    x = Dense(2, kernel_initializer='he_normal', activation = 'softmax')(x)

    model = Model(inputs = input, outputs = x)
    model.summary()
    # return the compiled model
    return model