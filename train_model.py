import numpy as np
import keras
from architecture import init_model
from keras.utils import to_categorical
import pickle
from keras_radam import RAdam

# Define our custom loss function
def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25

    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

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
    
    X_train = np.load(path + 'X_train.npy')
    y_train = np.load(path + 'Y_train.npy')
    print('Loaded Train Data')
    print(X_train.shape, y_train.shape)

    # Loading Test Data For Validation
    X_test = np.load(path + 'X_test.npy')
    y_test = np.load(path + 'Y_test.npy')
    print('Loaded Test Data')
    print(X_test.shape, y_test.shape)

    # Converting to Categorical Data
    y_train = to_categorical(y_train, num_classes = num_classes)
    y_test = to_categorical(y_test, num_classes = num_classes)

    # Creating model
    classifier = model.init_model(num_classes = num_classes)

    optimizer = RAdam(lr = LEARNING_RATE)
    # Compiling model with optimizer, loss and metrics
    classifier.compile(loss = focal_loss, optimizer = optimizer, metrics = ["accuracy"])

    callbacks = [
            keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph = True, write_images = True),
            keras.callbacks.ModelCheckpoint(os.path.join(WEIGHT_SAVE_PATH, 'weights{epoch:08d}.h5'),
                                    verbose=0, save_weights_only=True),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience = 5, verbose = 1)]

    # Starting training of model on undersampled training data
    history = classifier.fit(X_train, y_train, batch_size=batch_size,
                      epochs=epochs, verbose=1, validation_data = (X_test, y_test), callbacks = callbacks, initial_epoch = initial_epochs)