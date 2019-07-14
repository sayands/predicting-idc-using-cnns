# Importing neccessary packages and libraries
import numpy as np 
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

def evaluate_model():
    '''
    Tasks : 
    1. Load the already trained model and test data from root folder
    2. Run predictions on test data and compare with the ground truths
    3. Calculate and display - 
            a. Confusion Matrix 
            b. Classification Report - Precision, Recall and F1-Score
            c. Balanced Accuracy Score
    '''

    path = './data/NPY-Files/'

    X_test = np.load(path + 'X_test.npy')
    y_test = np.load(path + 'Y_test.npy')
    
    print("[INFO]No.of Test Images - {}".format(X_test.shape[0]))
    print("[INFO]Loading model...")
    model = load_model('idc_model.h5')

    y_pred_one_hot = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred_one_hot, axis = 1)

    y_true_labels = y_test

    cm = confusion_matrix(y_true=y_true_labels, y_pred=y_pred_labels)
    print("[INFO]Confusion Matrix - \n {}".format(cm))

    print("[INFO]Classification Report - {}".format(classification_report(y_true_labels, y_pred_labels)))

    print("[INFO]Balanced Accuracy Score - {}".format(balanced_accuracy_score(y_true_labels, y_pred_labels)))
