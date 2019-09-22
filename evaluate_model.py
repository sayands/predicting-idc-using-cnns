# Importing neccessary packages and libraries
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, accuracy_score
from utils import model
from keras import backend as K
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

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
    WEIGHTS_PATH = './model/weights00000017.h5' # To be changed with the best weights
    
    # Loading Test Data
    X_test = np.load(path + 'X_test.npy')
    y_test = np.load(path + 'Y_test.npy')

    print("[INFO]No.of Test Images - {}".format(X_test.shape[0]))
    print("[INFO]Loading weights...", WEIGHTS_PATH)

    classifier = model.init_model(num_classes = 2)
    classifier.load_weights(WEIGHTS_PATH)

    y_pred_one_hot = classifier.predict(X_test)
    y_pred_labels = np.argmax(y_pred_one_hot, axis = 1)

    y_true_labels = y_test

    cm = confusion_matrix(y_true=y_true_labels, y_pred=y_pred_labels)
    print("[INFO]Confusion Matrix - \n {}".format(cm))

    print("[INFO]Classification Report - \n {}".format(classification_report(y_true_labels, y_pred_labels, digits = 4)))

    print("[INFO]Balanced Accuracy Score - {}".format(balanced_accuracy_score(y_true_labels, y_pred_labels)))

    print("[INFO]Accuracy Score - {}".format(accuracy_score(y_true_labels, y_pred_labels)))

    # calculate the fpr and tpr for all thresholds of the classification
    probs = y_pred_one_hot 
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    # Plot ROC Curve for Binary Classification
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.5f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('Plot-AUCROC.png', eps = 600)