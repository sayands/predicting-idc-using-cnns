# Importing neccessary packages and libraries
import pickle
import matplotlib.pyplot as plt


# Function to plot accuracy and loss
def make_plots():

    # Saving dictionary as a pickle file
    history_file = open("../model/Pickle-Files/history.pckl",'rb')
    training = pickle.load(history_file)

    # Plot Training Accuracy vs Validation Accuracy Curve
    plt.plot(training['acc'])
    plt.plot(training['val_acc'])
    plt.title('Model accuracy with Best Optimizer(Adadelta)')
    plt.ylabel('Accuracy')
    plt.xlabel('No.of Epochs')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.grid('on')
    plt.savefig('../model/Accuracy_Epochs_Best_Opt_Grid.eps', type="eps", dpi = 1200, bbox_inches='tight', 
               transparent=True,
               pad_inches=0)
    plt.show()

    # Plot Training Loss vs Validation Loss
    plt.plot(training['loss'])
    plt.plot(training['val_loss'])
    plt.title('Model Loss with Best Optimizer(Adadelta)')
    plt.ylabel('Loss')
    plt.xlabel('No.of Epochs')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.grid('on')
    plt.savefig('../model/Loss_Epochs_Best_Opt_Grid.eps', type="eps", dpi = 1200, bbox_inches='tight', 
               transparent=True,
               pad_inches=0)
    plt.show()

make_plots()