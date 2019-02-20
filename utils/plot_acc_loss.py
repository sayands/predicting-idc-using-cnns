# Saving dictionary as a pickle file
import pickle
import matplotlib.pyplot as plt


history_file = open("../data/history.pckl",'rb')
training = pickle.load(history_file)

# Plot Training Accuracy vs Validation Accuracy Curve
plt.plot(training['acc'])
plt.plot(training['val_acc'])
plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('No.of Epochs')
plt.legend(['train', 'validation'], loc='lower right')
plt.savefig('Accuracy_Epochs.eps', type="eps", dpi = 1200, bbox_inches='tight', 
               transparent=True,
               pad_inches=0)
plt.show()

# Plot Training Loss vs Validation Loss
plt.plot(training['loss'])
plt.plot(training['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('No.of Epochs')
plt.legend(['train', 'validation'], loc='lower right')
plt.savefig('Loss_Epochs.eps', type="eps", dpi = 1200, bbox_inches='tight', 
               transparent=True,
               pad_inches=0)
plt.show()