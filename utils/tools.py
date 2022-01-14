import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
    
def plot_loss_accuracy(history):
    '''Plot training & validation loss & accuracy values, giving an argument
       'history' of type 'tensorflow.python.keras.callbacks.History'. '''
    
    plt.figure(figsize=(15,5))
    ax1 = plt.subplot(1,2,1)
    if history.history.get('accuracy'):
        ax1.plot(np.array(history.epoch)+1, history.history['accuracy'], 'o-',label='Train')
    if history.history.get('val_accuracy'):
        ax1.plot(np.array(history.epoch)+1, history.history['val_accuracy'], 'o-', label='Test')
    ax1.set_title('Model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch') 
    ax1.grid()
    ax1.legend(loc='best')
    
    # Plot training & validation loss values
    ax2 = plt.subplot(1,2,2)
    if history.history.get('loss'):
        ax2.plot(np.array(history.epoch)+1, history.history['loss'], 'o-', label='Train')
    if history.history.get('val_loss'):
        ax2.plot(np.array(history.epoch)+1, history.history['val_loss'], 'o-',  label='Test')
    ax2.set_title('Model loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='best')
    ax2.grid()
    plt.show()

def plot_images(image_array, r, L, C):
    '''Plot the images of image_array on a grid L x C, starting at
       rank r'''
    plt.figure(figsize=(C,L))
    for i in range(L*C):
        plt.subplot(L, C, i+1)
        plt.imshow(image_array[r+i], cmap='gray')
        plt.xticks([]); plt.yticks([])


def show_cm_mnist(target, results, classes):
    # target  : the actual labels (one-hot format)
    # results : the labels computed by the trained network (one-hot format)
    # classes : list of possible label values
    predicted = np.argmax(results, axis=-1) # tableau d'entiers entre 0 et 9 
    cm = confusion_matrix(target, predicted)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(11,9))
    heatmap(df_cm, annot=True, cbar=False, fmt="3d")
    plt.xlabel('actual label')
    plt.ylabel('predicted label')
    plt.show()
    
