import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import itertools
import pickle
import shutil
import os
import pandas as pd
import glob

def pickle_save(data, path):
    """
    save data in a form of pickle

    ------------
    Parameters
    ------------

    data : anything
    
    path : str
        path where you save data

    ------------
    Returns
    ------------

    ------------

    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        
def pickle_load(path):
    """
    load data with a pickle form

    ------------
    Parameters
    ------------

    path : str
        path where your data is located

    ------------
    Returns
    ------------
    
    data : anything

    ------------

    """
    with open(path, mode='rb') as f:
        data = pickle.load(f)
    return data

def min_max(x, axis=None, mean0=False, get_param=False):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min+1e-8)
    if mean0 :
        result = result*2 - 1
    if get_param:
        return result, min, max
    return result

def save_gif(data_list, gif_path, title, save_dir="contempolary_images/", fig_size=(8,8), font_title=24, duration=100):
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(data_list)):
        fig = plt.figure(figsize=fig_size)
        plt.cla()
        ax = fig.add_subplot(1,1,1)
        ax.imshow(data_list[i])
        plt.title(title, fontsize=font_title)
        save_path = save_dir + f"{str(i).zfill(3)}.png"
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        plt.savefig(save_path, dpi=64, facecolor = "lightgray", bbox_inches="tight", format="png")
        plt.close()
        plt.axis('off')
    files = sorted(glob.glob(save_dir + '*.png'))
    images = list(map(lambda file: Image.open(file), files))
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
    shutil.rmtree(save_dir, ignore_errors=True)
    
def plot_spectrogram(M, fig=None, subplot=(1,1,1), t=None, freq=None, title="", xlabel="", ylabel="", alpha=1, title_font=15):
    if fig==None:
        fig = plt.figure(figsize=(5,5))
    elif type(fig)==tuple:
        fig = plt.figure(figsize=fig)
    ax = fig.add_subplot(subplot[0], subplot[1], subplot[2])
    if type(t)!=np.ndarray:
        t = range(M.shape[1])
    if type(freq)!=np.ndarray:
        freq = range(M.shape[0])
    ax.pcolormesh(t, freq, M, cmap = 'jet', alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=title_font)
    return ax

def list_substruction(a, b):
    return [item for item in a if item not in b]

def dir2table(df_dir, column):
    for i, key in enumerate(list(df_dir.keys())):
        df_value = df_dir[key].reset_index(drop=True)
        df_basic = pd.DataFrame(np.array([key]*len(df_value)).reshape(-1, 1), columns=[column])
        df = pd.concat([df_basic, df_value], axis=1)
        if i==0:
            df_new = df
        else:
            df_new = pd.concat([df_new, df], axis=0)
    df_new = df_new.reset_index(drop=True)
    return df_new

############ ----------------------------------------------------- #############
############ https://www.kaggle.com/grfiv4/plot-a-confusion-matrix #############
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
############ ----------------------------------------------------- #############