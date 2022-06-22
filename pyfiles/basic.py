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
    
def plot_spectrogram(M, fig=(8,8), subplot=(1,1,1), t=None, freq=None, title="", xlabel="", ylabel="", alpha=1, title_font=15):
    """
    To plot spectrogram or mel-spectrogram with a simple command.

    ------------
    Parameters
    ------------

    M : ndarray, shape=(frequency-length, time-length)
        spectrogram or mel-spectrogram

    fig : tuple or matplotlib.figure.Figure, default=(8, 8)
        When it is tuple, the function creates new plt.figure and it indicates the size of fig.
    
    ------------
    Returns
    ------------

    ax : matplotlib.axes._subplots.AxesSubplot
        It can be used when you want to add some features to the plot. For example, ax.set_xtick_labels

    ------------
    Examples
    ------------

    Example 1:
    plot_spectrogram(M)

    Example 2:
    ax = plot_spectrogram(M, fig, (2, 1, 1), title_font=20)
    
    ------------

    """
    if fig==tuple:
        fig = plt.figure(figsize=fig)
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
    if type(column)==tuple:
        column = pd.MultiIndex.from_tuples([column])
    elif type(column)==str:
        column = [column]
        
    for i, key in enumerate(list(df_dir.keys())):
        df_value = df_dir[key].reset_index(drop=True)
        df_basic = pd.DataFrame(np.array([key]*len(df_value)).reshape(-1, 1), columns=column)
        df = pd.concat([df_basic, df_value], axis=1)
        if i==0:
            df_new = df
        else:
            df_new = pd.concat([df_new, df], axis=0)
    df_new = df_new.reset_index(drop=True)
    return df_new

def either_inclusion(list, reference_list):
    bool_list = []
    for l in list:
        bool_list.append(l in reference_list)
    return np.array(bool_list)

def get_bool_base_on_conditions(df, params, multiple_column=False):
    bool_str = ""
    for hue in params.keys():
        if multiple_column:
            bool_str += f'(either_inclusion(df[{hue}], params[{hue}]))*'
        else:
            bool_str += f'(either_inclusion(df["{hue}"], params["{hue}"]))*'
    return eval(bool_str[:-1])

def get_bool_base_on_conditions_with_exclusion(data, inclusion, exclusion, multiple_column=False):
    inclusion_bool = get_bool_base_on_conditions(data, inclusion, multiple_column)
    
    whole_params = {**inclusion, **exclusion}
    exclusion_bool = get_bool_base_on_conditions(data, whole_params, multiple_column)
    
    bool_list_int = np.array(inclusion_bool, dtype=np.int) - np.array(exclusion_bool, dtype=np.int)
    return np.array(bool_list_int, dtype=bool)

def select_columns(data, included=None, excluded=None):
    """

    ------------
    Parameters
    ------------

    ------------
    Returns
    ------------

    ------------
    Examples
    ------------

    Example 1:
    included = [
        '["basics", ["model name", "emotion", "mode"]]',
        '["fi", :]'
    ]

    ------------

    """
    # Get all the columns
    if included==None:
        whole_df = data.copy()
    else:
        dfs = []
        for i in range(len(included)):
            cols = eval(f'pd.IndexSlice{included[i]}')
            dfs += [data.loc[:, cols]]
        whole_df = pd.concat(dfs, axis=1)
        whole_df = whole_df.loc[:,~whole_df.columns.duplicated()]

    # Get the columns we want to exclude
    if excluded==None:
        output = whole_df.copy()
    else:
        excluded_columns = []
        for i in range(len(excluded)):
            cols = eval(f'pd.IndexSlice{excluded[i]}')
            excluded_columns += list(data.loc[:, cols].columns)
        output = whole_df.drop(excluded_columns, axis=1)
    return output

def key_inclusion(list, key, basename=False):
    bool_list = []
    for l in list:
        if basename:
            l = os.path.basename(l)
        bool_list.append(key in l)
    return np.array(bool_list)

def key_exclusion(list, key, basename=False):
    bool_list = []
    for l in list:
        if basename:
            l = os.path.basename(l)
        bool_list.append(not(key in l))
    return np.array(bool_list)

def select_from_pathlist(path_list, included="all", excluded="None", selection_mode_included="or", selection_mode_excluded="or", basename_included=False, basename_excluded=False):
    path_list = np.array(path_list)
    if included=="all":
        new_path_list = path_list
    else:
        if type(included) != list:
            included = [included]
        for i, key in enumerate(included):
            bl = np.array(key_inclusion(path_list, key, basename_included),dtype=int)
            if i==0:
                bool_list = bl
            else:
                if selection_mode_included=="or":
                    bool_list = bool_list + bl
                elif selection_mode_included=="and":
                    bool_list = bool_list * bl
                else:
                    print(selection_mode_included)
        new_path_list = path_list[np.array(bool_list,dtype=bool)]
    path_list = new_path_list
    
    if excluded=="None":
        new_path_list = path_list
    else:
        if type(excluded) != list:
            excluded = [excluded]
        for i, key in enumerate(excluded):
            bl = np.array(key_exclusion(path_list, key, basename_excluded),dtype=int)
            if i==0:
                bool_list = bl
            else:
                if selection_mode_excluded=="or":
                    bool_list = bool_list + bl
                elif selection_mode_excluded=="and":
                    bool_list = bool_list * bl
                else:
                    print(selection_mode_excluded)
        new_path_list = path_list[np.array(bool_list,dtype=bool)]
    return new_path_list

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