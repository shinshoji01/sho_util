# import torch
# import torch.nn as nn
# import torch.nn.functional as F
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

# def either_inclusion(lst, reference_list):
#     reference_set = set(reference_list)
#     bool_list = [item in reference_set for item in lst]
#     return np.array(bool_list)
def either_inclusion(lst, reference_list):
    if type(reference_list)==str:
        reference_list = [reference_list]
    reference_set = set(reference_list)
    bool_arr = np.fromiter((item in reference_set for item in lst), dtype=bool)
    return np.array(bool_arr)

def get_bool_base_on_conditions(df, params, multiple_column=False):
    if len(params)>0:
        bool_str = ""
        for hue in params.keys():
            if multiple_column:
                bool_str += f'(either_inclusion(df[{hue}], params[{hue}]))*'
            else:
                bool_str += f'(either_inclusion(df["{hue}"], params["{hue}"]))*'
        bool_list = eval(bool_str[:-1])
    else: 
        bool_list = np.array([True]*len(df))
    return bool_list

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

def key_inclusion(list, key, basename=False, mode="bool"):
    bool_list = []
    for l in list:
        if basename:
            l = os.path.basename(l)
        bool_list.append(key in l)
        
    bool_list = np.array(bool_list)
    if mode=="bool":
        output = bool_list
    elif mode=="list":
        output = np.array(list)[bool_list]
    return output

def key_exclusion(list, key, basename=False, mode="bool"):
    bool_list = []
    for l in list:
        if basename:
            l = os.path.basename(l)
        bool_list.append(not(key in l))

    bool_list = np.array(bool_list)
    if mode=="bool":
        output = bool_list
    elif mode=="list":
        output = np.array(list)[bool_list]
    return output

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

def get_outliers(values):
    q1, q3 = np.quantile(values, [0.25,0.75])
    iqr = q3-q1
    return (q1-1.5*iqr<=values)*(q3+1.5*iqr>=values)

def compute_mean_nooutliers(values):
    bool_list = get_outliers(values)
    mean = values[bool_list].mean()
    return mean

### Latex Text Generation from Pandas DataFrame ###

def get_latex_from_pandas(meandf, columns, indices, title="Title", label="Label", itvldf=None, multicolumns={}, multiindices={}, midrules=[], hdashedlines=[], pm_list=[], baseround_list=[], comparison=[], highestmin=[], twocolumn=False, tableratio=1.0, colorize=False, bestdf=None, remove_indices=False, captionbelow=False):
    """
    To plot spectrogram or mel-spectrogram with a simple command.

    ------------
    Parameters
    ------------

    meandf : pandas.DataFrame
        The main dataframe to be represented in latex
        
    columns : dict
        The dictionary contains the column name in key and the latex representation in value.
        
    indices : dict
        The dictionary contains the index name in key and the latex representation in value.
        
    title : str
        The title of the table
        
    label : str
        The label of the table
        
    itvldf : pandas.DataFrame or None, default=None
        The interval of each value
    
    multicolumns : dict, default={}
        Represent the multi columns in latex. The latex representation in key and the span in value.
        
    multiindices : dict, default={}
        Represent the multi indices in latex. The latex representation in key and the span in value.
        
    midrules : list, default=[]
        The place to put midrules
        
    hdashedlines : list, default=[]
        The place to put horizontal dashed lines
        
    pm_list : list, default=[]
        The column index where you want to put the interval
        
    baseround_list : list, length=len(columns), default=[]
        The list of the rounding number for each column
        
    comparison : list, default=[]
        The list of indices that you want to compare. The best values are highlighed by bold.
        
    highestmin : list, default=[]
        The column index where you want to make it the lower the better.
        
    meandf : pandas.DataFrame, shape=(frequency-length, time-length)
        spectrogram or mel-spectrogram

    fig : tuple or matplotlib.figure.Figure, default=(8, 8)
        When it is tuple, the function creates new plt.figure and it indicates the size of fig.
        
    ------------
    Examples
    ------------

    Example 1:
    
    title = "title 1"
    label = "label_name"
    indices = {
        "index key 1 in dataframe": "Name in Latex",
        "index key 2 in dataframe": "Name in Latex",
        "index key 3 in dataframe": "Name in Latex",
        "index key 4 in dataframe": "Name in Latex",
        "index key 5 in dataframe": "Name in Latex",
    }
    columns = {
        "column key 1 in dataframe": "Name in Latex",
        "column key 2 in dataframe": "Name in Latex",
        "column key 3 in dataframe": "Name in Latex",
        "column key 4 in dataframe": "Name in Latex",
        "column key 5 in dataframe": "Name in Latex",
    }
    multicolumns = {
        "Hierarchical ED": 3, # this column includes columns 1, 2, and 3
        "Speech Quality": 2, # this column includes columns 4, 5
    }
    baseround_list = [2, 3, 3, 3, 3] # This represents the rounding number of each column
    highestmin = [4,6] # The lower the better in 4th and 6th columns
    midrules = [4] # [] # Put the midrules after 4th index
    hdashedlines = [6] # [] # Put horizontal dashed line after 6th index
    comparison = [[1, 2, 3], [4, 5], [6, 7]] # Indices in each list are compared with each other
    pm_list = [3] # Put the interval information in the column 3.
    
    multiindices = {}
    twocolumn = False 
    remove_indices = False
    ratio = 0.7
    
    bestdf = get_latex_from_pandas(meandf, columns, indices, title, label, itvldf, multicolumns, multiindices, midrules, hdashedlines, pm_list, baseround_list, comparison, highestmin, twocolumn, ratio, remove_indices)

    ------------
        
    
    """
    if len(baseround_list)==0:
        baseround_list = [3]*len(columns)
    if colorize:
        maxval = max(df.values.max(), (-df.values).max())

    indexnum = 1 + int(len(multiindices))
    if type(bestdf)==type(None):
        flip = np.array([(-1)**int(idx in highestmin) for idx in range(len(columns))]).reshape(1, -1)
        bestdf = meandf.copy()
        valuedf = meandf.copy()
        valuedf = valuedf.applymap(lambda x: np.nan if isinstance(x, str) else x)*flip
        bestdf[:] = 0.0
        if len(comparison)>0:
            if type(comparison[0])==int:
                bestdf.iloc[comparison] = (valuedf.iloc[comparison]==np.nanmax(valuedf.iloc[comparison].values, axis=0, keepdims=True).astype(float)).astype(float)
            else:
                for comp in comparison:
                    bestdf.iloc[comp] = (valuedf.iloc[comp]==np.nanmax(valuedf.iloc[comp].values, axis=0, keepdims=True).astype(float)).astype(float)

    if twocolumn:
        print("\\begin{table*}[!h]")
    else:
        print("\\begin{table}[!h]")
    print("\caption{" + title + "}")
    print("\label{table:" + label + "}")
    print("\\centering")
    print(f"\scalebox{{{tableratio}}}{{")
    cl_def = ""
    if not(remove_indices):
        cl_def += "l"*(indexnum)
    cl_def += "c"*len(columns)

    text = "\\begin{tabular}{" + cl_def + "}"
    print(text)
    print("\\toprule")
    if len(multicolumns)>0:
        a = " & ".join([""]*(indexnum) + [f"\multicolumn{{{multicolumns[key]}}}{{c}}{{{key}}}" for key in multicolumns]) + "\\\\"
        print(a[2*int(remove_indices):])
        linetxt = ""
        if remove_indices:
            start = indexnum
        else:
            start = indexnum+1
        for key in multicolumns:
            end = start+multicolumns[key]-1
            linetxt += f"\\cmidrule(lr){{{start}-{end}}}"
            start = end+1
        print(linetxt)
    a = f" & ".join([""]*(indexnum+1) + list(columns.values()))[2:] + "\\\\"
    print(a[3*int(remove_indices):])
    print(f"\midrule")
    for i in range(len(indices)):
        key = list(indices.keys())[i]
        keyname = indices[key]
        if remove_indices:
            start = f""
        else:
            start = f"{keyname} &"
        vlist = []
        for k in range(len(columns)):
            base_round = baseround_list[k]
            valuekey = list(columns.keys())[k]
            v = meandf.loc[key, valuekey]
            if str(v)=="nan":
                vstr = "-"
            else:
                if base_round==0:
                    v = int(v)
                    val = f"{v}"
                else:
                    if v==0:
                        val = "0."+"0"*base_round
                    else:
                        if type(v)==str:
                            val = v
                        else:
                            v = np.round(v,base_round)
                            val = f"{v}".ljust(int(np.log10(v))+base_round+2, '0')
                if colorize and v!=0:
                    color = "blue" if v>0 else "red"
                    # val = f"\\cellcolor{{{color}!{abs(int(100*float(v)/maxval))}}}{{{val}}}"
                    if abs(v)<50:
                        color = f"{color}!60"
                    val = f"\\textcolor{{{color}}}{{{val}}}"
                vstr = str(val)
                if bestdf.loc[key, valuekey]==1.0:
                    vstr = f"\\textbf{{{vstr}}}"
            if k in pm_list:
                ivl = itvldf.loc[key, valuekey]
                ivl = f"{np.round(ivl,base_round)}".ljust(base_round+2, '0')
                vstr = vstr + f"{{\\tiny $\\pm$ {{{ivl}}} }}"
            vlist += [vstr]
        print(start + " & ".join(vlist) + "\\\\")
        if i+1 in midrules:
            print("\midrule")
        if i+1 in hdashedlines:
            print("\\addlinespace[0.1em]\\hdashline\\addlinespace[0.3em]")
    print("\\bottomrule")
    print("\end{tabular}")
    print("}")
    if captionbelow:
        print(f"\caption*{{{captionbelow}}}")
    if twocolumn:
        print("\end{table*}")
    else:
        print("\end{table}")
    return bestdf


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