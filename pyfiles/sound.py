import warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
import IPython
import librosa
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn

def play_audio(data, rate):
    IPython.display.display(IPython.display.Audio(data=data,rate=rate))
    
def silence_removal(x, top_db=25, trim_window=256, trim_stride=128, only_edge=True, top_db_intermediate=50):
    nonsilence = librosa.effects.split(x, top_db=top_db, frame_length=trim_window, hop_length=trim_stride)
    x_rec = np.array([])
    x_remove_edge = x[nonsilence[0][0]:nonsilence[-1][1]]
    if only_edge:
        x_rec = x_remove_edge
    else:
        nonsilence_low = librosa.effects.split(x_remove_edge, top_db=top_db_intermediate, frame_length=trim_window, hop_length=trim_stride)
        for i in range(len(nonsilence_low)):
            x_split = x_remove_edge[nonsilence_low[i][0]:nonsilence_low[i][1]]
            x_rec = np.append(x_rec, x_split)
    return x_rec