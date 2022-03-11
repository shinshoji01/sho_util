import numpy as np
import IPython
import librosa
import noisereduce as nr

def play_audio(data, rate):
    IPython.display.display(IPython.display.Audio(data=data,rate=rate))
    
def silence_removal(x, top_db=25, trim_window=256, trim_stride=128, only_edge=True, top_db_intermediate=50, mode="trim"):
    if mode=="split":
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
    elif mode=="trim":
        x_rec, _ = librosa.effects.trim(x, top_db=top_db, frame_length=trim_window, hop_length=trim_stride)
    
    return x_rec
    
class silent_sections_removal():
    def __init__(self, fs=22050, mode="trim"):
        self.mode = mode
        self.fs = fs
        
        args = {}
        args["ravdess-speech"] = {}
        args["ravdess-song"] = {}
        args["tess"] = {}
        args["crema-d"] = {}
        args["savee"] = {}
        for dataset in args.keys():
            args[dataset]["trim_window"] = 2048
            args[dataset]["trim_stride"] = 512
        args["ravdess-speech"]["top_db"] = 25
        args["ravdess-song"]["top_db"] = 25
        args["tess"]["top_db"] = 25
        args["crema-d"]["top_db"] = 25
        args["savee"]["top_db"] = 20
        self.args = args
    
    def get(self, x, dataset="ravdess-speech", noise_removal=False):
        top_db = self.args[dataset]["top_db"]
        trim_window = self.args[dataset]["trim_window"]
        trim_stride = self.args[dataset]["trim_stride"]
        if (dataset=="crema-d") or (noise_removal):
            x = nr.reduce_noise(x, self.fs)
        x_rec = silence_removal(x, top_db, trim_window, trim_stride, mode=self.mode)
        return x_rec