import numpy as np
from fastdtw import fastdtw
import pyworld as pw
from scipy.spatial.distance import euclidean
from pymcd.mcd import Calculate_MCD

class get_evaluation_scores():
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
    
    def __init__(self, MCD_mode="dtw", sr=16000):
        mcd_toolbox = Calculate_MCD(MCD_mode)
        self.mt = mcd_toolbox
        self.mt.SAMPLIG_RATE = sr
        self.sr = sr
        
        
    def _get_pitch_contour(self, wav, remove_zero=True, logscale=False):
        x = wav.astype(np.float64)
        f0, t = pw.harvest(x, self.sr)
        if remove_zero:
            f0 = f0[f0>0]
        if logscale:
            f0 = np.log1p(f0)
        return f0
        
    def _get_energy_contour(self, wav, logscale=False, frame=0.005):
        frame_length = int(self.sr*frame) 
        hop_length = int(self.sr*frame)
        y = wav
        energy = np.array([
            sum(abs(y[i:i+frame_length]**2))
            for i in range(0, len(y), hop_length)
        ])
        if logscale:
            energy = np.log1p(energy)
        return energy
    
    def get(self, file1, file2, features=["mcd", "pitch", "pitch_remove0", "energy"], p_logscale=False, e_logscale=False):
        loaded_ref_wav = self.mt.load_wav(file1, sample_rate=self.sr)
        loaded_syn_wav = self.mt.load_wav(file2, sample_rate=self.sr)
        mcd, pscore, escore = 0, 0, 0
        mcd_distance, pitch_distance, energy_distance = 0, 0, 0
        if "mcd" in features:
            ref_mcep_vec = self.mt.wav2mcep_numpy(loaded_ref_wav)
            syn_mcep_vec = self.mt.wav2mcep_numpy(loaded_syn_wav)
            distance, path = fastdtw(ref_mcep_vec[:,1:], syn_mcep_vec[:,1:], dist=euclidean)
            frames_tot, min_cost_tot = self.mt.calculate_mcd_distance(ref_mcep_vec, syn_mcep_vec, path)
            mcd_distance = distance/frames_tot
            mcd = self.mt.log_spec_dB_const * min_cost_tot/frames_tot
            a = np.array(path)
            fd_mcd = np.sqrt(np.mean((a[:,0]-a[:,1])**2))
            
        if "pitch" in features or "pitch_remove0" in features:
            pitch_features = np.array(features)[["pitch" in a for a in features]]
            d_pscore = {}
            d_pitch_ppc = {}
            d_pitch_distance = {}
            d_fd_pitch = {}
            for pf in pitch_features:
                p_remove0 = "remove0" in pf
                pc_ref = self._get_pitch_contour(loaded_ref_wav, p_remove0, p_logscale).reshape(-1, 1)
                pc_syn = self._get_pitch_contour(loaded_syn_wav, p_remove0, p_logscale).reshape(-1, 1)
                distance, pc_path = fastdtw(pc_ref, pc_syn, dist=euclidean)
                frames_tot, min_cost_tot = self.mt.calculate_mcd_distance(pc_ref, pc_syn, pc_path)
                pitch_distance = distance/frames_tot
                pscore = min_cost_tot/frames_tot

                a = pc_ref[np.array(pc_path)[:,0], 0]
                b = pc_syn[np.array(pc_path)[:,1], 0]
                pitch_ppc = np.corrcoef(a, b)[0, 1]

                pc_ref = self._get_pitch_contour(loaded_ref_wav, False, p_logscale).reshape(-1, 1)
                pc_syn = self._get_pitch_contour(loaded_syn_wav, False, p_logscale).reshape(-1, 1)
                distance, pc_path = fastdtw(pc_ref, pc_syn, dist=euclidean)
                a = np.array(pc_path)
                fd_pitch = np.sqrt(np.mean((a[:,0]-a[:,1])**2))
                
                d_pscore[pf] = pscore
                d_pitch_ppc[pf] = pitch_ppc
                d_pitch_distance[pf] = pitch_distance
                d_fd_pitch[pf] = fd_pitch
            
        if "energy" in features:
            ec_ref = self._get_energy_contour(loaded_ref_wav, e_logscale).reshape(-1, 1)
            ec_syn = self._get_energy_contour(loaded_syn_wav, e_logscale).reshape(-1, 1)
            distance, ec_path = fastdtw(ec_ref, ec_syn, dist=euclidean)
            frames_tot, min_cost_tot = self.mt.calculate_mcd_distance(ec_ref, ec_syn, ec_path)
            energy_distance = distance/frames_tot
            escore = min_cost_tot/frames_tot
            a = np.array(ec_path)
            fd_energy = np.sqrt(np.mean((a[:,0]-a[:,1])**2))
            
            a = ec_ref[np.array(ec_path)[:,0], 0]
            b = ec_syn[np.array(ec_path)[:,1], 0]
            energy_ppc = np.corrcoef(a, b)[0, 1]
        
        data = {
            "mcd": {
                "distance": mcd_distance if "mcd" in features else None,
                "score": mcd if "mcd" in features else None,
                "fd": fd_mcd if "mcd" in features else None,
                "ppc": 0 if "mcd" in features else None,
            },
            "pitch": {
                "distance": d_pitch_distance["pitch"] if "pitch" in features else None,
                "score": d_pscore["pitch"] if "pitch" in features else None,
                "fd": d_fd_pitch["pitch"] if "pitch" in features else None,
                "ppc": d_pitch_ppc["pitch"] if "pitch" in features else None,
            },
            "pitch_remove0": {
                "distance": d_pitch_distance["pitch_remove0"] if "pitch_remove0" in features else None,
                "score": d_pscore["pitch_remove0"] if "pitch_remove0" in features else None,
                "fd": d_fd_pitch["pitch_remove0"] if "pitch_remove0" in features else None,
                "ppc": d_pitch_ppc["pitch_remove0"] if "pitch_remove0" in features else None,
            },
            "energy": {
                "distance": energy_distance if "energy" in features else None,
                "score": escore if "energy" in features else None,
                "fd": fd_energy if "energy" in features else None,
                "ppc": energy_ppc if "energy" in features else None,
            }
        }
        return data
    