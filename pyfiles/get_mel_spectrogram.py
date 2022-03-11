import numpy as np
import librosa
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from librosa.filters import mel as librosa_mel_fn
from scipy.io.wavfile import read
import random

class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length=800, hop_length=200, win_length=800,
                 window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :].astype(np.float32))

        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(
            torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False)
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction
    
def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output
    
def load_wav_to_torch(full_path=None, audio=None, sampling_rate=None):
#     sampling_rate, data = read(full_path)
    if full_path==None:
        data = audio
    else:
        data, sampling_rate = librosa.load(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, args):
        self.max_wav_value = args["max_wav_value"]
        self.sampling_rate = args["sampling_rate"]
        self.stft = TacotronSTFT(
            args["filter_length"], args["hop_length"], args["win_length"],
            args["n_mel_channels"], args["sampling_rate"], args["mel_fmin"],
            args["mel_fmax"])
        random.seed(1234)

    def get_mel(self, filename, audio=None, sampling_rate=None):
        audio, sampling_rate = load_wav_to_torch(filename, audio, sampling_rate)
        assert(sampling_rate == self.stft.sampling_rate)
#         audio_norm = audio / self.max_wav_value
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)

        return melspec
    
class audio2mel():
    """
    To generate mel-spectrogram from audio

    ------------
    Parameters
    ------------

    args : dir or None
        it indicates the parameters of melspectrogram generation and when None is inserted, args becomes the default parameters for Tacotron2. The default parameters are listed below.
        samples:
        args = {}
        args["max_wav_value"] = 2**15
        args["filter_length"] = 1024
        args["hop_length"] = 256
        args["win_length"] = 1024
        args["n_mel_channels"] = 80
        args["sampling_rate"] = 22050
        args["mel_fmin"] = 0
        args["mel_fmax"] = 8000
        
    
    sampling_rate : int or None
        sampling rate
        
    ------------
    Attributes
    ------------
    
    mel_generation : To generate mel-spectrogram from audio
        
    """
    def __init__(self, args=None, sampling_rate=None):
        if args is None:
            args = {}
            args["max_wav_value"] = 2**15
            args["filter_length"] = 1024
            args["hop_length"] = 256
            args["win_length"] = 1024
            args["n_mel_channels"] = 80
            args["sampling_rate"] = 22050
            args["mel_fmin"] = 0
            args["mel_fmax"] = 8000
        self.args = args
        self.sampling_rate = sampling_rate
    
    def mel_generation(self, x, sampling_rate=None):
        """
        To generate melspectrogram from audio

        ------------
        Parameters
        ------------

        x : ndarray, shape=(length)
            audio
            
        sampling_rate : int or None
            sampling_rate
            
        ------------
        Returns
        ------------

        mel : ndarray, shape=(args["n_mel_channels"], time-length)
            mel-spectrogram
            
        ------------
        Examples
        ------------
            
        am = audio2mel()
        x, sampling_rate = librosa.load(path, sampling_rate)
        mel = am.mel_generation(x, sampling_rate)

        ------------

        """
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        data_loader = TextMelLoader(self.args)
        mel = data_loader.get_mel(None, x, sampling_rate)
        return mel