import torch
import librosa
import numpy as np
import torch.utils.data
from torch import stft

def get_mel_spectrogram(wav_path, sr=22050):
    # 1. Load audio
    y, _ = librosa.load(wav_path, sr=sr)
    
    # 2. Compute Mel-Spectrogram (VITS standard params)
    # n_fft=1024, hop_length=256, win_length=1024 are common for 22k sr
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    
    # 3. Convert to Log Scale (Decibels)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

def spectrogram_torch(y, n_fft, hop_size, win_size, center=False):
    """Calculates the linear spectrogram (STFT) for the Posterior Encoder."""
    if torch.min(y) < -1.: print('min value is ', torch.min(y))
    if torch.max(y) > 1.: print('max value is ', torch.max(y))

    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    hann_window = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, 
                      window=hann_window, center=center, pad_mode='reflect', 
                      normalized=False, onesided=True, return_complex=True)

    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-6)
    return spec

if __name__ == "__main__":
    # Test on one of our toy files
    mel = get_mel_spectrogram("data/raw/wavs/toy_000.wav")
    print(f"Mel-Spectrogram shape: {mel.shape}") # Expect (80, T)