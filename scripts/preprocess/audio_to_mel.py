import torch
import librosa
import numpy as np

def get_mel_spectrogram(wav_path, sr=22050):
    # 1. Load audio
    y, _ = librosa.load(wav_path, sr=sr)
    
    # 2. Compute Mel-Spectrogram (VITS standard params)
    # n_fft=1024, hop_length=256, win_length=1024 are common for 22k sr
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    
    # 3. Convert to Log Scale (Decibels)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

if __name__ == "__main__":
    # Test on one of our toy files
    mel = get_mel_spectrogram("data/raw/wavs/toy_000.wav")
    print(f"Mel-Spectrogram shape: {mel.shape}") # Expect (80, T)