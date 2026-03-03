import torch
import numpy as np
from utils.text_utils import text_to_sequence
from scripts.preprocess.audio_to_mel import spectrogram_torch, get_mel_spectrogram
import librosa

class TextAudioLoader(torch.utils.data.Dataset):
    def __init__(self, manifest_path, hparams):
        # Expecting manifest format: wav_path|phonemes
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.items = [line.strip().split("|") for line in f]
        self.hparams = hparams
        self.sampling_rate = hparams.sampling_rate

    def get_audio_text_pair(self, item):
        wav_path, phonemes = item[0], item[1]
        
        # 1. Load Audio
        audio, _ = librosa.load(wav_path, sr=self.sampling_rate)
        audio = torch.FloatTensor(audio).unsqueeze(0)

        # 2. Get Linear Spectrogram (for Posterior Encoder)
        spec = spectrogram_torch(audio, self.hparams.n_fft, self.hparams.hop_length, self.hparams.win_length)
        spec = spec.squeeze(0)

        # 3. Get Phoneme Sequence
        text_norm = text_to_sequence(phonemes)
        text_norm = torch.IntTensor(text_norm)

        return (text_norm, spec, audio)

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.items[index])

    def __len__(self):
        return len(self.items)

class TextAudioCollate():
    """Zero-pads sequences to the max length in a batch."""
    def __call__(self, batch):
        # Sort by text length for efficiency (optional)
        batch.sort(key=lambda x: x[0].size(0), reverse=True)
        
        t_lengths = torch.LongTensor([x[0].size(0) for x in batch])
        s_lengths = torch.LongTensor([x[1].size(1) for x in batch])
        w_lengths = torch.LongTensor([x[2].size(1) for x in batch])

        # Pad Tensors
        max_t_len = max(t_lengths)
        max_s_len = max(s_lengths)
        max_w_len = max(w_lengths)

        t_padded = torch.LongTensor(len(batch), max_t_len).zero_()
        s_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_s_len).zero_()
        w_padded = torch.FloatTensor(len(batch), 1, max_w_len).zero_()

        for i in range(len(batch)):
            t_padded[i, :batch[i][0].size(0)] = batch[i][0]
            s_padded[i, :, :batch[i][1].size(1)] = batch[i][1]
            w_padded[i, :, :batch[i][2].size(1)] = batch[i][2]

        return t_padded, t_lengths, s_padded, s_lengths, w_padded, w_lengths