import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from utils.text_utils import text_to_sequence

class VITSDataset(Dataset):
    def __init__(self, manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.metadata = [line.strip().split("|") for line in f]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        wav_path, phonemes, speaker_id = self.metadata[index]
        
        # 1. Load Audio
        audio, sampling_rate = torchaudio.load(wav_path)
        audio = audio.squeeze(0) # [1, T] -> [T]
        
        # 2. Convert Phonemes to Integers
        text_seq = torch.LongTensor(text_to_sequence(phonemes))
        
        return text_seq, audio, int(speaker_id)

def collate_fn(batch):
    """
    Pads sequences to the same length within a batch.
    Important for GPU memory efficiency.
    """
    # Sort by text length (common optimization for RNNs/Transformers)
    batch.sort(key=lambda x: x[0].size(0), reverse=True)
    
    texts, audios, speaker_ids = zip(*batch)
    
    # Pad Text
    text_lengths = torch.LongTensor([len(x) for x in texts])
    texts_padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True)
    
    # Pad Audio
    audio_lengths = torch.LongTensor([len(x) for x in audios])
    # audio padding is a bit different; we usually pad with 0s at the end
    max_audio_len = max(audio_lengths)
    audios_padded = torch.zeros(len(audios), max_audio_len)
    for i in range(len(audios)):
        audios_padded[i, :audio_lengths[i]] = audios[i]
        
    return texts_padded, text_lengths, audios_padded, audio_lengths, torch.LongTensor(speaker_ids)