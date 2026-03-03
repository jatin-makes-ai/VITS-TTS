from scripts.training.dataset import VITSDataset, collate_fn
from torch.utils.data import DataLoader

dataset = VITSDataset("data/processed/train_list.txt")
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Get one batch
texts, text_lens, audios, audio_lens, ids = next(iter(loader))

print(f"Batch Text Shape: {texts.shape}")   # [Batch, Max_Text_Len]
print(f"Batch Audio Shape: {audios.shape}") # [Batch, Max_Audio_Len]
print(f"Text Lengths: {text_lens}")