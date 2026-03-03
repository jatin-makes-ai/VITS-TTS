import torch
from torch import nn
from models.vits.attentions import MultiHeadAttention
import math

class TextEncoder(nn.Module):
    def __init__(self,
                 n_vocab,
                 out_channels=192,
                 hidden_channels=192,
                 filter_channels=768,
                 n_heads=2,
                 n_layers=6,
                 kernel_size=3,
                 p_dropout=0.1):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # 1. Phoneme Embedding Layer
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        # 2. Transformer Layers (Simplified for this project)
        self.encoder = nn.ModuleList([
            MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout)
            for _ in range(n_layers)
        ])
        
        # 3. Final Projection to Latent Space
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        # x: [Batch, Sequence_Len]
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [B, T, H]
        x = torch.transpose(x, 1, 2)  # [B, H, T] for Conv1d

        # Base padding mask: [B, T]
        base_mask = self.sequence_mask(x_lengths, x.size(2))
        # Attention mask: [B, 1, 1, T] so it broadcasts over heads and query positions
        attn_mask = base_mask[:, None, None, :].to(x.dtype)

        for layer in self.encoder:
            x = layer(x, x, attn_mask=attn_mask)

        # Projection mask for conv output: [B, 1, T]
        proj_mask = base_mask[:, None, :].to(x.dtype)
        # Project to Mean and Logs (for the VAE part of VITS)
        stats = self.proj(x) * proj_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)

        # Return the simple [B, 1, T] mask used for downstream modules
        return x, m, logs, proj_mask

    def sequence_mask(self, length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        return x.unsqueeze(0) < length.unsqueeze(1)