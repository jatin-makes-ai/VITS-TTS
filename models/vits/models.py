import torch
from torch import nn
from torch.nn import functional as F

from .text_encoder import TextEncoder
from .posterior_encoder import PosteriorEncoder
from .generator import Generator
from .modules import ResidualCouplingBlock

try:
    from monotonic_align import maximum_path
except ImportError:
    from monotonic_align.core import maximum_path

class SynthesizerTrn(nn.Module):
    def __init__(self, 
                 n_vocab, 
                 spec_channels, 
                 segment_size,
                 inter_channels, 
                 hidden_channels, 
                 filter_channels,
                 n_heads, 
                 n_layers, 
                 kernel_size, 
                 p_dropout, 
                 resblock, 
                 resblock_kernel_sizes, 
                 resblock_dilation_sizes, 
                 upsample_rates, 
                 upsample_initial_channel, 
                 upsample_kernel_sizes,
                 **kwargs):
        super().__init__()
        
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.segment_size = segment_size

        # 1. Text Encoder (Prior)
        self.enc_p = TextEncoder(n_vocab, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        
        # 2. Decoder (HiFi-GAN Generator)
        self.dec = Generator(inter_channels, resblock_kernel_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes)
        
        # 3. Posterior Encoder
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16)
        
        # 4. Flow (Normalizing Flow)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4)

    def forward(self, x, x_lengths, y, y_lengths):
        # 1. Get Text Latents
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        # 2. Get Audio Latents (from linear spectrogram)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths)

        # 3. Flow: Transform z (audio space) to z_p (text space)
        z_p = self.flow(z, y_mask, g=None)

        # 4. Monotonic Alignment Search (neg_cent [B, T_y, T_x], mask [B, T_y, T_x])
        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * logs_p).unsqueeze(2)  # [B, C, 1, T_x]
            diff = z_p.unsqueeze(3) - m_p.unsqueeze(2)  # [B, C, T_y, T_x]
            neg_cent1 = torch.sum(-0.5 * s_p_sq_r * diff**2, 1)  # [B, T_y, T_x]
            neg_cent2 = torch.sum(-1.0 * logs_p, 1).unsqueeze(1)  # [B, 1, T_x]
            neg_cent = neg_cent1 + neg_cent2

            attn_mask = (torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, 3)).squeeze(1).transpose(1, 2)
            attn = maximum_path(neg_cent, attn_mask).unsqueeze(1).detach()

        # 5. Expand text features to match audio length using the alignment
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        # 6. Slice a random segment for the Generator (to save VRAM)
        # We don't synthesize the whole audio during training to stay under 4GB
        z_slice, ids_slice = self.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice)

        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def rand_slice_segments(self, x, lengths, segment_size):
        # Slices segment_size//hop frames for the generator (hop=256)
        b, c, t = x.size()
        segment_frames = segment_size // 256
        # Clamp against actual tensor length — lengths (from audio_lengths // hop)
        # can exceed t because STFT with center=False produces fewer frames
        safe_lengths = lengths.clamp(max=t)
        ids_str_max = (safe_lengths - segment_frames).clamp(min=0)
        ids_str = (torch.rand([b]).to(device=x.device) * (ids_str_max.float() + 1e-8)).long()
        ret = torch.zeros([b, c, segment_frames], device=x.device, dtype=x.dtype)
        for i in range(b):
            start = ids_str[i].item()
            end = min(start + segment_frames, safe_lengths[i].item())
            if end > start:
                ret[i, :, : end - start] = x[i, :, start:end]
        return ret, ids_str