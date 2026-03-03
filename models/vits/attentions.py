import torch
from torch import nn
from torch.nn import functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, channels, out_channels, n_heads, window_size=None, heads_share_window=True, p_dropout=0.0, proximal_bias=False, proximal_init=False):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.window_size = window_size
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x_shape = q.shape
        c_shape = k.shape
        q = q.view(x_shape[0], self.n_heads, self.k_channels, x_shape[2]).transpose(2, 3)
        k = k.view(c_shape[0], self.n_heads, self.k_channels, c_shape[2])
        v = v.view(c_shape[0], self.n_heads, self.k_channels, c_shape[2]).transpose(2, 3)

        attn = (q @ k) * (self.k_channels ** -0.5)

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e4)

        attn = torch.softmax(attn, dim=-1)
        attn = self.drop(attn)

        x = (attn @ v).transpose(2, 3).reshape(x_shape[0], self.channels, x_shape[2])
        x = self.conv_o(x)
        return x