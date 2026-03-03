import torch
from torch import nn
from torch.nn import functional as F

class PosteriorEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        # Initial projection from spectrogram channels to hidden channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        
        # Residual blocks (simplified WN structure)
        self.enc = nn.ModuleList()
        for i in range(n_layers):
            self.enc.append(
                nn.utils.weight_norm(
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size, 
                              stride=1, padding=(kernel_size - 1) // 2)
                )
            )

        # Final projection to Mean and Logs
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def sequence_mask(self, length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        return (x.unsqueeze(0) < length.unsqueeze(1)).unsqueeze(1).float()

    def forward(self, x, x_lengths):
        # x: [Batch, Spectrogram_Channels, Time]
        x_mask = self.sequence_mask(x_lengths, x.size(2)).to(x.dtype)
        x = self.pre(x) * x_mask
        
        for layer in self.enc:
            # Residual connection with ReLU and mask
            y = layer(x)
            x = (x + F.relu(y)) * x_mask
            
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        # Clamp logs to prevent exp() explosion → NaN early in training
        logs = torch.clamp(logs, min=-10, max=10)
        # Reparameterization: z = m + exp(logs) * eps
        z = m + torch.exp(logs) * torch.randn_like(m)
        return z, m, logs, x_mask