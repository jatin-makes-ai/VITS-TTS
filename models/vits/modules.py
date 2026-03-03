import torch
from torch import nn
from torch.nn import Conv1d

class WN(nn.Module):
    """WaveNet-style residual blocks used inside the Flow layers."""
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
        super(WN, self).__init__()
        self.n_layers = n_layers
        self.res_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        self.cond_layer = nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1) if gin_channels > 0 else None

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            self.res_layers.append(nn.utils.weight_norm(
                Conv1d(hidden_channels, 2*hidden_channels, kernel_size, dilation=dilation, padding=padding)))
            self.skip_layers.append(nn.utils.weight_norm(
                Conv1d(hidden_channels, hidden_channels, 1)))

    def forward(self, x, x_mask, g=None):
        output = torch.zeros_like(x)
        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.res_layers[i](x)
            if g is not None:
                x_in += g[:, 2*i*x.shape[1] : 2*(i+1)*x.shape[1], :]
            
            acts = torch.split(x_in, x_in.shape[1] // 2, dim=1)
            x_tanh, x_sigmoid = acts[0], acts[1]
            x_res = torch.tanh(x_tanh) * torch.sigmoid(x_sigmoid)
            
            x = (x + self.skip_layers[i](x_res)) * x_mask
            output = (output + x_res) * x_mask
        return output

class ResidualCouplingLayer(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, reverse=False):
        super().__init__()
        self.channels = channels
        self.reverse = reverse
        self.half_channels = channels // 2
        self.pre = Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout)
        self.post = Conv1d(hidden_channels, self.half_channels, 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels]*2, dim=1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        m = self.post(h) * x_mask
        
        if reverse or self.reverse:
            x1 = (x1 - m) * x_mask
        else:
            x1 = (x1 + m) * x_mask
        return torch.cat([x0, x1], dim=1)

class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows=4):
        super().__init__()
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, reverse=bool(i % 2)))

    def forward(self, x, x_mask, g=None, reverse=False):
        if reverse:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=True)
        else:
            for flow in self.flows:
                x = flow(x, x_mask, g=g, reverse=False)
        return x