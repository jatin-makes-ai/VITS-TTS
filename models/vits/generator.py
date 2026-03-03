import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

LRELU_SLOPE = 0.1

class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=(kernel_size * dilation[0] - dilation[0]) // 2)),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=(kernel_size * dilation[1] - dilation[1]) // 2)),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=(kernel_size * dilation[2] - dilation[2]) // 2))
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=(kernel_size - 1) // 2)) for _ in range(3)
        ])

    forward = lambda self, x: [ (x := x + self.convs2[i](torch.nn.functional.leaky_relu(self.convs1[i](torch.nn.functional.leaky_relu(x, LRELU_SLOPE)), LRELU_SLOPE))) for i in range(3) ][-1]

class Generator(torch.nn.Module):
    def __init__(self, initial_channels=192, resblock_kernel_sizes=[3,7,11], upsample_rates=[8,8,2,2], upsample_initial_channel=512, upsample_kernel_sizes=[16,16,4,4]):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.conv_pre = weight_norm(Conv1d(initial_channels, upsample_initial_channel, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)), k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for k in resblock_kernel_sizes:
                self.resblocks.append(ResBlock1(ch, k))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

    def forward(self, x):
        x = self.conv_pre(x)
        for i, up in enumerate(self.ups):
            x = torch.nn.functional.leaky_relu(up(x), LRELU_SLOPE)
            xs = sum(self.resblocks[i*self.num_kernels + j](x) for j in range(self.num_kernels)) / self.num_kernels
            x = xs
        return torch.tanh(self.conv_post(x))