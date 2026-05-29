import numpy as np
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, L, W, AR, pad=True):
        super().__init__()
        self.norm1 = nn.InstanceNorm1d(L)
        s = 1
        # padding calculation: https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338/2
        if pad:
            padding = int(1 / 2 * (1 - L + AR * (W - 1) - s + L * s))
        else:
            padding = 0
        self.conv1 = nn.Conv1d(L, L, W, dilation=AR, padding=padding)
        self.norm2 = nn.InstanceNorm1d(L)
        self.conv2 = nn.Conv1d(L, L, W, dilation=AR, padding=padding)

    def forward(self, x):
        out = self.norm1(x)
        out = torch.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = out + x
        return out


class Pangolin(nn.Module):
    '''
    Predicting RNA splicing from DNA sequence using Pangolin.
    https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02664-4
    '''
    def __init__(self,
                 L: int = None,
                 W: np.array = None,
                 AR: np.array = None,
                 in_channels: int = 1):
        super().__init__()
        if L is None:
            # number of convolutional filters.
            L = 32
        if W is None:
            # convolution window size in residual units
            W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                            21, 21, 21, 21, 41, 41, 41, 41])
        if AR is None:
            # atrous rate in residual units
            AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                             10, 10, 10, 10, 25, 25, 25, 25])

        self.L = L
        self.W = W
        self.AR = AR

        self.conv1 = nn.Conv1d(in_channels, L, 1)
        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(W)):
            self.resblocks.append(ResBlock(L, W[i], AR[i]))
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.conv_last = nn.Conv1d(L, 1, 1)

    def forward(self, x):
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(self.W)):
            conv = self.resblocks[i](conv)
            if (((i + 1) % 4 == 0) or ((i + 1) == len(self.W))):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense

        out = self.conv_last(skip)
        return out

