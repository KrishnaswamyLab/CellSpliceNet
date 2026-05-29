
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from operator import mul
from functools import reduce
from sinkhorn_transformer import SinkhornTransformer


class AxialPositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        axial_shape: tuple[int, ...],
        axial_dims: tuple[int, ...] | None = None
    ):
        super().__init__()

        self.dim = dim
        self.shape = axial_shape
        self.max_seq_len = reduce(mul, axial_shape, 1)

        self.summed = axial_dims is None
        axial_dims = ((dim,) * len(axial_shape)) if self.summed else axial_dims

        assert len(self.shape) == len(axial_dims), 'number of axial dimensions must equal the number of dimensions in the shape'
        assert self.summed or not self.summed and sum(axial_dims) == dim, f'axial dimensions must sum up to the target dimension {dim}'

        self.weights = nn.ParameterList([])

        for ind, (shape, axial_dim) in enumerate(zip(self.shape, axial_dims)):
            ax_shape = [1] * len(self.shape)
            ax_shape[ind] = shape
            ax_shape = (1, *ax_shape, axial_dim)
            ax_emb = nn.Parameter(torch.zeros(ax_shape).normal_(0, 1))
            self.weights.append(ax_emb)

    def forward(self, x):

        batch, seq_len, _ = x.shape
        assert (seq_len <= self.max_seq_len), f'Sequence length ({seq_len}) must be less than the maximum sequence length allowed ({self.max_seq_len})'

        embs = []

        for ax_emb in self.weights:
            axial_dim = ax_emb.shape[-1]
            expand_shape = (batch, *self.shape, axial_dim)
            emb = ax_emb.expand(expand_shape).reshape(batch, self.max_seq_len, axial_dim)
            embs.append(emb)

        pos_emb = sum(embs) if self.summed else torch.cat(embs, dim=-1)

        return pos_emb[:, :seq_len].to(x)


class ResBlock(nn.Module):

    def __init__(self, L, W, AR, pad=True):
        super(ResBlock, self).__init__()
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


class AttnBlock(nn.Module):
    def __init__(self,
                 dim: int = 32,
                 depth: int = 6,
                 max_seq_len: int = 4096,
                 bucket_size: int = 64,
                 causal: bool = False,
                 reversible: bool = False) -> None:
        super().__init__()
        assert max_seq_len % bucket_size == 0, f'max_seq_len {max_seq_len} must be divisible by bucket_size {bucket_size}.'
        axial_position_shape = ((max_seq_len // bucket_size), bucket_size)
        self.pos_emb = AxialPositionalEmbedding(dim, axial_position_shape)
        self.attn = SinkhornTransformer(
            dim=dim,
            depth=depth,
            bucket_size=bucket_size,
            heads=8,
            n_local_attn_heads=2,
            max_seq_len=max_seq_len,
            attn_layer_dropout=0.1,
            layer_dropout=0.1,
            ff_dropout=0.1,
            ff_chunks=10,
            causal=causal,
            reversible=reversible,
            non_permutative=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = torch.transpose(x, 1, 2).contiguous()
        x = x + self.pos_emb(x)
        x = self.attn(x)
        x = self.norm(x)
        x = torch.transpose(x, 1, 2).contiguous()
        return x


class SpEncoder(nn.Module):
    def __init__(self, L, in_channels: int = 1):
        super().__init__()
        self.W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                             21, 21, 21, 21, 21, 21, 21, 21])
        self.AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                              10, 10, 10, 10, 20, 20, 20, 20])
        self.conv1 = nn.Conv1d(in_channels, L, 1)
        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(self.W)):
            self.resblocks.append(ResBlock(L, self.W[i], self.AR[i]))
            if (((i + 1) % 4 == 0) or ((i + 1) == len(self.W))):
                self.convs.append(nn.Conv1d(L, L, 1))

    def forward(self, x):
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(self.W)):
            conv = self.resblocks[i](conv)  # important
            if (((i + 1) % 4 == 0) or ((i + 1) == len(self.W))):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense

        return skip


class SpliceTransformer(nn.Module):
    '''
    SpliceTransformer predicts tissue-specific splicing linked to human diseases.
    https://www.nature.com/articles/s41467-024-53088-6
    '''
    def __init__(self,
                 in_channels : int = 1,
                 dim: int = 32,
                 attn_depth: int = 2,
                 dim_encoder: int = 32,
                 max_seq_len: int = 4096,
                 bucket_size: int = 64) -> None:
        super().__init__()
        self.dim_encoder = dim_encoder
        self.encoder = SpEncoder(L=dim_encoder, in_channels=in_channels)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, dim, 1),
            nn.Conv1d(dim, dim, 1),
        )
        self.conv2 = nn.Conv1d(dim + self.dim_encoder, dim, 1)
        self.max_seq_len = max_seq_len
        self.attn = AttnBlock(
            dim,
            depth=attn_depth,
            max_seq_len=self.max_seq_len,
            bucket_size=bucket_size)
        self.conv_last = nn.Conv1d(dim, 1, 1)

    def forward(self, x):
        x = x[:, :, :self.max_seq_len]
        feat1 = self.encoder(x)
        feat2 = self.conv1(x)

        seq_len = x.size(2)
        odd_fix = seq_len & 1
        feat1 = F.pad(feat1, (-(seq_len-self.max_seq_len)//2, -(seq_len-self.max_seq_len)//2 + odd_fix))
        feat2 = F.pad(feat2, (-(seq_len-self.max_seq_len)//2, -(seq_len-self.max_seq_len)//2 + odd_fix))

        emb = torch.concat([feat1, feat2], dim=1)
        emb = self.conv2(emb)
        att = self.attn(emb)
        out = self.conv_last(att)

        return out