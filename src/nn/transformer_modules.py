from functools import wraps

import torch
import torch.nn.functional as F
from torch import nn, einsum  
from einops import rearrange  

class OutputHook:
    """Hook to capture module outputs."""

    def __init__(self):
        self.outputs = []

    def __call__(self, module, input, output):
        self.outputs.append(output)

    def clear(self):
        self.outputs = []


"""
FOLLOWING CODE  WAS COPIED FROM
https://github.com/lucidrains/zorro-pytorch/blob/main/zorro_pytorch/zorro_pytorch.py

 """

  
def exists(val):
    return val is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


def round_down_nearest_multiple(n, divisor):
    return n // divisor * divisor


def pair(t):
    return (t, t) if not isinstance(t, tuple) else t


def cum_mul(it):
    return functools.reduce(lambda x, y: x * y, it, 1)


def divisible_by(numer, denom):
    return (numer % denom) == 0


# decorators


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)  

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


# geglu feedforward


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


# attention
class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=True)

    def forward(self, x, context=None, attn_mask=None, return_attn=False):
   
        x = self.norm(x)
        kv_x = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(kv_x).chunk(2, dim=-1))

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )
        

        q = q * self.scale
        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        

        if exists(attn_mask):
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)
 
        attn = sim.softmax(dim=-1) 
        out = einsum("b h i j, b h j d -> b h i d", attn, v) 
        out = rearrange(out, "b h n d -> b n (h d)")
 
        if return_attn:
            return self.to_out(out), attn 
        else:
            return self.to_out(out)

        # import torchvision 
        # torchvision.utils.save_image(attn_mask.float(), 'attn_mask.png')



# # attention
# class Attention(nn.Module):
#     def __init__(self, dim, dim_head=64, heads=8):
#         super().__init__()
#         self.scale = dim_head**-0.5
#         self.heads = heads
#         inner_dim = dim_head * heads

#         self.norm = LayerNorm(dim)

#         self.to_q = nn.Linear(dim, inner_dim, bias=False)
#         self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
#         self.to_out = nn.Linear(inner_dim, dim, bias=False)

#     def forward(self, x, context=None, attn_mask=None, return_attn=False):
#         x = self.norm(x)
#         kv_x = default(context, x)

#         q, k, v = (self.to_q(x), *self.to_kv(kv_x).chunk(2, dim=-1))

#         q, k, v = map(
#             lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
#         )

#         q = q * self.scale
#         sim = einsum("b h i d, b h j d -> b h i j", q, k)

#         if exists(attn_mask):
#             sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

#         attn = sim.softmax(dim=-1)

#         import pdb;pdb.set_trace()
#         out = einsum("b h i j, b h j d -> b h i d", attn, v)

#         out = rearrange(out, "b h n d -> b n (h d)")

#         if return_attn:
#             return self.to_out(out), attn

#         else:
#             return self.to_out(out)


class Attention_w_dropout(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, context=None, attn_mask=None, return_attn=False):
     
        x = self.norm(x)
        kv_x = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(kv_x).chunk(2, dim=-1))

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )

        q = q * self.scale
        sim = einsum("b h i d, b h j d -> b h i j", q, k)
         
        if exists(attn_mask):
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)          # The numerical properties of a torch.dtype  

        attn = sim.softmax(dim=-1) 
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v) 
        out = rearrange(out, "b h n d -> b n (h d)")  

        if return_attn:
            return self.to_out(out), attn 
        else:
            return self.to_out(out)
 