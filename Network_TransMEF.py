# -*- coding: utf-8 -*-
# Citation:
# @article{qu2021transmef,
#   title={TransMEF: A Transformer-Based Multi-Exposure Image Fusion Framework using Self-Supervised Multi-Task Learning},
#   author={Qu, Linhao and Liu, Shaolei and Wang, Manning and Song, Zhijian},
#   journal={arXiv preprint arXiv:2112.01030},
#   year={2021}
# }
import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange


def save_grad(grads, name):
    def hook(grad):
        grads[name] = grad

    return hook


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class OutConv(nn.Module):
    """1*1 conv before the output"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    """features extraction"""

    def __init__(self):
        super(Encoder, self).__init__()
        self.inc = DoubleConv(1, 16)
        self.layer1 = DoubleConv(16, 32)
        self.layer2 = DoubleConv(32, 48)

    def forward(self, x, grads=None, name=None):
        x = self.inc(x)
        x = self.layer1(x)
        x = self.layer2(x)

        if grads is not None:
            x.register_hook(save_grad(grads, name + "_x"))
        return x


class Encoder_Trans(nn.Module):
    """features extraction"""

    def __init__(self):
        super(Encoder_Trans, self).__init__()
        self.inc = DoubleConv(1, 16)
        self.layer1 = DoubleConv(17, 32)
        self.layer2 = DoubleConv(32, 48)
        self.transformer = ViT(image_size=256, patch_size=16, dim=256, depth=12, heads=16, mlp_dim=1024, dropout=0.1,
                               emb_dropout=0.1)

    def forward(self, x, grads=None, name=None):
        x_e = self.inc(x)
        x_t = self.transformer(x)
        x = torch.cat((x_e, x_t), dim=1)
        x = self.layer1(x)
        x = self.layer2(x)

        if grads is not None:
            x.register_hook(save_grad(grads, name + "_x"))
        return x


class Decoder(nn.Module):
    """reconstruction"""

    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = DoubleConv(48, 32)
        self.layer2 = DoubleConv(32, 16)
        self.outc = OutConv(16, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        output = self.outc(x)
        return output


class Decoder_Trans(nn.Module):
    """reconstruction"""

    def __init__(self):
        super(Decoder_Trans, self).__init__()
        self.layer3 = DoubleConv(49, 48)
        self.layer4 = DoubleConv(48, 48)
        self.layer1 = DoubleConv(48, 32)
        self.layer2 = DoubleConv(32, 16)
        self.outc = OutConv(16, 1)

    def forward(self, x):
        x = self.layer4(self.layer3(x))
        x = self.layer1(x)
        x = self.layer2(x)
        output = self.outc(x)
        return output


class SimNet(nn.Module):
    """easy network for self-reconstruction task"""

    def __init__(self):
        super(SimNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=1, dim_head=64,
                 dropout=0., emb_dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim)
        )
        self.dim = dim
        self.patch_size = patch_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.convd1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, img):
        x = self.to_patch_embedding(img)  # [B,256,256]
        b, n, _ = x.shape

        x = self.transformer(x)
        x = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.patch_size, h=16, c=1)(x)  # [B,1,256,256]

        return x


class TransNet(nn.Module):
    """U-based network for self-reconstruction task"""

    def __init__(self):
        super(TransNet, self).__init__()

        self.encoder = Encoder_Trans()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
