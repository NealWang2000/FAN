# Copyright (c) SITP
"""Shared network blocks used by FAN model variants."""

import math

import torch
import torch.nn as nn


class LayerNorm2D(nn.Module):
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x = x * self.gamma + self.beta
        return x

    def flops(self, H, W):
        return H * W * self.num_channels


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1, use_layer_norm=False):
        super().__init__()
        self.strides = strides
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        if use_layer_norm:
            self.LN = LayerNorm2D(out_channel)
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        out = self.block(x)
        if hasattr(self, "LN"):
            out = self.LN(out)
        return out + self.conv11(x)

    def flops(self, H, W):
        output_h = H // self.strides
        output_w = W // self.strides
        flops = output_h * output_w * self.in_channel * self.out_channel * 3 * 3
        flops += output_h * output_w * self.out_channel * self.out_channel * 3 * 3
        flops += output_h * output_w * self.in_channel * self.out_channel
        if hasattr(self, "LN"):
            flops += self.LN.flops(output_h, output_w)
        return flops


class CEBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.block1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        out = self.block1(x) + x
        out = self.block2(out) + out
        return out

    def flops(self, H, W):
        # 6 depthwise conv layers, each: H*W*hidden_dim*(3*3) ops
        return H * W * self.hidden_dim * 3 * 3 * 6


class CABlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.channel = channel
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2)
        y = self.avg_pool(x_flat).view(B, C)
        y = self.fc(y).view(B, C, 1)
        x_out = x_flat * y.expand_as(x_flat)
        x_out = x_out.view(B, C, H, W)
        y_att = y.view(B, C, 1, 1)
        return x_out, y_att

    def flops(self):
        return self.channel * self.channel / self.reduction * 2


class Mlp(nn.Module):
    """Two-layer MLP with optional dropout, used by Swin and Mamba blocks."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def flops(self, H, W):
        flops = H * W * self.in_features * self.hidden_features
        flops += H * W * self.hidden_features * self.out_features
        return flops


class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        return self.conv(x).flatten(2).transpose(1, 2).contiguous()

    def flops(self, H, W):
        return H / 2 * W / 2 * self.in_channel * self.out_channel * 4 * 4


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        return self.deconv(x).flatten(2).transpose(1, 2).contiguous()

    def flops(self, H, W):
        return H * 2 * W * 2 * self.in_channel * self.out_channel * 2 * 2


class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None,
                 act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
            act_layer(inplace=True),
        )
        self.norm = norm_layer(out_channel) if norm_layer is not None else None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = H * W * self.in_channel * self.out_channel * 3 * 3
        if self.norm is not None:
            flops += H * W * self.out_channel
        return flops


class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None, act_layer=None):
        super().__init__()
        layers = [
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
        ]
        if act_layer is not None:
            layers.append(act_layer(inplace=True))
        self.proj = nn.Sequential(*layers)
        self.norm = norm_layer(out_channel) if norm_layer is not None else None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = H * W * self.in_channel * self.out_channel * 3 * 3
        if self.norm is not None:
            flops += H * W * self.out_channel
        return flops
