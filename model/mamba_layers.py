# Copyright (c) SITP
"""Mamba layers used by FAN mamba variants."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from timm.models.layers import DropPath

from model.common_blocks import Mlp


class MambaVisionMixer(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            layer_idx=None,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)

        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(
            input=x,
            weight=self.conv1d_x.weight,
            bias=self.conv1d_x.bias,
            padding="same",
            groups=self.d_inner // 2,
        ))
        z = F.silu(F.conv1d(
            input=z,
            weight=self.conv1d_z.weight,
            bias=self.conv1d_z.bias,
            padding="same",
            groups=self.d_inner // 2,
        ))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=None,
        )
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)

    def flops(self, sequence_length):
        inner_half = self.d_inner // 2
        flops = sequence_length * self.d_model * self.d_inner
        flops += 2 * sequence_length * inner_half * self.d_conv
        flops += sequence_length * inner_half * (self.dt_rank + 2 * self.d_state)
        flops += sequence_length * self.dt_rank * inner_half
        flops += sequence_length * inner_half * self.d_state * 4
        flops += sequence_length * self.d_inner * self.d_model
        return flops


class MambaBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.5, drop_path=0.1, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, mlp_block=Mlp, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.mixer = MambaVisionMixer(d_model=dim, d_state=8, d_conv=3, expand=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_block(in_features=dim, hidden_features=int(dim * mlp_ratio),
                             act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

    def flops(self, sequence_length):
        flops = sequence_length * self.dim * 2
        flops += self.mixer.flops(sequence_length)
        flops += sequence_length * self.dim * 2
        side = int(math.sqrt(sequence_length))
        flops += self.mlp.flops(side, side)
        return flops


class MambaLayerStack(nn.Module):
    def __init__(self, dim, depth, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.mamba_block = nn.ModuleList([MambaBlock(dim=dim) for _ in range(depth)])

    def extra_repr(self):
        return f"dim={self.dim}, depth={self.depth}"

    def forward(self, x, mask=None):
        for blk in self.mamba_block:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x, mask)
        return x

    def flops(self, sequence_length):
        return sum(blk.flops(sequence_length) for blk in self.mamba_block)
