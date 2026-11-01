# Copyright (c) SITP
import math

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from model.common_blocks import CABlock, CEBlock, Downsample, InputProj, OutputProj, ResBlock, Upsample
from model.swin_layers import SwinTransformerLayer

class FANBaseTransformer(nn.Module):
    def __init__(self, img_size=128, in_chans=312,
                 out_chans=1, embed_dim=64, depths=[2, 4, 6, 4, 2], num_heads=[1, 2, 4, 2, 1],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0.5, attn_drop_rate=0.2, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, dowsample=Downsample, upsample=Upsample, shift_flag=True, **kwargs):
        super().__init__()

        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.win_size = win_size
        self.reso = img_size
        self.in_chans = in_chans
        self.output_chans = out_chans

        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[2]
        dec_dpr = enc_dpr[::-1]


        self.input_proj = InputProj(in_channel=in_chans, out_channel=embed_dim, kernel_size=3, stride=1,
                                    act_layer=nn.LeakyReLU)


        self.output_proj = OutputProj(in_channel=embed_dim * 2, out_channel=out_chans, kernel_size=3, stride=1)

        self.encoderlayer_0 = SwinTransformerLayer(dim=embed_dim,
                                                input_resolution=(img_size,
                                                                  img_size),
                                                depth=depths[0],
                                                num_heads=num_heads[0],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                shift_flag=shift_flag)
        self.dowsample_0 = dowsample(embed_dim, embed_dim * 2)
        self.encoderlayer_1 = SwinTransformerLayer(dim=embed_dim * 2,
                                                input_resolution=(img_size // 2,
                                                                  img_size // 2),
                                                depth=depths[1],
                                                num_heads=num_heads[1],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                shift_flag=shift_flag)
        self.dowsample_1 = dowsample(embed_dim * 2, embed_dim * 4)

        self.conv = SwinTransformerLayer(dim=embed_dim * 4,
                                      input_resolution=(img_size // (2 ** 2),
                                                        img_size // (2 ** 2)),
                                      depth=depths[2],
                                      num_heads=num_heads[2],
                                      win_size=win_size,
                                      mlp_ratio=self.mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop_rate, attn_drop=attn_drop_rate,
                                      drop_path=conv_dpr,
                                      norm_layer=norm_layer,
                                      use_checkpoint=use_checkpoint,
                                      shift_flag=shift_flag)

        self.upsample_0 = upsample(embed_dim * 4, embed_dim * 2)
        self.decoderlayer_0 = SwinTransformerLayer(dim=embed_dim * 4,
                                                input_resolution=(img_size // 2,
                                                                  img_size // 2),
                                                depth=depths[3],
                                                num_heads=num_heads[3],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[:depths[3]],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                shift_flag=shift_flag)

        self.upsample_1 = upsample(embed_dim * 4, embed_dim)
        self.decoderlayer_1 = SwinTransformerLayer(dim=embed_dim * 2,
                                                input_resolution=(img_size,
                                                                  img_size),
                                                depth=depths[4],
                                                num_heads=num_heads[4],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[3:4]):sum(depths[3:5])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                shift_flag=shift_flag)

        self.net_process1 = ResBlock(embed_dim, embed_dim, use_layer_norm=True)
        self.net_process2 = ResBlock(embed_dim * 2, embed_dim * 2, use_layer_norm=True)
        self.ce_process = CEBlock(in_chans)
        self.ca_process = CABlock(channel=in_chans, reduction=4)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, win_size={self.win_size}"

    def forward(self, x, mask=None):

        y = self.ce_process(x)
        y_c, att_channel = self.ca_process(y)
        y_new = y_c
        y = self.input_proj(y_new)
        y = self.net_process1(y)
        y = y.flatten(2).transpose(1, 2).contiguous()

        conv0 = self.encoderlayer_0(y, mask=mask)
        pool0 = self.dowsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0, mask=mask)
        pool1 = self.dowsample_1(conv1)

        conv4 = self.conv(pool1, mask=mask)

        up0 = self.upsample_0(conv4)
        deconv0 = torch.cat([up0, conv1], -1)
        deconv0 = self.decoderlayer_0(deconv0, mask=mask)
        up1 = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1, conv0], -1)
        deconv1 = self.decoderlayer_1(deconv1, mask=mask)

        B, L, C = deconv1.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        deconv1 = deconv1.transpose(1, 2).view(B, C, H, W)
        deconv1 = self.net_process2(deconv1)
        y = self.output_proj(deconv1)

        return y, att_channel

    def flops(self):
        flops = 0
        flops += self.ce_process.flops(self.reso, self.reso)
        flops += self.ca_process.flops()
        flops += self.input_proj.flops(self.reso, self.reso)
        flops += self.net_process1.flops(self.reso, self.reso)
        flops += self.encoderlayer_0.flops() + self.dowsample_0.flops(self.reso, self.reso)
        flops += self.encoderlayer_1.flops() + self.dowsample_1.flops(self.reso // 2, self.reso // 2)

        flops += self.conv.flops()

        flops += self.upsample_0.flops(self.reso // 4, self.reso // 4) + self.decoderlayer_0.flops()
        flops += self.upsample_1.flops(self.reso // 2, self.reso // 2) + self.decoderlayer_1.flops()

        flops += self.net_process2.flops(self.reso, self.reso)
        flops += self.output_proj.flops(self.reso, self.reso)
        return flops

