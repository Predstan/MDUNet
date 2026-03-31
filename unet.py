import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Tuple
from typing_extensions import Literal
from functools import partial

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import os
import warnings

from torch import Tensor
from torch import nn

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_q: int,
        dim_k: int,
        dim_v: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.to_q = nn.Linear(dim_q, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim_k, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim_v, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # q: [B, N, Cq]
        # k: [B, M, Ck]
        # v: [B, M, Cv]
        # return: [B, N, C]

        B, N, _ = q.shape
        M = k.shape[1]
        
        q = self.scale * self.to_q(q).reshape(B, N, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3) # [B, nh, N, C/nh]
        k = self.to_k(k).reshape(B, M, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3) # [B, nh, M, C/nh]
        v = self.to_v(v).reshape(B, M, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3) # [B, nh, M, C/nh]

        attn = q @ k.transpose(-2, -1) # [B, nh, N, M]

        attn = attn.softmax(dim=-1) # [B, nh, N, M]
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1) # [B, nh, N, M] @ [B, nh, M, C/nh] --> [B, nh, N, C/nh] --> [B, N, nh, C/nh] --> [B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffCrossAttention(CrossAttention):
    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, _ = q.shape
        M = k.shape[1]

        q = self.scale * self.to_q(q).reshape(B, N, self.num_heads, self.dim // self.num_heads) # [B, N, nh, C/nh]
        k = self.to_k(k).reshape(B, M, self.num_heads, self.dim // self.num_heads) # [B, M, nh, C/nh]
        v = self.to_v(v).reshape(B, M, self.num_heads, self.dim // self.num_heads) # [B, M, nh, C/nh]

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MVAttention(nn.Module):
    def __init__(
        self, 
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        groups: int = 32,
        eps: float = 1e-5,
        residual: bool = True,
        skip_scale: float = 1,
    ):
        super().__init__()

        self.residual = residual
        self.skip_scale = skip_scale

        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim, eps=eps, affine=True)
        self.attn = MemEffAttention(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop)

    def forward(self, x):
        # x: [B*V, C, H, W]
        B, C, H, W  = x.shape
        res = x
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1).reshape(B, -1, C )
        x = self.attn(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        if self.residual:
            x = (x + res) * self.skip_scale
        return x

class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resample: Literal['default', 'up', 'down'] = 'default',
        groups: int = 32,
        eps: float = 1e-5,
        skip_scale: float = 1, # multiplied to output
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_scale = skip_scale

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.act = F.silu

        self.resample = None
        if resample == 'up':
            self.resample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
        elif resample == 'down':
            self.resample = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.shortcut = nn.Identity()
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    
    def forward(self, x):
        res = x

        x = self.norm1(x)
        x = self.act(x)

        if self.resample:
            res = self.resample(res)
            x = self.resample(x)
        
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x = (x + self.shortcut(res)) * self.skip_scale

        return x

class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        downsample: bool = True,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
    ):
        super().__init__()
 
        nets = []
        attns = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            nets.append(ResnetBlock(in_channels, out_channels, skip_scale=skip_scale))
            if attention:
                attns.append(MVAttention(out_channels, attention_heads, skip_scale=skip_scale))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

        self.downsample = None
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        xs = []

        for attn, net in zip(self.attns, self.nets):
            x = net(x)
            if attn:
                x = attn(x)
            xs.append(x)

        if self.downsample:
            x = self.downsample(x)
            xs.append(x)
  
        return x, xs


class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
    ):
        super().__init__()

        nets = []
        attns = []
        # first layer
        nets.append(ResnetBlock(in_channels, in_channels, skip_scale=skip_scale))
        # more layers
        for i in range(num_layers):
            nets.append(ResnetBlock(in_channels, in_channels, skip_scale=skip_scale))
            if attention:
                attns.append(MVAttention(in_channels, attention_heads, skip_scale=skip_scale))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)
        
    def forward(self, x):
        x = self.nets[0](x)
        for attn, net in zip(self.attns, self.nets[1:]):
            if attn:
                x = attn(x)
            x = net(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_out_channels: int,
        out_channels: int,
        num_layers: int = 1,
        upsample: bool = True,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
        unet_blocks=True,
    ):
        super().__init__()

        nets = []
        attns = []
        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            cskip = prev_out_channels if (i == num_layers - 1) or unet_blocks is False else out_channels
            nets.append(ResnetBlock(cin + cskip, out_channels, skip_scale=skip_scale))
            if attention:
                attns.append(MVAttention(out_channels, attention_heads, skip_scale=skip_scale))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

        self.upsample = None
        if upsample:
            self.upsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.unet_blocks=unet_blocks
    def forward(self, x, xs=None):

        if self.unet_blocks:
            for attn, net in zip(self.attns, self.nets):
                res_x = xs[-1]
                xs = xs[:-1]
                x = torch.cat([x, res_x], dim=1)
                x = net(x)
                if attn:
                    x = attn(x)
                    
        else:
            for attn, net in zip(self.attns, self.nets):
                x = net(x)
                if attn:
                    x = attn(x)
            
            
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')
            x = self.upsample(x)

        return x


# it could be asymmetric!
class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        image_out_channels: int = 3,
        sdf_out_channels: int = 16,
        down_channels: Tuple[int, ...] = (64, 128, 256, 512, 768),
        down_attention: Tuple[bool, ...] = (False, False, False, True, True),
        mid_attention: bool = True,
        sdf_up_channels: Tuple[int, ...] = (768, 512,),
        sdf_up_attention: Tuple[bool, ...] = (True, True, False),
        image_up_channels: Tuple[int, ...] = (768, 512, 256, 128),
        image_up_attention: Tuple[bool, ...] = (True, True, False, False),
        layers_per_block: int = 2,
        skip_scale: float = np.sqrt(0.5),
    ):
        super().__init__()

        # first
        self.conv_in = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, stride=1, padding=1)

        # down
        down_blocks = []
        cout = down_channels[0]
        for i in range(len(down_channels)):
            cin = cout
            cout = down_channels[i]

            down_blocks.append(DownBlock(
                cin, cout, 
                num_layers=layers_per_block, 
                downsample=(i != len(down_channels) - 1), # not final layer
                attention=down_attention[i],
                skip_scale=skip_scale,
            ))
        self.down_blocks = nn.ModuleList(down_blocks)

        # mid
        self.mid_block = MidBlock(down_channels[-1], attention=mid_attention, skip_scale=skip_scale)

        # up
        up_blocks = []
        cout = image_up_channels[0]
        for i in range(len(image_up_channels)):
            cin = cout
            cout = image_up_channels[i]
            cskip = down_channels[max(-2 - i, -len(down_channels))] # for assymetric

            up_blocks.append(UpBlock(
                cin, cskip, cout, 
                num_layers=layers_per_block + 1, # one more layer for up
                upsample=(i != len(image_up_channels) - 1), # not final layer
                attention=image_up_attention[i],
                skip_scale=skip_scale,
            ))
        self.image_up_blocks = nn.ModuleList(up_blocks)

        # last
        self.norm_image_out = nn.GroupNorm(num_channels=image_up_channels[-1], num_groups=32, eps=1e-5)
        self.conv_image_out = nn.Conv2d(image_up_channels[-1], image_out_channels, kernel_size=3, stride=1, padding=1)
        
        
        # up
        up_blocks = []
        cout = sdf_up_channels[0]
        for i in range(len(sdf_up_channels)):
            cin = cout
            cout = sdf_up_channels[i]
            cskip = down_channels[max(-2 - i, -len(down_channels))] # for assymetric

            up_blocks.append(UpBlock(
                cin, cskip, cout, 
                num_layers=layers_per_block + 1, # one more layer for up
                upsample=(i != len(sdf_up_channels) - 1), # not final layer
                attention=sdf_up_attention[i],
                skip_scale=skip_scale,
            ))
        self.sdf_up_blocks = nn.ModuleList(up_blocks)

        # last
        self.norm_sdf_out = nn.GroupNorm(num_channels=sdf_up_channels[-1], num_groups=32, eps=1e-5)
        self.conv_sdf_out = nn.Conv2d(sdf_up_channels[-1], sdf_out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x, image_only=False, sdf_only=False):
        # x: [B, Cin, H, W]
        
        assert not (image_only and sdf_only)

        # first
        x = self.conv_in(x)
        
        # down
        xss_image, xss_sdf = [x], [x]
        for block in self.down_blocks:
            x, xs = block(x)
            xss_image.extend(xs)
            xss_sdf.extend(xs)
        
        # mid
        x = self.mid_block(x)

        # up
        image, sdf = x, x
        if not sdf_only:
            for block in self.image_up_blocks:
                xs = xss_image[-len(block.nets):]
                xss_image = xss_image[:-len(block.nets)]
                image = block(image, xs)

            # last
            image = self.norm_image_out(image)
            image = F.silu(image)
            image = self.conv_image_out(image) # [B, Cout, H', W']
            if image_only:
                return image, 0, x.flatten(start_dim=2).permute(0, 2, 1)
        
        for block in self.sdf_up_blocks:
            xs = xss_sdf[-len(block.nets):]
            xss_sdf = xss_sdf[:-len(block.nets)]
            sdf = block(sdf, xs)
    
        sdf = self.norm_sdf_out(sdf)
        sdf = F.silu(sdf)
        sdf = self.conv_sdf_out(sdf) # [B, Cout, H', W']
        if sdf_only:
            return 0, sdf, x.flatten(start_dim=2).permute(0, 2, 1)

        return image, sdf, x.flatten(start_dim=2).permute(0, 2, 1)



# it could be asymmetric!
class UNetSuperResolution(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        image_out_channels: int = 3,
        down_channels: Tuple[int, ...] = (64, 128, 256, 512),
        down_attention: Tuple[bool, ...] = (False, False, False, True, True),
        mid_attention: bool = True,
        upsampling_channels: Tuple[int, ...] = (64, 32),
        image_up_channels: Tuple[int, ...] = (512, 256, 128, 64),
        image_up_attention: Tuple[bool, ...] = (True, True, False, False),
        layers_per_block: int = 2,
        skip_scale: float = np.sqrt(0.5),
        feature_channel=768,
    ):
        super().__init__()

        # first
        self.conv_in = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, stride=1, padding=1)
        self.feature_down = nn.Conv2d(feature_channel, 256, kernel_size=3, stride=1, padding=1)

        # down
        down_blocks = []
        cout = down_channels[0]
        for i in range(len(down_channels)):
            cin = cout
            cout = down_channels[i]

            down_blocks.append(DownBlock(
                cin, cout, 
                num_layers=layers_per_block, 
                downsample=(i != len(down_channels) - 1), # not final layer
                attention=down_attention[i],
                skip_scale=skip_scale,
            ))
        self.down_blocks = nn.ModuleList(down_blocks)

        # mid
        self.mid_block = MidBlock(down_channels[-1]+256, attention=mid_attention, skip_scale=skip_scale)

        # up
        up_blocks = []
        cout = image_up_channels[0]+256
        for i in range(len(image_up_channels)):
            cin = cout
            cout = image_up_channels[i]
            cskip = down_channels[max(-2 - i, -len(down_channels))] # for assymetric

            up_blocks.append(UpBlock(
                cin, cskip, cout, 
                num_layers=layers_per_block + 1, # one more layer for up
                upsample=(i != len(image_up_channels) - 1), # not final layer
                attention=image_up_attention[i],
                skip_scale=skip_scale,
            ))
        self.image_up_blocks = nn.ModuleList(up_blocks)
        upsampling_block = []
        
        for ch in upsampling_channels:
            cin = cout
            cout = ch
            
            upsampling_block.append(UpBlock(
                cin, 0, cout, 
                num_layers=layers_per_block + 1, # one more layer for up
                upsample=(i != len(upsampling_channels) - 1), # not final layer
                attention=False,
                skip_scale=skip_scale,
                unet_blocks=False,
            ))
        self.upsampling_blocks = nn.ModuleList(upsampling_block)
        
        # last
        self.norm_image_out = nn.GroupNorm(num_channels=upsampling_channels[-1], num_groups=32, eps=1e-5)
        self.conv_image_out = nn.Conv2d(upsampling_channels[-1], image_out_channels, kernel_size=3, stride=1, padding=1)
        


    def forward(self, x, m_features):
        # x: [B, Cin, H, W]

        # first
        x = self.conv_in(x)
        
        # down
        xss_image, xss_sdf = [x], [x]
        for block in self.down_blocks:
            x, xs = block(x)
            xss_image.extend(xs)
            xss_sdf.extend(xs)
        
        # mid
        image = self.mid_block(torch.cat([x, self.feature_down(m_features)], 1))
        
        # up
        for block in self.image_up_blocks:
            xs = xss_image[-len(block.nets):]
            xss_image = xss_image[:-len(block.nets)]
            image = block(image, xs)
            
        for block in self.upsampling_blocks:
            image = block(image)

        # last
        image = self.norm_image_out(image)
        image = F.silu(image)
        image = self.conv_image_out(image) # [B, Cout, H', W']
        return image
    
# def UNetSuperResolution
    
# model = UNetSuperResolution()
# # model = UNet()
# print(sum(p.numel() for p in model.parameters()))
# x = model(torch.rand(1, 3, 64, 64), torch.rand(1, 768, 8, 8))
# print(x.shape)