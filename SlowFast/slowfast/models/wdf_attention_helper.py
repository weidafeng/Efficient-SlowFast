#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FILE NAME = ''
AUTHOR = 'weidafeng'
DATE = '2020/6/8'
"""

import torch
from torch import nn


class SpatialAttention(nn.Module):
    """ spatial temporal attention module"""
    # https://github.com/junfu1115/DANet/blob/master/encoding/nn/attention.py
    # modified from SAGAN
    def __init__(self, channel, reduction=8):
        super(SpatialAttention, self).__init__()
        self.input_channel = channel

        self.query_conv = nn.Conv3d(in_channels=self.input_channel,
                                    out_channels=self.input_channel // reduction,
                                    kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=self.input_channel,
                                  out_channels=self.input_channel // reduction,
                                  kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=self.input_channel,
                                    out_channels=self.input_channel,
                                    kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B, C, T, H, W)
            returns :
                out : attention value + input feature
                attention: B X (TXHxW) X (TXHxW)
        """
        m_batchsize, C, T, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1,
                                             T * width * height).permute(0, 2,
                                                                         1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, T * width * height)
        attention = self.softmax(torch.bmm(proj_query, proj_key))
        proj_value = self.value_conv(x).view(m_batchsize, -1,
                                             T * width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, T, height, width)

        out = self.gamma * out + x
        return out


class ECA(nn.Module):
    """Constructs a ECA module.
    CVPR 2020
    https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py

    3D version, by wdf 2020-07-07

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size

    """

    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, t, h, w]
        # b, c, t, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2)) \
            .transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


## Channel Attention (CA) Layer
# https://github.com/yulunzhang/RCAN/blob/master/RCAN_TrainCode/code/model/rcan.py
# https://github.com/yjn870/RCAN-pytorch/blob/master/model.py
class ChannelAttention(nn.Module):
    ''' channel attention module

    Self CPU time total: 12.109ms
    CUDA time total: 17.368ms
    '''

    def __init__(self, channel, reduction=16):
        '''
        :param channel:   input channel size, [B,C,T,H,W] 中的 C
        :param reduction:  bottle neck scale
        '''
        super(ChannelAttention, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # feature channel downscale and upscale --> channel weight
        inner_channel = channel // reduction if channel // reduction != 0 else 2
        self.conv_du = nn.Sequential(
            nn.Conv3d(channel, inner_channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(inner_channel, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y + x  # 0629 add residual path


##################### Small Block ###################################
# https://github.com/jackie840129/STE-NVAN/blob/master/net/resnet.py
class NonLocalBlock(nn.Module):

    def __init__(self, in_channels, inter_channels=None, sub_sample=False,
                 bn_layer=True, instance='soft'):
        super(NonLocalBlock, self).__init__()
        self.sub_sample = sub_sample
        self.instance = instance
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        conv_nd = nn.Conv3d
        max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
        bn = nn.BatchNorm3d
        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)

        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        if self.instance == 'soft':
            f_div_C = F.softmax(f, dim=-1)
        elif self.instance == 'dot':
            f_div_C = f / f.shape[1]
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


class Stripe_NonLocalBlock(nn.Module):

    def __init__(self, stripe, in_channels, inter_channels=None,
                 pool_type='mean', instance='soft'):
        super(Stripe_NonLocalBlock, self).__init__()
        self.instance = instance
        self.stripe = stripe
        self.in_channels = in_channels
        self.pool_type = pool_type
        if pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'mean':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'meanmax':
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.maxpool = nn.AdaptiveMaxPool2d(1)
            self.in_channels *= 2
        if inter_channels == None:
            self.inter_channels = in_channels // 2
        else:
            self.inter_channels = inter_channels
        self.g = nn.Conv3d(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv3d(in_channels=self.in_channels,
                               out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv3d(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        if pool_type == 'meanmax':
            self.in_channels //= 2
        self.W = nn.Sequential(
            nn.Conv3d(in_channels=self.inter_channels,
                      out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(self.in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

    def forward(self, x):
        # x.shape = (b,c,t,h,w)
        b, c, t, h, w = x.shape
        assert self.stripe * (h // self.stripe) == h
        if self.pool_type == 'meanmax':
            discri_a = self.avgpool(
                x.reshape(b * c * t, self.stripe, (h // self.stripe),
                          w)).reshape(b, c, t, self.stripe, 1)
            discri_m = self.maxpool(
                x.reshape(b * c * t, self.stripe, (h // self.stripe),
                          w)).reshape(b, c, t, self.stripe, 1)
            discri = torch.cat([discri_a, discri_m], dim=1)
        else:
            discri = self.pool(
                x.reshape(b * c * t, self.stripe, (h // self.stripe),
                          w)).reshape(b, c, t, self.stripe, 1)
        g = self.g(discri).reshape(b, self.inter_channels, -1)
        g = g.permute(0, 2, 1)
        theta = self.theta(discri).reshape(b, self.inter_channels, -1)
        theta = theta.permute(0, 2, 1)
        phi = self.phi(discri).reshape(b, self.inter_channels, -1)
        f = torch.matmul(theta, phi)
        if self.instance == 'soft':
            f_div_C = F.softmax(f, dim=-1)
        elif self.instance == 'dot':
            f_div_C = f / f.shape[1]
        y = torch.matmul(f_div_C, g)
        y = y.permute(0, 2, 1).contiguous()
        y = y.reshape(b, self.inter_channels, *discri.size()[2:])
        W_y = self.W(y)
        W_y = W_y.repeat(1, 1, 1, 1, h // self.stripe * w).reshape(b, c, t, h,
                                                                   w)
        z = W_y + x

        return z


# GCNet CVPR19
# https://github.com/xvjiarui/GCNet/blob/master/mmdet/ops/gcb/context_block.py

from mmcv.cnn import constant_init, kaiming_init


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


class ContextBlock3D(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio=1.,
                 pooling_type='att',
                 fusion_types=('channel_add',)):
        super(ContextBlock3D, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv3d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv3d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv3d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv3d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv3d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True
        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, temp, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, T * H * W]
            input_x = input_x.view(batch, channel, temp * height * width)
            # [N, 1, C, T * H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, T, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, T, H * W]
            context_mask = context_mask.view(batch, 1, temp * height * width)
            # [N, 1, T * H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, T * H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        # [N, C, 1, 1, 1]
        context = self.spatial_pool(x)
        out = x
        # print(x.shape, context.shape)
        # torch.Size([1, 32, 4, 56, 56]) torch.Size([1, 32, 1, 1, 1])

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out
