#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['ghost_net']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio,
                                      divisor)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv_reduce = nn.Conv3d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv3d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv3d(in_chs, out_chs, kernel_size, stride,
                              kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm3d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1,
                 relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv3d(inp, init_channels,
                      kernel_size=(1, kernel_size, kernel_size),
                      stride=(1, stride, stride),
                      padding=(0, kernel_size // 2, kernel_size // 2),
                      bias=False),
            nn.BatchNorm3d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv3d(init_channels, new_channels, kernel_size=dw_size,
                      stride=1, padding=dw_size // 2,
                      groups=init_channels, bias=False),
            nn.BatchNorm3d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, ...]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride
        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)
        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv3d(mid_chs, mid_chs,
                                     kernel_size=(
                                         1, dw_kernel_size, dw_kernel_size),
                                     stride=(1, stride, stride),
                                     padding=(0, (dw_kernel_size - 1) // 2,
                                              (dw_kernel_size - 1) // 2),
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm3d(mid_chs)
        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            # 空间降采样
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_chs, in_chs,
                          kernel_size=(1, dw_kernel_size, dw_kernel_size),
                          stride=(1, stride, stride),
                          padding=(0, (dw_kernel_size - 1) // 2,
                                   (dw_kernel_size - 1) // 2),
                          groups=in_chs,
                          bias=False),
                nn.BatchNorm3d(in_chs),
                nn.Conv3d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(out_chs),
            )

    def forward(self, x):
        residual = x
        # 1st ghost bottleneck
        x = self.ghost1(x)
        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)
        # 2nd ghost bottleneck
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x


class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * width, 2)
        self.conv_stem = nn.Conv3d(3, output_channel, kernel_size=3,
                                   stride=(1, 2, 2), padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(
                    block(input_channel,
                          hidden_channel,
                          output_channel,
                          dw_kernel_size=k,
                          stride=s,
                          se_ratio=se_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(
            nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel
        self.blocks = nn.Sequential(*stages)

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv_head = nn.Conv3d(input_channel, output_channel, 1, 1, 0,
                                   bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)


    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


def ghostnet(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s
        # k: kenel_size of depth_wise conv
        # t: hidden layer channel size
        # c: output layer channel size
        # SE: 0 means no SE block, otherwise means squeeze ratio of SE block
        # s: stride, only the first layer of each stage is 2(means down sampling)
        # stage1
        [[3, 16, 16, 0, 1]],
        # stage2
        [[3, 48, 24, 0, 2]],
        [[3, 72, 24, 0, 1]],
        # stage3
        [[5, 72, 40, 0.25, 2]],
        [[5, 120, 40, 0.25, 1]],
        # stage4
        [[3, 240, 80, 0, 2]],
        [[3, 200, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
         ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
         ]
    ]
    return GhostNet(cfgs, **kwargs)


# -------------------------- wdf add for ghost net slowfast --------------------
class GhostNet_Inverted_Residual_Block(nn.Module):
    def __init__(self, input_channel, cfg):

        super(GhostNet_Inverted_Residual_Block, self).__init__()

        # building inverted residual blocks
        layers = []
        for k, exp_size, c, se_ratio, s in cfg:
            output_channel = _make_divisible(c, 2)
            hidden_channel = _make_divisible(exp_size, 2)
            layers.append(
                GhostBottleneck(input_channel,
                      hidden_channel,
                      output_channel,
                      dw_kernel_size=k,
                      stride=s,
                      se_ratio=se_ratio))
            input_channel = output_channel

        self.features = nn.Sequential(*layers)

        self._initialize_weights()

    def forward(self, x):
        # print("Inverted stage: \n", self.features)
        x = self.features(x)
        # print("Inverted stage: ", x[0].shape, x[1].shape)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[
                    2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class GhostNet_Stage(nn.Module):
    def __init__(self,
                 input_channel,
                 slow_cfg,
                 fast_cfg,
                 ):

        super(GhostNet_Stage, self).__init__()

        self.features = []
        self.slow_cfg = slow_cfg
        self.fast_cfg = fast_cfg

        self.num_pathways = len(input_channel)
        for pathway in range(self.num_pathways):
            if pathway == 0:
                res_block = GhostNet_Inverted_Residual_Block(
                    input_channel=input_channel[pathway],
                    cfg=slow_cfg,
                )
                self.add_module("pathway{}_channel_{}".format(
                    pathway,
                    self.slow_cfg[-1][2]),
                    res_block)
            elif pathway == 1:
                res_block = GhostNet_Inverted_Residual_Block(
                    input_channel=input_channel[pathway],
                    cfg=fast_cfg,
                )
                self.add_module("pathway{}_channel_{}".format(
                    pathway,
                    self.fast_cfg[-1][2]),
                    res_block)
            self._initialize_weights()

    def forward(self, inputs):
        output = []
        for pathway in range(self.num_pathways):
            x = inputs[pathway]
            if pathway == 0:
                m = getattr(self, "pathway{}_channel_{}".format(
                    pathway,
                    self.slow_cfg[-1][2]))
            elif pathway == 1:
                m = getattr(self, "pathway{}_channel_{}".format(
                    pathway,
                    self.fast_cfg[-1][2]))
            x = m(x)
            output.append(x)
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[
                    2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
