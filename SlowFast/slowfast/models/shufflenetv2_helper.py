#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FILE NAME = ''
AUTHOR = 'weidafeng'
DATE = '2020/7/14'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1, 1, 1),
                  bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )


def channel_shuffle(x, groups):
    '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
    batchsize, num_channels, depth, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, depth, height, width)
    # permute
    x = x.permute(0, 2, 1, 3, 4, 5).contiguous()
    # flatten
    x = x.view(batchsize, num_channels, depth, height, width)
    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.stride == 1:
            # assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv3d(oup_inc, oup_inc, 3, (1, stride, stride), 1,
                          groups=oup_inc,
                          bias=False),
                nn.BatchNorm3d(oup_inc),
                # pw-linear
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv3d(inp, inp, 3, (1, stride, stride), 1, groups=inp,
                          bias=False),
                nn.BatchNorm3d(inp),
                # pw-linear
                nn.Conv3d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv3d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv3d(oup_inc, oup_inc, 3, (1, stride, stride), 1,
                          groups=oup_inc,
                          bias=False),
                nn.BatchNorm3d(oup_inc),
                # pw-linear
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if self.stride == 1:
            x1 = x[:, :(x.shape[1] // 2), :, :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif self.stride == 2:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=400, sample_size=224, width_mult=1.):
        super(ShuffleNetV2, self).__init__()
        assert sample_size % 16 == 0

        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.25:
            self.stage_out_channels = [-1, 24, 32, 64, 128, 1024]
        elif width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(width_mult))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, stride=(1, 2, 2))
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)

        self.features = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                stride = 2 if i == 0 else 1
                self.features.append(
                    InvertedResidual(input_channel, output_channel, stride))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building last several layers
        self.conv_last = conv_1x1x1_bn(input_channel,
                                       self.stage_out_channels[-1])

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.stage_out_channels[-1], num_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.features(out)
        out = self.conv_last(out)
        out = F.avg_pool3d(out, out.data.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class ShuffleNetV2_Inverted_Residual_Block(nn.Module):

    def __init__(self, input_channel, idxstage, stage_out_channels):
        super(ShuffleNetV2_Inverted_Residual_Block, self).__init__()

        self.stage_repeats = [4, 8, 4]
        # self.stage_out_channels = stage_out_channels

        self.features = []

        # building inverted residual blocks
        numrepeat = self.stage_repeats[idxstage]
        output_channel = stage_out_channels[idxstage + 2]
        for i in range(numrepeat):
            stride = 2 if i == 0 else 1
            self.features.append(
                InvertedResidual(input_channel, output_channel, stride))
            input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
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


class ShuffleNetV2_Stage(nn.Module):

    def __init__(self,
                 input_channel,
                 idxstage,
                 slow_stage_out_channels,
                 fast_stage_out_channels,
                 ):
        '''
        :param input_channel:
        :param idxstage: stage， [0, 1, 2]
        :param slow_stage_out_channels:
        :param fast_stage_out_channels:
        '''
        super(ShuffleNetV2_Stage, self).__init__()

        self.features = []
        self.slow_stage_out_channels = slow_stage_out_channels
        self.fast_stage_out_channels = fast_stage_out_channels
        self.idxstage = idxstage

        self.num_pathways = len(input_channel)
        for pathway in range(self.num_pathways):
            if pathway == 0:
                res_block = ShuffleNetV2_Inverted_Residual_Block(
                    input_channel[pathway],
                    idxstage=self.idxstage,
                    stage_out_channels=self.slow_stage_out_channels
                )
                self.add_module("pathway{}_channel_{}".format(
                    pathway,
                    slow_stage_out_channels[idxstage + 2]),
                    res_block)
            elif pathway == 1:
                res_block = ShuffleNetV2_Inverted_Residual_Block(
                    input_channel[pathway],
                    idxstage=self.idxstage,
                    stage_out_channels=self.fast_stage_out_channels
                )
                self.add_module("pathway{}_channel_{}".format(
                    pathway,
                    fast_stage_out_channels[idxstage + 2]),
                    res_block)
            self._initialize_weights()

    def forward(self, inputs):
        output = []
        for pathway in range(self.num_pathways):
            x = inputs[pathway]
            if pathway == 0:
                m = getattr(self, "pathway{}_channel_{}".format(
                    pathway,
                    self.slow_stage_out_channels[self.idxstage + 2]))
            elif pathway == 1:
                m = getattr(self, "pathway{}_channel_{}".format(
                    pathway,
                    self.fast_stage_out_channels[self.idxstage + 2]))
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
