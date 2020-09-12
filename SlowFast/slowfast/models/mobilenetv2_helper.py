#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''MobilenetV2 in PyTorch.

See the paper "MobileNetV2: Inverted Residuals and Linear Bottlenecks" for more details.
'''
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1, 1, 1),
                  bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == (1, 1, 1) and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileV2_Inverted_Residual_Block(nn.Module):

    def __init__(self, input_channel, interverted_residual_setting, width_mult, beta_inv=None):
        super(MobileV2_Inverted_Residual_Block, self).__init__()
        self.features = []
        # building inverted residual blocks
        if isinstance(interverted_residual_setting[0], list):
            for t, c, n, s in interverted_residual_setting:
                output_channel = int(c * width_mult) if beta_inv is None else int(c * width_mult // beta_inv)
                for i in range(n):
                    stride = s if i == 0 else (1, 1, 1)
                    self.features.append(
                        InvertedResidual(input_channel, output_channel, stride,
                                         expand_ratio=t))
                    input_channel = output_channel

        else:
            t, c, n, s = interverted_residual_setting
            output_channel = int(c * width_mult) if beta_inv is None else int(
                c * width_mult // beta_inv)
            for i in range(n):
                stride = s if i == 0 else (1, 1, 1)
                self.features.append(
                    InvertedResidual(input_channel, output_channel, stride,
                                     expand_ratio=t))
                input_channel = output_channel
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        # self._initialize_weights()

    def forward(self, x):
        # print("Inverted stage: \n", self.features)
        x = self.features(x)
        return x


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError(
            "Unsupported ft_portion: 'complete' or 'last_layer' expected")


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, sample_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        input_channel = 32
        last_channel = 1280

        slow_interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, (1, 1, 1)],
            [6, 24, 2, (1, 2, 2)],  # *
            [6, 32, 3, (1, 2, 2)],  # *
            [6, 64, 4, (1, 2, 2)],  # *
            [6, 96, 3, (1, 1, 1)],
            [6, 160, 3, (1, 2, 2)],  # *
            [6, 320, 1, (1, 1, 1)],
        ]


        # building first layer
        assert sample_size % 16 == 0.
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(
            last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, (1, 2, 2))]

        # building inverted residual blocks
        for t, c, n, s in slow_interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else (1, 1, 1)
                self.features.append(
                    InvertedResidual(input_channel, output_channel, stride,
                                     expand_ratio=t))
                input_channel = output_channel

        # building last several layers
        self.features.append(conv_1x1x1_bn(input_channel, self.last_channel))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool3d(x, x.data.size()[-3:])
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
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


class Stem(nn.Module):
    def __init__(self, input_channel=32, sample_size=224, width_mult=1.):
        super(Stem, self).__init__()
        assert sample_size % 16 == 0.
        input_channel = int(input_channel * width_mult)
        features = [conv_bn(3, input_channel, (1, 2, 2))]
        self.features = nn.Sequential(*features)

        self._initialize_weights()

    def forward(self, x):
        return self.features(x)

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



class ClassificationHead(nn.Module):
    def __init__(self, input_channel, last_channel, num_classes):
        super(ClassificationHead, self).__init__()
        # building last several layers
        features = []
        features.append(conv_1x1x1_bn(input_channel, last_channel))

        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool3d(x, x.data.size()[-3:])
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MobileNetV2_Stage(nn.Module):

    def __init__(self,
                 input_channel,
                 slow_residual_setting,
                 fast_residual_setting=None,
                 width_mult=1.,
                 beta_inv=4):
        super(MobileNetV2_Stage, self).__init__()
        # print(slow_residual_setting)
        assert (isinstance(slow_residual_setting[0], list) and
                isinstance(fast_residual_setting[0], list)) or (
                           isinstance(slow_residual_setting,
                                      list) and isinstance(
                       fast_residual_setting, list))

        self.slow_residual_setting = slow_residual_setting
        self.fast_residual_setting = fast_residual_setting

        self.width_mult = width_mult
        self.features = []

        self.num_pathways = len(input_channel)
        for pathway in range(self.num_pathways):
            # Construct the block.
            if pathway == 0:
                # print(self.slow_residual_setting)
                res_block = MobileV2_Inverted_Residual_Block(
                    input_channel[pathway],
                    self.slow_residual_setting,
                    self.width_mult,)
                self.add_module("pathway{}_channel_{}".format(
                    pathway,
                    self.slow_residual_setting[0][1]),
                    res_block,)
                self._initialize_weights()

            elif pathway == 1:
                # print(self.fast_residual_setting)
                res_block = MobileV2_Inverted_Residual_Block(
                    input_channel[pathway],
                    self.fast_residual_setting,
                    self.width_mult,
                    beta_inv=beta_inv)
                self.add_module("pathway{}_channel_{}".format(
                    pathway,
                    self.fast_residual_setting[0][1]),
                    res_block)
                self._initialize_weights()

            else:
                raise Exception('Only support 1 or 2 pathways')

    def forward(self, inputs):

        output = []
        for pathway in range(self.num_pathways):
            x = inputs[pathway]
            if pathway == 0:
                m = getattr(self, "pathway{}_channel_{}".format(
                    pathway,
                    self.slow_residual_setting[0][1]))
                x = m(x)
            elif pathway == 1:
                m = getattr(self, "pathway{}_channel_{}".format(
                    pathway,
                    self.fast_residual_setting[0][1]))
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



def get_model(**kwargs):
    """
    Returns the model.
    """
    model = MobileNetV2(**kwargs)
    return model
