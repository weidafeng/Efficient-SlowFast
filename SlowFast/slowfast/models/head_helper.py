#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn
from detectron2.layers import ROIAlign


class ResNetRoIHead(nn.Module):
    """
    ResNe(X)t RoI head.
    """

    def __init__(
            self,
            dim_in,
            num_classes,
            pool_size,
            resolution,
            scale_factor,
            dropout_rate=0.0,
            act_func="softmax",
            aligned=True,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetRoIHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            resolution (list): the list of spatial output size from the ROIAlign.
            scale_factor (list): the list of ratio to the input boxes by this
                number.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            aligned (bool): if False, use the legacy implementation. If True,
                align the results more perfectly.
        Note:
            Given a continuous coordinate c, its two neighboring pixel indices
            (in our pixel model) are computed by floor (c - 0.5) and ceil
            (c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal at
            continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect
            alignment (relative to our pixel model) when performing bilinear
            interpolation.
            With `aligned=True`, we first appropriately scale the ROI and then
            shift it by -0.5 prior to calling roi_align. This produces the
            correct neighbors; It makes negligible differences to the model's
            performance if ROIAlign is used together with conv layers.
        """
        super(ResNetRoIHead, self).__init__()
        assert (
                len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            temporal_pool = nn.AvgPool3d(
                [pool_size[pathway][0], 1, 1], stride=1
            )
            self.add_module("s{}_tpool".format(pathway), temporal_pool)

            roi_align = ROIAlign(
                resolution[pathway],
                spatial_scale=1.0 / scale_factor[pathway],
                sampling_ratio=0,
                aligned=aligned,
            )
            self.add_module("s{}_roi".format(pathway), roi_align)
            spatial_pool = nn.MaxPool2d(resolution[pathway], stride=1)
            self.add_module("s{}_spool".format(pathway), spatial_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs, bboxes):
        assert (
                len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            t_pool = getattr(self, "s{}_tpool".format(pathway))
            out = t_pool(inputs[pathway])
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2)

            roi_align = getattr(self, "s{}_roi".format(pathway))
            out = roi_align(out, bboxes)

            s_pool = getattr(self, "s{}_spool".format(pathway))
            pool_out.append(s_pool(out))

        # B C H W.
        x = torch.cat(pool_out, 1)

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        x = self.projection(x)
        x = self.act(x)
        return x


class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
            self,
            dim_in,
            num_classes,
            pool_size,
            dropout_rate=0.0,
            act_func="softmax",
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetBasicHead, self).__init__()
        assert (
                len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs):
        assert (
                len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))

        x = torch.cat(pool_out, 1)

        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x

    def forward_debug(self, inputs):
        assert (
                len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
            print('x_{}.shape: {}'.format(pathway, pool_out[pathway].shape))

        x = torch.cat(pool_out, 1)
        print('x.shape: {}'.format(x.shape))

        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        print('x.shape after permute: {}'.format(x.shape))

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)
        print('x.shape after projection: {}'.format(x.shape))

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])
        print('x.shape after act: {}'.format(x.shape))

        x = x.view(x.shape[0], -1)
        print('x.shape after view: {}'.format(x.shape))
        '''
        x_0.shape: torch.Size([1, 2048, 1, 1, 1])
        x_1.shape: torch.Size([1, 256, 1, 1, 1])
        x.shape: torch.Size([1, 2304, 1, 1, 1])
        x.shape after permute: torch.Size([1, 1, 1, 1, 2304])
        x.shape after projection: torch.Size([1, 1, 1, 1, 3])
        x.shape after act: torch.Size([1, 3])
        x.shape after view: torch.Size([1, 3])
        '''
        return x



class ResNetBasicHead_SlowPath(nn.Module):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
            self,
            dim_in,
            num_classes,
            pool_size,
            dropout_rate=0.0,
            act_func="softmax",
            slow_or_fast=None,  # wdf add, None表示使用2个分支，'slow' or 'fast'表示仅使用一个分支预测
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetBasicHead_SlowPath, self).__init__()
        assert (
                len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.slow_or_fast = slow_or_fast
        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        if slow_or_fast is None:
            total_dim = sum(dim_in)
        elif slow_or_fast == 'slow':
            total_dim = dim_in[0]
        elif slow_or_fast == 'fast':
            total_dim = dim_in[1]
        else:
            assert '[wdf warning]: only support None, slow or fast!'
        # 仅使用slow分支做预测  # wdf-0826 add
        self.projection = nn.Linear(total_dim, num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs):
        assert (
                len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []

        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))

        # 是否只使用一个分支
        if self.slow_or_fast == 'slow':
            pool_out = pool_out[0]
        elif self.slow_or_fast == 'fast':
            pool_out = pool_out[1]

        x = torch.cat(pool_out, 1)

        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x

    def forward_debug(self, inputs):
        assert (
                len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
            print('x_{}.shape: {}'.format(pathway, pool_out[pathway].shape))

        x = torch.cat(pool_out, 1)
        print('x.shape: {}'.format(x.shape))

        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        print('x.shape after permute: {}'.format(x.shape))

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)
        print('x.shape after projection: {}'.format(x.shape))

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])
        print('x.shape after act: {}'.format(x.shape))

        x = x.view(x.shape[0], -1)
        print('x.shape after view: {}'.format(x.shape))
        '''
        x_0.shape: torch.Size([1, 2048, 1, 1, 1])
        x_1.shape: torch.Size([1, 256, 1, 1, 1])
        x.shape: torch.Size([1, 2304, 1, 1, 1])
        x.shape after permute: torch.Size([1, 1, 1, 1, 2304])
        x.shape after projection: torch.Size([1, 1, 1, 1, 3])
        x.shape after act: torch.Size([1, 3])
        x.shape after view: torch.Size([1, 3])
        '''
        return x



# --------------------------------------------------------
# wdf add for MobilenetV2
import torch.nn.functional as F


# 最后一个1*1*1卷积
def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


class MobileNetV2BasicHead(nn.Module):
    def __init__(self, input_channel, last_channel, num_classes, dropout_rate,
                 act_func="softmax"):
        super(MobileNetV2BasicHead, self).__init__()
        # 构建最后一个stage，以及最后的分类head
        self.num_pathways = len(input_channel)

        for pathway in range(self.num_pathways):
            # building last several layers
            features = conv_1x1x1_bn(input_channel[pathway],
                                     last_channel[pathway])

            self.add_module("pathway{}_conv1x1x1".format(pathway), features)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(sum(last_channel), num_classes, bias=True),
        )

    def forward(self, inputs):
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_conv1x1x1".format(pathway))
            # print("before conv : ", inputs[pathway].shape)
            x = m(inputs[pathway])
            # print("after conv : ", x.shape)
            x = F.avg_pool3d(x, x.data.size()[-3:])
            # print("after pooling : ", x.shape)
            pool_out.append(x)

        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))

        x = self.classifier(x)

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)

        return x


# ###############################################################
# wdf add for ShuffleNetV2, 最后一个stage，以及最后的分类head
def conv_1x1x1_bn_of_shufflenetv2(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )


class ShuffleNetV2BasicHead(nn.Module):
    def __init__(self, input_channel, last_channel, num_classes, dropout_rate,
                 act_func="softmax"):
        super(ShuffleNetV2BasicHead, self).__init__()
        # 构建最后一个stage，以及最后的分类head
        self.num_pathways = len(input_channel)

        for pathway in range(self.num_pathways):
            features = []
            features.append(
                conv_1x1x1_bn_of_shufflenetv2(input_channel[pathway],
                                              last_channel[pathway]))
            # make it nn.Sequential
            features = nn.Sequential(*features)
            self.add_module("pathway{}_conv1x1x1".format(pathway), features)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(sum(last_channel), num_classes, bias=True),
        )

    def forward(self, inputs):
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_conv1x1x1".format(pathway))
            # print("before conv : ", inputs[pathway].shape)
            x = m(inputs[pathway])
            # print("after conv : ", x.shape)
            x = F.avg_pool3d(x, x.data.size()[-3:])
            # print("after pooling : ", x.shape)
            pool_out.append(x)
            # print('x_{}.shape: {}'.format(pathway, x.shape))

        x = torch.cat(pool_out, 1)
        # print('x.shape: {}'.format( x.shape))

        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # print('x.shape after permute: {}'.format( x.shape))

        x = self.classifier(x)
        # print('x.shape after classifier: {}'.format( x.shape))

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        # print('x.shape after view: {}'.format( x.shape))

        return x


# ###############################################################
# wdf add for ShuffleNet, 最后一个stage，以及最后的分类head
class ShuffleNetBasicHead(nn.Module):
    def __init__(self, input_channel, num_classes, dropout_rate,
                 act_func="softmax"):
        super(ShuffleNetBasicHead, self).__init__()
        # 构建最后一个stage，以及最后的分类head
        self.num_pathways = len(input_channel)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(sum(input_channel), num_classes, bias=True),
        )

    def forward(self, inputs):
        pool_out = []
        for pathway in range(self.num_pathways):
            x = inputs[pathway]
            # print("after conv : ", x.shape)
            x = F.avg_pool3d(x, x.data.size()[-3:])
            # print("after pooling : ", x.shape)
            pool_out.append(x)
            # print('x_{}.shape: {}'.format(pathway, x.shape))

        x = torch.cat(pool_out, 1)
        # print('x.shape: {}'.format( x.shape))

        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # print('x.shape after permute: {}'.format( x.shape))

        x = self.classifier(x)
        # print('x.shape after classifier: {}'.format( x.shape))

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        # print('x.shape after view: {}'.format( x.shape))

        return x


# ###############################################################
# wdf add for GhostNet, 最后一个stage，以及最后的分类head
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


class GhostNetBasicHead(nn.Module):
    def __init__(self, input_channel, mid_channel, output_channel, num_classes,
                 dropout_rate,
                 act_func="softmax"):
        super(GhostNetBasicHead, self).__init__()
        # 构建最后一个stage，以及最后的分类head
        self.num_pathways = len(input_channel)
        self.input_channel = input_channel
        self.mid_channel = mid_channel
        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        # stage 5 最后还有一层卷积
        self.stage5_conv_slow = ConvBnAct(input_channel[0], mid_channel[0], 1)
        self.stage5_conv_fast = ConvBnAct(input_channel[1], mid_channel[1], 1)

        # 全局池化之后还有一层卷积
        self.conv_head_slow = nn.Conv3d(mid_channel[0], output_channel[0], 1, 1,
                                        0, bias=True)
        self.conv_head_fast = nn.Conv3d(mid_channel[1], output_channel[1], 1, 1,
                                        0, bias=True)
        self.act = nn.ReLU(inplace=True)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(sum(output_channel), num_classes, bias=True),
        )

    def forward(self, inputs):
        pool_out = []
        for pathway in range(self.num_pathways):
            if pathway == 0:
                x = inputs[pathway]
                x = self.stage5_conv_slow(x)
                # print("after conv : ", x.shape)
                x = F.avg_pool3d(x, x.data.size()[-3:])
                # print("after pooling : ", x.shape)
                x = self.conv_head_slow(x)
            elif pathway == 1:
                x = inputs[pathway]
                x = self.stage5_conv_fast(x)
                # print("after conv : ", x.shape)
                x = F.avg_pool3d(x, x.data.size()[-3:])
                # print("after pooling : ", x.shape)
                x = self.conv_head_fast(x)

            x = self.act(x)
            pool_out.append(x)
            # print('x_{}.shape: {}'.format(pathway, x.shape))

        x = torch.cat(pool_out, 1)
        # print('x.shape: {}'.format( x.shape))

        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # print('x.shape after permute: {}'.format( x.shape))

        x = self.classifier(x)
        # print('x.shape after classifier: {}'.format( x.shape))

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        # print('x.shape after view: {}'.format( x.shape))
        return x
