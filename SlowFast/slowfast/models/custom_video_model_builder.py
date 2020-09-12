#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


"""
A More Flexible Video models.

Support models:
    1. Dual attention slowfast
    2. Efficient models:
        2.1 MobileNet V2
        2.2 ShuffleNet
        2.3 ShuffleNet v2
        2.4 GhostNet
All these models use the same dual attention module.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.batchnorm_helper import get_norm

from . import head_helper, resnet_helper, stem_helper
from .build import MODEL_REGISTRY

# attention module
from .wdf_attention_helper import (SpatialAttention, ChannelAttention,
                                   ECA, ContextBlock3D)

# shuffle net v2
from .shufflenetv2_helper import ShuffleNetV2_Stage
# shuffle net
from .shufflenet_helper import ShuffleNet_Stage
# ghost net
from .ghostnet_helper import GhostNet_Stage, _make_divisible
# mobile net v2
from .mobilenetv2_helper import MobileNetV2_Stage


class FuseFastAndSlow(nn.Module):
    """
    Conduct the operations with higher computational cost when the feature
    graph is smaller by optimizing the execution order.

    Fusion method：
        fast path: [B, C//BETA_INV,     T,          H, W], eg. [1,  64//4,  32,     112, 112]
        slow path: [B, C,               T//ALPHA,   H, W], eg. [1,  64,     32//4,  112, 112]

    Fusion result：
        new slow path: [B, C + C //BETA_INV,            T//ALPHA, H, W]
        new fast path: [B, C//BETA_INV + C//BETA_INV,   T,        H, W]
    """

    def __init__(
            self,
            dim_in,
            alpha,
            beta_inv,
            eps=1e-5,
            bn_mmt=0.1,
            inplace_relu=True,
            norm_module=nn.BatchNorm3d,
            reduction=1,
    ):
        """
        Args:
            dim_in (list(int)): the channel dimension of the input.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            beta_inv(int): the channel ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            reduction (int): reduction ratio in the attention model.
        """
        super(FuseFastAndSlow, self).__init__()

        # Fast to Slow
        # 1. T --> T / ALPHA
        self.downsample_t_of_fast = nn.MaxPool3d(kernel_size=(alpha, 1, 1),
                                                 stride=(alpha, 1, 1))
        # 2. channel attention
        self.attention_channel_f2s = ECA(
            dim_in[1],
            # reduction=1, # save all channels, do not reduct,
        )
        print('fusion layer dim input: ', dim_in)
        self.bn_f2s = norm_module(
            num_features=dim_in[1],
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu_f2s = nn.ReLU(inplace_relu)

        # Slow to Fast
        # 1. C --> C // BETA_INV
        self.downsample_c_of_slow = nn.Conv3d(
            dim_in[0],
            dim_in[0] // beta_inv,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            bias=False,
        )
        # 2. spatial attention
        self.attention_spatial_s2f = SpatialAttention(
            int(dim_in[0] // beta_inv),
            reduction=reduction)
        self.bn_s2f = norm_module(
            num_features=int(dim_in[0] // beta_inv),
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu_s2f = nn.ReLU(inplace_relu)
        # 3. T//ALPHA -->  T
        self.upsample_s2f = nn.Upsample(scale_factor=(alpha, 1, 1),
                                        mode='nearest')

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]

        # Fast to Slow
        # origin method, directly sampling (T)
        # fuse_from_fast = x_f[:, :, ::self.alpha, ...]
        # new method, temporally max pooling
        fuse_from_fast = self.downsample_t_of_fast(x_f)
        fuse_from_fast = self.attention_channel_f2s(fuse_from_fast)
        fuse_from_fast = self.bn_f2s(fuse_from_fast)
        fuse_from_fast = self.relu_f2s(fuse_from_fast)
        x_s_fuse = torch.cat([x_s, fuse_from_fast], 1)

        # Slow to Fast
        # origin method, directly sampling (C)
        # fuse_from_slow = x_s[:, ::self.beta_inv * self.fusion_ratio, ...]
        # new method, 1*1*1 conv
        fuse_from_slow = self.downsample_c_of_slow(x_s)
        fuse_from_slow = self.attention_spatial_s2f(fuse_from_slow)
        fuse_from_slow = self.bn_s2f(fuse_from_slow)
        fuse_from_slow = self.relu_s2f(fuse_from_slow)
        fuse_from_slow = self.upsample_s2f(fuse_from_slow)
        x_f_fuse = torch.cat([fuse_from_slow, x_f], 1)

        return [x_s_fuse, x_f_fuse]

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3),
                      18: (2, 2, 2, 2), 34: (3, 4, 6, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
}

_POOL1 = {
    "slowfast": [[1, 1, 1], [1, 1, 1]],
}



@MODEL_REGISTRY.register()
class SlowFastDualAttention(nn.Module):
    """
    Efficient Dual Attention SlowFast Networks for Video Action Recognition

    Dafeng Wei, Ye Tian, Liqing Wei, Hong Zhong, Siqian Chen, Shiliang Pu, Hongtao Lu
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFastDualAttention, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 2
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP  # resnet: 64
        dim_inner = num_groups * width_per_group

        out_dim_ratio = cfg.SLOWFAST.BETA_INV  # WDF-FIX

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        # s1: slow 3-->64, fast 3-->8(64/BETA)
        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group, width_per_group // cfg.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=self.norm_module,
        )

        self.s1_fuse = FuseFastAndSlow(
            dim_in=[
                width_per_group,
                width_per_group // cfg.SLOWFAST.BETA_INV,
            ],
            alpha=cfg.SLOWFAST.ALPHA,
            beta_inv=cfg.SLOWFAST.BETA_INV,
            norm_module=self.norm_module,
            reduction=1,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // cfg.SLOWFAST.BETA_INV + width_per_group // out_dim_ratio,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner, dim_inner // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        self.s2_fuse = FuseFastAndSlow(
            dim_in=[
                width_per_group * 4,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            alpha=cfg.SLOWFAST.ALPHA,
            beta_inv=cfg.SLOWFAST.BETA_INV,
            norm_module=self.norm_module,
            reduction=1,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV + width_per_group * 4 // out_dim_ratio,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )
        self.s3_fuse = FuseFastAndSlow(
            dim_in=[
                width_per_group * 8,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV
            ],
            alpha=cfg.SLOWFAST.ALPHA,
            beta_inv=cfg.SLOWFAST.BETA_INV,
            norm_module=self.norm_module,
            reduction=1,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV + width_per_group * 8 // out_dim_ratio,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )
        self.s4_fuse = FuseFastAndSlow(
            dim_in=[
                width_per_group * 16,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            alpha=cfg.SLOWFAST.ALPHA,
            beta_inv=cfg.SLOWFAST.BETA_INV,
            norm_module=self.norm_module,
            reduction=1,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV + width_per_group * 16 // out_dim_ratio,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if cfg.DETECTION.ENABLE:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        1,
                        1,
                    ],
                    [cfg.DATA.NUM_FRAMES // pool_size[1][0], 1, 1],
                ],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2] * 2,
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR] * 2,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ],
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[1][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[1][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[1][2],
                    ],
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x


@MODEL_REGISTRY.register()
class SlowFastShuffleNetV2(nn.Module):
    """
    Efficient Dual Attention SlowFast Networks for Video Action Recognition

    Dafeng Wei, Ye Tian, Liqing Wei, Hong Zhong, Siqian Chen, Shiliang Pu, Hongtao Lu

    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFastShuffleNetV2, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 2

        width_mult = cfg.SLOWFAST.WIDTH_MULTI
        if width_mult == 0.25:
            self.stage_out_channels = [-1, 24, 32, 64, 128, 1024]
        elif width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 240, 464, 1024]  # 232 -> 240
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 496, 976, 2048]  # 488 -> 496
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(width_mult))
        self.fast_stage_out_channels = [c // cfg.SLOWFAST.BETA_INV for c in
                                        self.stage_out_channels]
        self._construct_network(cfg)

        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """

        width_per_group = self.stage_out_channels[1]
        last_channel = self.stage_out_channels[-1]

        self.s1 = stem_helper.ShuffleNetV2_Model_Stem(
            input_channels=[
                width_per_group,
                width_per_group // cfg.SLOWFAST.BETA_INV
            ],
            sample_size=cfg.DATA.CROP_SIZE,
            width_mult=[
                cfg.SLOWFAST.WIDTH_MULTI,  # base width multi for slow path
                cfg.SLOWFAST.WIDTH_MULTI / cfg.SLOWFAST.BETA_INV
            ],
            img_dim=len(cfg.DATA.MEAN)
        )

        self.s1_fuse = FuseFastAndSlow(
            dim_in=[self.stage_out_channels[1],
                    self.fast_stage_out_channels[1]],
            alpha=cfg.SLOWFAST.ALPHA,
            beta_inv=cfg.SLOWFAST.BETA_INV,
            norm_module=self.norm_module,
        )

        self.s2 = ShuffleNetV2_Stage(
            input_channel=[
                self.stage_out_channels[1] + self.fast_stage_out_channels[1],
                self.fast_stage_out_channels[1] + self.stage_out_channels[
                    1] // cfg.SLOWFAST.BETA_INV
            ],
            idxstage=0,
            slow_stage_out_channels=self.stage_out_channels,
            fast_stage_out_channels=self.fast_stage_out_channels,
        )

        self.s2_fuse = FuseFastAndSlow(
            dim_in=[self.stage_out_channels[2],
                    self.fast_stage_out_channels[2]],
            alpha=cfg.SLOWFAST.ALPHA,
            beta_inv=cfg.SLOWFAST.BETA_INV,
            norm_module=self.norm_module,
        )

        self.s3 = ShuffleNetV2_Stage(
            input_channel=[
                self.stage_out_channels[2] + self.fast_stage_out_channels[2],
                self.fast_stage_out_channels[2] + self.stage_out_channels[
                    2] // cfg.SLOWFAST.BETA_INV
            ],
            idxstage=1,
            slow_stage_out_channels=self.stage_out_channels,
            fast_stage_out_channels=self.fast_stage_out_channels,
        )

        self.s3_fuse = FuseFastAndSlow(
            dim_in=[self.stage_out_channels[3],
                    self.fast_stage_out_channels[3]],
            alpha=cfg.SLOWFAST.ALPHA,
            beta_inv=cfg.SLOWFAST.BETA_INV,
            norm_module=self.norm_module,
        )

        self.s4 = ShuffleNetV2_Stage(
            input_channel=[
                self.stage_out_channels[3] + self.fast_stage_out_channels[3],
                self.fast_stage_out_channels[3] + self.stage_out_channels[
                    3] // cfg.SLOWFAST.BETA_INV
            ],
            idxstage=2,
            slow_stage_out_channels=self.stage_out_channels,
            fast_stage_out_channels=self.fast_stage_out_channels,
        )

        self.s4_fuse = FuseFastAndSlow(
            dim_in=[self.stage_out_channels[4],
                    self.fast_stage_out_channels[4]],
            alpha=cfg.SLOWFAST.ALPHA,
            beta_inv=cfg.SLOWFAST.BETA_INV,
            norm_module=self.norm_module,
        )

        # building last several layers
        self.head = head_helper.ShuffleNetV2BasicHead(
            input_channel=[
                self.stage_out_channels[4] + self.fast_stage_out_channels[4],
                self.fast_stage_out_channels[4] + self.stage_out_channels[
                    4] // cfg.SLOWFAST.BETA_INV
            ],
            last_channel=[
                self.stage_out_channels[-1],
                self.fast_stage_out_channels[-1]
            ],
            num_classes=cfg.MODEL.NUM_CLASSES,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        # for pathway in range(self.num_pathways):
        #     pool = getattr(self, "pathway{}_pool".format(pathway))
        #     x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        # x = self.s5(x)
        # print('s5: ', x[0].shape, x[1].shape)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x


@MODEL_REGISTRY.register()
class SlowFastShuffleNet(nn.Module):
    """
    Efficient Dual Attention SlowFast Networks for Video Action Recognition

    Dafeng Wei, Ye Tian, Liqing Wei, Hong Zhong, Siqian Chen, Shiliang Pu, Hongtao Lu
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFastShuffleNet, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 2

        width_mult = cfg.SLOWFAST.WIDTH_MULTI
        groups = cfg.SLOWFAST.GROUPS
        self.num_blocks = [4, 8, 4]
        self.groups = groups

        if groups == 1:
            out_planes = [24, 144, 288, 567]
        elif groups == 2:
            out_planes = [24, 200, 400, 800]
        elif groups == 3:
            out_planes = [24, 240, 480, 960]
        elif groups == 4:
            out_planes = [24, 272, 544, 1088]
        elif groups == 8:
            out_planes = [24, 384, 768, 1536]
        else:
            raise ValueError(
                """{} groups is not supported for
                   1x1 Grouped Convolutions""".format(groups))

        self.stage_out_channels = [int(i * width_mult) for i in out_planes]
        self.fast_stage_out_channels = [c // cfg.SLOWFAST.BETA_INV for c in
                                        self.stage_out_channels]

        self.in_planes = out_planes[0]

        self._construct_network(cfg)

        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """

        self.s1 = stem_helper.ShuffleNet_Model_Stem(
            input_channels=[
                self.stage_out_channels[0],
                self.fast_stage_out_channels[0]
            ],
            sample_size=cfg.DATA.CROP_SIZE,
            img_dim=len(cfg.DATA.MEAN)
        )

        self.s1_fuse = FuseFastAndSlow(
            dim_in=[self.stage_out_channels[0],
                    self.fast_stage_out_channels[0]],
            alpha=cfg.SLOWFAST.ALPHA,
            beta_inv=cfg.SLOWFAST.BETA_INV,
            norm_module=self.norm_module,
        )

        self.s2 = ShuffleNet_Stage(
            input_channel=[
                self.stage_out_channels[0] + self.fast_stage_out_channels[0],
                self.fast_stage_out_channels[0] + self.stage_out_channels[
                    0] // cfg.SLOWFAST.BETA_INV
            ],
            slow_stage_out_channels=self.stage_out_channels[1],
            fast_stage_out_channels=self.fast_stage_out_channels[1],
            num_block=self.num_blocks[0],
            group=cfg.SLOWFAST.GROUPS,
        )

        self.s2_fuse = FuseFastAndSlow(
            dim_in=[self.stage_out_channels[1],
                    self.fast_stage_out_channels[1]],
            alpha=cfg.SLOWFAST.ALPHA,
            beta_inv=cfg.SLOWFAST.BETA_INV,
            norm_module=self.norm_module,
        )

        self.s3 = ShuffleNet_Stage(
            input_channel=[
                self.stage_out_channels[1] + self.fast_stage_out_channels[1],
                self.fast_stage_out_channels[1] + self.stage_out_channels[
                    1] // cfg.SLOWFAST.BETA_INV
            ],
            slow_stage_out_channels=self.stage_out_channels[2],
            fast_stage_out_channels=self.fast_stage_out_channels[2],
            num_block=self.num_blocks[1],
            group=cfg.SLOWFAST.GROUPS,
        )

        self.s3_fuse = FuseFastAndSlow(
            dim_in=[self.stage_out_channels[2],
                    self.fast_stage_out_channels[2]],
            alpha=cfg.SLOWFAST.ALPHA,
            beta_inv=cfg.SLOWFAST.BETA_INV,
            norm_module=self.norm_module,
        )

        self.s4 = ShuffleNet_Stage(
            input_channel=[
                self.stage_out_channels[2] + self.fast_stage_out_channels[2],
                self.fast_stage_out_channels[2] + self.stage_out_channels[
                    2] // cfg.SLOWFAST.BETA_INV
            ],
            slow_stage_out_channels=self.stage_out_channels[3],
            fast_stage_out_channels=self.fast_stage_out_channels[3],
            num_block=self.num_blocks[2],
            group=cfg.SLOWFAST.GROUPS,
        )

        self.s4_fuse = FuseFastAndSlow(
            dim_in=[self.stage_out_channels[3],
                    self.fast_stage_out_channels[3]],
            alpha=cfg.SLOWFAST.ALPHA,
            beta_inv=cfg.SLOWFAST.BETA_INV,
            norm_module=self.norm_module,
        )

        # building last several layers
        self.head = head_helper.ShuffleNetBasicHead(
            input_channel=[
                self.stage_out_channels[3] + self.fast_stage_out_channels[3],
                self.fast_stage_out_channels[3] + self.stage_out_channels[
                    3] // cfg.SLOWFAST.BETA_INV
            ],
            num_classes=cfg.MODEL.NUM_CLASSES,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        # for pathway in range(self.num_pathways):
        #     pool = getattr(self, "pathway{}_pool".format(pathway))
        #     x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        # x = self.s5(x)
        # print('s5: ', x[0].shape, x[1].shape)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x


@MODEL_REGISTRY.register()
class SlowFastGhostNet(nn.Module):
    """
    Efficient Dual Attention SlowFast Networks for Video Action Recognition

    Dafeng Wei, Ye Tian, Liqing Wei, Hong Zhong, Siqian Chen, Shiliang Pu, Hongtao Lu
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFastGhostNet, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 2

        self.num_blocks = [4, 8, 4]
        cfgs_of_ghost_stages = [
            # k, t, c, SE, s
            # k: kenel_size of depth_wise conv
            # t: hidden layer channel size
            # c: output layer channel size
            # SE: 0 means no SE block, otherwise means squeeze ratio of SE block
            # s: stride, only the first layer of each stage is 2(means down sampling)
            # stage1
            [[3, 16, 16, 0, 1]],
            # stage2
            [[3, 48, 24, 0, 2],
             [3, 72, 24, 0, 1]],
            # stage3
            [[5, 72, 40, 0.25, 2],
             [5, 120, 40, 0.25, 1]],
            # stage4
            [[3, 240, 80, 0, 2],
             [3, 200, 80, 0, 1],
             [3, 184, 80, 0, 1],
             [3, 184, 80, 0, 1],
             [3, 480, 112, 0.25, 1],
             [3, 672, 112, 0.25, 1]
             ],
            # stage5
            [[5, 672, 160, 0.25, 2],
             [5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1],
             [5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1]
             ]
        ]

        self.fast_cfgs = []
        self.slow_cfgs = []
        for cfg_stage in cfgs_of_ghost_stages:
            fast_tmp = []
            slow_tmp = []
            for c in cfg_stage:
                fast_tmp.append(
                    [c[0],
                     _make_divisible(c[
                             1] * cfg.SLOWFAST.WIDTH_MULTI // cfg.SLOWFAST.BETA_INV, 4),
                     _make_divisible(c[
                             2] * cfg.SLOWFAST.WIDTH_MULTI // cfg.SLOWFAST.BETA_INV, 4),
                     c[3], c[4]])
                slow_tmp.append(
                    [c[0], _make_divisible(c[1] * cfg.SLOWFAST.WIDTH_MULTI, 4),
                     _make_divisible(c[2] * cfg.SLOWFAST.WIDTH_MULTI, 4), c[3], c[4]])
            self.fast_cfgs.append(fast_tmp)
            self.slow_cfgs.append(slow_tmp)
        print(self.slow_cfgs)
        print(self.fast_cfgs)
        self._construct_network(cfg)

        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """

        channel_witdh = 16
        channel_witdhs = [
            _make_divisible(channel_witdh * cfg.SLOWFAST.WIDTH_MULTI, 4),
            _make_divisible(
                channel_witdh * cfg.SLOWFAST.WIDTH_MULTI // cfg.SLOWFAST.BETA_INV,
                4)
        ]

        output_channel = 1280
        output_channels = [
            int(output_channel * cfg.SLOWFAST.WIDTH_MULTI),
            int(
                output_channel * cfg.SLOWFAST.WIDTH_MULTI // cfg.SLOWFAST.BETA_INV),
        ]

        # first several convs
        self.s0 = stem_helper.GhostNet_Model_Stem(
            input_channels=[
                channel_witdhs[0],
                channel_witdhs[1]
            ],
            sample_size=cfg.DATA.CROP_SIZE,
            img_dim=len(cfg.DATA.MEAN)
        )

        # stage 1
        self.s1 = GhostNet_Stage(
            input_channel=[
                channel_witdhs[0],
                channel_witdhs[1]
            ],
            slow_cfg=self.slow_cfgs[0],
            fast_cfg=self.fast_cfgs[0],
        )

        self.s1_fuse = FuseFastAndSlow(
            dim_in=[self.slow_cfgs[0][-1][2],
                    self.fast_cfgs[0][-1][2]],
            alpha=cfg.SLOWFAST.ALPHA,
            beta_inv=cfg.SLOWFAST.BETA_INV,
            norm_module=self.norm_module,
        )
        # stage 2
        self.s2 = GhostNet_Stage(
            input_channel=[
                self.slow_cfgs[0][0][2] + self.fast_cfgs[0][-1][2],
                self.fast_cfgs[0][0][2] + self.slow_cfgs[0][-1][
                    2] // cfg.SLOWFAST.BETA_INV
            ],
            slow_cfg=self.slow_cfgs[1],
            fast_cfg=self.fast_cfgs[1],
        )

        self.s2_fuse = FuseFastAndSlow(
            dim_in=[self.slow_cfgs[1][-1][2],
                    self.fast_cfgs[1][-1][2]],
            alpha=cfg.SLOWFAST.ALPHA,
            beta_inv=cfg.SLOWFAST.BETA_INV,
            norm_module=self.norm_module,
        )

        self.s3 = GhostNet_Stage(
            input_channel=[
                self.slow_cfgs[1][0][2] + self.fast_cfgs[1][-1][2],
                self.fast_cfgs[1][0][2] + self.slow_cfgs[1][-1][
                    2] // cfg.SLOWFAST.BETA_INV
            ],
            slow_cfg=self.slow_cfgs[2],
            fast_cfg=self.fast_cfgs[2],
        )

        self.s3_fuse = FuseFastAndSlow(
            dim_in=[self.slow_cfgs[2][-1][2],
                    self.fast_cfgs[2][-1][2]],
            alpha=cfg.SLOWFAST.ALPHA,
            beta_inv=cfg.SLOWFAST.BETA_INV,
            norm_module=self.norm_module,
        )

        self.s4 = GhostNet_Stage(
            input_channel=[
                self.slow_cfgs[2][0][2] + self.fast_cfgs[2][-1][2],
                self.fast_cfgs[2][0][2] + self.slow_cfgs[2][-1][
                    2] // cfg.SLOWFAST.BETA_INV
            ],
            slow_cfg=self.slow_cfgs[3],
            fast_cfg=self.fast_cfgs[3],
        )

        self.s4_fuse = FuseFastAndSlow(
            dim_in=[self.slow_cfgs[3][-1][2],
                    self.fast_cfgs[3][-1][2]],
            alpha=cfg.SLOWFAST.ALPHA,
            beta_inv=cfg.SLOWFAST.BETA_INV,
            norm_module=self.norm_module,
        )

        self.s5 = GhostNet_Stage(
            input_channel=[
                self.slow_cfgs[3][-1][2] + self.fast_cfgs[3][-1][2],
                self.fast_cfgs[3][-1][2] + self.slow_cfgs[3][-1][
                    2] // cfg.SLOWFAST.BETA_INV
            ],
            slow_cfg=self.slow_cfgs[4],
            fast_cfg=self.fast_cfgs[4],
        )

        # building last several layers
        self.head = head_helper.GhostNetBasicHead(
            input_channel=[
                self.slow_cfgs[4][-1][2],
                self.fast_cfgs[4][-1][2]
            ],
            mid_channel=[
                self.slow_cfgs[4][-1][1],
                self.fast_cfgs[4][-1][1]
            ],
            output_channel=[
                output_channels[0],
                output_channels[1]
            ],
            num_classes=cfg.MODEL.NUM_CLASSES,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )

    def forward(self, x, bboxes=None):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        # for pathway in range(self.num_pathways):
        #     pool = getattr(self, "pathway{}_pool".format(pathway))
        #     x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        # print('s5: ', x[0].shape, x[1].shape)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x


_MOBILE_NET_V2_CONFIGS = {
    # slow path
    "slow_interverted_residual_setting": [
        # t, c, n, s
        [1, 16, 1, (1, 1, 1)],
        [6, 24, 2, (1, 2, 2)],  # *
        [6, 32, 3, (1, 2, 2)],  # *
        [6, 64, 4, (1, 2, 2)],  # *
        [6, 96, 3, (1, 1, 1)],
        [6, 160, 3, (1, 2, 2)],  # *
        [6, 320, 1, (1, 1, 1)],
    ],

    # fast path
    "fast_interverted_residual_setting": [
        # t, c, n, s
        [1, 16, 1, (1, 1, 1)],
        [6, 24, 2, (1, 2, 2)],  # *
        [6, 32, 3, (1, 2, 2)],  # *
        [6, 64, 4, (1, 2, 2)],  # *
        [6, 96, 3, (1, 1, 1)],
        [6, 160, 3, (1, 2, 2)],  # *
        [6, 320, 1, (1, 1, 1)],
    ],

}


@MODEL_REGISTRY.register()
class SlowFastMoibleNetV2(nn.Module):
    """
    Efficient Dual Attention SlowFast Networks for Video Action Recognition

    Dafeng Wei, Ye Tian, Liqing Wei, Hong Zhong, Siqian Chen, Shiliang Pu, Hongtao Lu

    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFastMoibleNetV2, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 2
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1

        width_per_group = 32
        last_channel = 1280
        self.last_channel = int(last_channel * cfg.SLOWFAST.WIDTH_MULTI) if \
            cfg.SLOWFAST.WIDTH_MULTI > 1.0 else last_channel

        # mobile net v2 - stage 1
        self.s1 = stem_helper.MobilenetV2_Model_Stem(
            input_channels=[
                width_per_group,
                width_per_group
            ],
            sample_size=cfg.DATA.CROP_SIZE,
            width_mult=[
                cfg.SLOWFAST.WIDTH_MULTI,  # base width multi for slow path
                cfg.SLOWFAST.WIDTH_MULTI / cfg.SLOWFAST.BETA_INV
            ],
            img_dim=len(cfg.DATA.MEAN)
        )

        slow_layers = _MOBILE_NET_V2_CONFIGS[
            "slow_interverted_residual_setting"]
        fast_layers = _MOBILE_NET_V2_CONFIGS[
            "fast_interverted_residual_setting"]

        self.s2 = MobileNetV2_Stage(
            input_channel=[
                int(width_per_group * cfg.SLOWFAST.WIDTH_MULTI),
                int(
                    width_per_group * cfg.SLOWFAST.WIDTH_MULTI // cfg.SLOWFAST.BETA_INV)
            ],
            slow_residual_setting=slow_layers[0:2],
            fast_residual_setting=fast_layers[0:2],
            width_mult=cfg.SLOWFAST.WIDTH_MULTI,
            beta_inv=cfg.SLOWFAST.BETA_INV,
        )

        self.s3_fuse = FuseFastAndSlow(
            dim_in=[
                int(slow_layers[1][1] * cfg.SLOWFAST.WIDTH_MULTI),
                int(slow_layers[1][
                        1] * cfg.SLOWFAST.WIDTH_MULTI) // cfg.SLOWFAST.BETA_INV,
            ],
            alpha=cfg.SLOWFAST.ALPHA,
            beta_inv=cfg.SLOWFAST.BETA_INV,
            norm_module=self.norm_module,
        )

        self.s4 = MobileNetV2_Stage(
            input_channel=[
                int(slow_layers[1][1] * cfg.SLOWFAST.WIDTH_MULTI +
                    slow_layers[1][
                        1] * cfg.SLOWFAST.WIDTH_MULTI // cfg.SLOWFAST.BETA_INV),
                int(slow_layers[1][
                        1] * cfg.SLOWFAST.WIDTH_MULTI // cfg.SLOWFAST.BETA_INV +
                    slow_layers[1][
                        1] * cfg.SLOWFAST.WIDTH_MULTI // cfg.SLOWFAST.BETA_INV)
            ],
            slow_residual_setting=slow_layers[2:3],
            fast_residual_setting=fast_layers[2:3],
            width_mult=cfg.SLOWFAST.WIDTH_MULTI,
            beta_inv=cfg.SLOWFAST.BETA_INV,
        )
        self.s4_fuse = FuseFastAndSlow(
            dim_in=[
                int(slow_layers[2][1] * cfg.SLOWFAST.WIDTH_MULTI),
                int(slow_layers[2][
                        1] * cfg.SLOWFAST.WIDTH_MULTI) // cfg.SLOWFAST.BETA_INV,
            ],
            alpha=cfg.SLOWFAST.ALPHA,
            beta_inv=cfg.SLOWFAST.BETA_INV,
            norm_module=self.norm_module,
        )

        self.s5 = MobileNetV2_Stage(
            input_channel=[
                int(slow_layers[2][1] * cfg.SLOWFAST.WIDTH_MULTI +
                    slow_layers[2][
                        1] * cfg.SLOWFAST.WIDTH_MULTI // cfg.SLOWFAST.BETA_INV),
                int(slow_layers[2][
                        1] * cfg.SLOWFAST.WIDTH_MULTI // cfg.SLOWFAST.BETA_INV +
                    slow_layers[2][
                        1] * cfg.SLOWFAST.WIDTH_MULTI // cfg.SLOWFAST.BETA_INV)
            ],
            slow_residual_setting=slow_layers[3:4],
            fast_residual_setting=fast_layers[3:4],
            width_mult=cfg.SLOWFAST.WIDTH_MULTI,
            beta_inv=cfg.SLOWFAST.BETA_INV,
        )
        self.s5_fuse = FuseFastAndSlow(
            dim_in=[
                int(slow_layers[3][1] * cfg.SLOWFAST.WIDTH_MULTI),
                int(slow_layers[3][
                        1] * cfg.SLOWFAST.WIDTH_MULTI) // cfg.SLOWFAST.BETA_INV,
            ],
            alpha=cfg.SLOWFAST.ALPHA,
            beta_inv=cfg.SLOWFAST.BETA_INV,
            norm_module=self.norm_module,
        )

        self.s6 = MobileNetV2_Stage(
            input_channel=[
                int(slow_layers[3][1] * cfg.SLOWFAST.WIDTH_MULTI +
                    slow_layers[3][
                        1] * cfg.SLOWFAST.WIDTH_MULTI // cfg.SLOWFAST.BETA_INV),
                int(slow_layers[3][
                        1] * cfg.SLOWFAST.WIDTH_MULTI // cfg.SLOWFAST.BETA_INV +
                    slow_layers[3][
                        1] * cfg.SLOWFAST.WIDTH_MULTI // cfg.SLOWFAST.BETA_INV)
            ],
            slow_residual_setting=slow_layers[4:5],
            fast_residual_setting=fast_layers[4:5],
            width_mult=cfg.SLOWFAST.WIDTH_MULTI,
            beta_inv=cfg.SLOWFAST.BETA_INV,
        )

        self.s7 = MobileNetV2_Stage(
            input_channel=[
                int(slow_layers[4][1] * cfg.SLOWFAST.WIDTH_MULTI),
                int(slow_layers[4][
                        1] * cfg.SLOWFAST.WIDTH_MULTI // cfg.SLOWFAST.BETA_INV)
            ],
            slow_residual_setting=slow_layers[5:6],
            fast_residual_setting=fast_layers[5:6],
            width_mult=cfg.SLOWFAST.WIDTH_MULTI,
            beta_inv=cfg.SLOWFAST.BETA_INV,
        )
        self.s7_fuse = FuseFastAndSlow(
            dim_in=[
                int(slow_layers[5][1] * cfg.SLOWFAST.WIDTH_MULTI),
                int(slow_layers[5][1] * cfg.SLOWFAST.WIDTH_MULTI)//cfg.SLOWFAST.BETA_INV,
            ],
            alpha=cfg.SLOWFAST.ALPHA,
            beta_inv=cfg.SLOWFAST.BETA_INV,
            norm_module=self.norm_module,
        )

        self.s8 = MobileNetV2_Stage(
            input_channel=[
                int(slow_layers[5][1] * cfg.SLOWFAST.WIDTH_MULTI +
                    slow_layers[5][
                        1] * cfg.SLOWFAST.WIDTH_MULTI // cfg.SLOWFAST.BETA_INV),
                int(slow_layers[5][
                        1] * cfg.SLOWFAST.WIDTH_MULTI // cfg.SLOWFAST.BETA_INV +
                    slow_layers[5][
                        1] * cfg.SLOWFAST.WIDTH_MULTI // cfg.SLOWFAST.BETA_INV)
            ],
            slow_residual_setting=slow_layers[6:],
            fast_residual_setting=fast_layers[6:],
            width_mult=cfg.SLOWFAST.WIDTH_MULTI,
            beta_inv=cfg.SLOWFAST.BETA_INV,
        )

        self.head = head_helper.MobileNetV2BasicHead(
            input_channel=[
                int(slow_layers[6][1] * cfg.SLOWFAST.WIDTH_MULTI),
                int(slow_layers[6][
                        1] * cfg.SLOWFAST.WIDTH_MULTI // cfg.SLOWFAST.BETA_INV)
            ],
            last_channel=[
                self.last_channel,
                self.last_channel // cfg.SLOWFAST.BETA_INV
            ],
            num_classes=cfg.MODEL.NUM_CLASSES,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        # x = self.s1_fuse(x)
        x = self.s2(x)
        # x = self.s2_fuse(x)
        # for pathway in range(self.num_pathways):
        #     pool = getattr(self, "pathway{}_pool".format(pathway))
        #     x[pathway] = pool(x[pathway])
        # x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        x = self.s5_fuse(x)
        x = self.s6(x)
        # x = self.s6_fuse(x)
        x = self.s7(x)
        x = self.s7_fuse(x)
        x = self.s8(x)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x
