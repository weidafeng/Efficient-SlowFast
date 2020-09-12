#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""


def add_custom_config(_C):
    # Add your own customized configs.

    # Calculate top(1, TOPK). wdf-fix
    _C.TRAIN.TOPK = 5

    # mobile net
    _C.SLOWFAST.WIDTH_MULTI = 2.
    _C.TENSORBOARD.ENABLE = True

    _C.MODEL.WEIGHTED_RANDOM_SAMPLER = False

    # root path to dataset
    _C.DATA.PATH_TO_DATA_DIR = "/dataset/guojietian/kinetics400/"

    # for our wheel dataset
    _C.DATA.PATH_TO_TRAIN_DATA_TXT = "train_data_191105.txt"
    _C.DATA.PATH_TO_VAL_DATA_TXT = "train_data_for_191025_test.txt"

    # for our tired dataset, whether to use only top half face
    _C.DATA.HALF_FACE = False

    _C.TENSORBOARD.HISTOGRAM.TOPK = 3

    # Model architectures that has one single pathway.
    _C.MODEL.SINGLE_PATHWAY_ARCH = ["c2d", "i3d", "slow", "fast"]

    # shuffle net groups
    _C.SLOWFAST.GROUPS = 1