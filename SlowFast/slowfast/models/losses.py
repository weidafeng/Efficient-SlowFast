#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""
import torch.nn as nn

# wdf add 加权损失函数 06-30
import torch
# weights = torch.FloatTensor([1, 4, 1]).cuda()
# weights = torch.FloatTensor([2, 3, 1]).cuda(non_blocking=True)

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss(reduction="mean"),
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    # "cross_entropy_weighted": nn.CrossEntropyLoss(weight=weights, reduction="mean"),
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
