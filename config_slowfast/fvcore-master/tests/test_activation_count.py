# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import typing
import unittest
from collections import Counter, defaultdict

import torch
import torch.nn as nn
from fvcore.nn.activation_count import activation_count
from numpy import prod


class SmallConvNet(nn.Module):
    """
    A network with three conv layers. This is used for testing convolution
    layers for activation count.
    """

    def __init__(self, input_dim: int) -> None:
        super(SmallConvNet, self).__init__()
        conv_dim1 = 8
        conv_dim2 = 4
        conv_dim3 = 2
        self.conv1 = nn.Conv2d(input_dim, conv_dim1, 1, 1)
        self.conv2 = nn.Conv2d(conv_dim1, conv_dim2, 1, 2)
        self.conv3 = nn.Conv2d(conv_dim2, conv_dim3, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def get_gt_activation(self, x: torch.Tensor) -> int:
        count = 0
        x = self.conv1(x)
        count += prod(list(x.size()))
        x = self.conv2(x)
        count += prod(list(x.size()))
        x = self.conv3(x)
        count += prod(list(x.size()))
        return count


class TestActivationCount(unittest.TestCase):
    """
    Unittest for activation_count.
    """

    def test_conv2d(self) -> None:
        """
        Test the activation count for convolutions.
        """
        batch_size = 1
        input_dim = 3
        spatial_dim = 32
        x = torch.randn(batch_size, input_dim, spatial_dim, spatial_dim)
        convNet = SmallConvNet(input_dim)
        ac_dict, _ = activation_count(convNet, (x,))
        gt_count = convNet.get_gt_activation(x)

        gt_dict = defaultdict(float)
        gt_dict["conv"] = gt_count / 1e6
        self.assertDictEqual(
            gt_dict,
            ac_dict,
            "ConvNet with 3 layers failed to pass the activation count test.",
        )

    def test_linear(self) -> None:
        """
        Test the activation count for fully connected layer.
        """
        batch_size = 1
        input_dim = 10
        output_dim = 20
        netLinear = nn.Linear(input_dim, output_dim)
        x = torch.randn(batch_size, input_dim)
        ac_dict, _ = activation_count(netLinear, (x,))
        gt_count = batch_size * output_dim
        gt_dict = defaultdict(float)
        gt_dict["addmm"] = gt_count / 1e6
        self.assertEquals(
            gt_dict, ac_dict, "FC layer failed to pass the activation count test."
        )

    def test_supported_ops(self) -> None:
        """
        Test the activation count for user provided handles.
        """

        def dummy_handle(
            inputs: typing.List[object], outputs: typing.List[object]
        ) -> typing.Counter[str]:
            return Counter({"conv": 100})

        batch_size = 1
        input_dim = 3
        spatial_dim = 32
        x = torch.randn(batch_size, input_dim, spatial_dim, spatial_dim)
        convNet = SmallConvNet(input_dim)
        sp_ops = {"aten::_convolution": dummy_handle}
        ac_dict, _ = activation_count(convNet, (x,), sp_ops)
        gt_dict = defaultdict(float)
        conv_layers = 3
        gt_dict["conv"] = 100 * conv_layers / 1e6
        self.assertDictEqual(
            gt_dict,
            ac_dict,
            "ConvNet with 3 layers failed to pass the activation count test.",
        )
