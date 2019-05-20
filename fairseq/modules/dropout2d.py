# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
from torch import nn
import torch.nn.functional as F


class Dropout2d(nn.Module):
    """Dropout that drops the entire 2D channel.
    """

    def __init__(self, p=0.5, dim1=0, dim2=1):
        super().__init__()
        self.p = p
        self.dim1 = dim1
        self.dim2 = dim2

    def extra_repr(self):
        return 'p={}, dim1={}, dim2={}'.format(self.p, self.dim1, self.dim2)

    def forward(self, x):
        """
        Drop the entire 2D channel (along dim1 & dim2) with probability `p`.
        :param x: tensor with any shape
        :return: tensor with the same shape and dtype as x
        """
        noise_shape = list(x.size())
        noise_shape[self.dim1] = 1
        noise_shape[self.dim2] = 1
        x = x * F.dropout(torch.ones(noise_shape, dtype=x.dtype, layout=x.layout, device=x.device),
                          p=self.p, training=self.training)
        return x
