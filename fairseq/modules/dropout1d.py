# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
from torch import nn
import torch.nn.functional as F


class Dropout1d(nn.Module):
    """Dropout that drops the entire 1D channel.
    """

    def __init__(self, p=0.5, dim=0):
        super().__init__()
        self.p = p
        self.dim = dim

    def extra_repr(self):
        return 'p={}, dim={}'.format(self.p, self.dim)

    def forward(self, x):
        """
        Drop the entire 1D (along `dim`) channel with probability `p`.
        :param x: tensor with any shape
        :return: tensor with the same shape and dtype as x
        """
        noise_shape = list(x.size())
        noise_shape[self.dim] = 1
        x = x * F.dropout(torch.ones(noise_shape, dtype=x.dtype, layout=x.layout, device=x.device),
                          p=self.p, training=self.training)
        return x
