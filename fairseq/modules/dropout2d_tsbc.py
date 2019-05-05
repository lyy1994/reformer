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

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}'.format(self.p)

    def forward(self, x):
        """
        Drop the entire 2D channel with probability `p`.
        :param x: tensor with shape T x S x B x C
        :return: tensor with the same shape and dtype as x
        """
        noise_shape = [1, 1, x.size(2), x.size(3)]
        x = x * F.dropout(torch.ones(noise_shape, dtype=x.dtype, layout=x.layout, device=x.device),
                          p=self.p, training=self.training)
        return x
