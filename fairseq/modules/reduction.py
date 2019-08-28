# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

_VALID_REDUCTION = {}


def register_to(name: str, mapping: dict):
    def wrapper(fn):
        mapping[name] = fn
        return fn

    return wrapper


class Reduction(nn.Module):
    """
    Dimension reduction that reduces source/target dimension of the input representation.
    """
    VALID_REDUCTION = {}

    def __init__(self, method: str, normalize_before: bool, args) -> None:
        super().__init__()
        self.method = method
        self.normalize_before = normalize_before
        self.specific_repr = None
        self.layer_norm = nn.LayerNorm(args.decoder_model_dim)
        self.customize_forward = self.VALID_REDUCTION[self.method](self, args)

    def extra_repr(self):
        general_repr = 'method={}, normalize_before={},'.format(self.method, self.normalize_before)
        specific_repr = f' {self.specific_repr},' if self.specific_repr is not None else ''
        return general_repr + specific_repr

    def forward(self, *args, **kwargs):
        return self.customize_forward(*args, **kwargs)

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    @register_to('max', _VALID_REDUCTION)
    def max(self, args):
        """
        Reduce the source dimension based on their maximum
        :param args: Namespace
        :return: Callable
        """

        def _forward(x, mask, incremental_state=None):
            """
            Customized forward function
            :param x: torch.FloatTensor, T x S x B x C
            :param mask: torch.ByteTensor, B x S, masked elements indicated by 1
            :param incremental_state: Dictionary
            :return: torch.FloatTensor, T x B x C
            """
            x = self.maybe_layer_norm(self.layer_norm, x, before=True)
            # T x S x B x C
            if mask is not None:
                x = x.masked_fill(
                    mask.transpose(0, 1).unsqueeze(0).unsqueeze(-1),
                    float('-inf'),
                )
            # T x B x C
            x = x.max(dim=1)[0]
            x = self.maybe_layer_norm(self.layer_norm, x, after=True)
            return x

        return _forward

    @register_to('attn', _VALID_REDUCTION)
    def attn(self, args):
        """
        Reduce the given dimension based on the distribution computed by an affine transformation
        (compute attention similar to multi-hop with nhop=model_dim
        but apply attention similar to multi-head with nhead=model_dim).
        :param args: Namespace
        :return: Callable
        """
        self.weights = Parameter(torch.Tensor(args.decoder_model_dim, args.decoder_model_dim))
        nn.init.normal_(self.weights, mean=0, std=args.decoder_model_dim ** -0.5)

        def _forward(x, mask, incremental_state=None):
            """
            Customized forward function
            :param x: torch.FloatTensor, T x S x B x C
            :param mask: torch.ByteTensor, B x S, masked elements indicated by 1
            :param incremental_state: Dictionary
            :return: torch.FloatTensor, T x B x C
            """
            x = self.maybe_layer_norm(self.layer_norm, x, before=True)
            # T x S x B x C
            weights = F.linear(x, self.weights)
            if mask is not None:
                mask = mask.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
                weights = weights.masked_fill(
                    mask,
                    float('-inf'),
                )
            prob = F.softmax(weights, dim=1)
            x = torch.sum(prob * x, dim=1)
            x = self.maybe_layer_norm(self.layer_norm, x, after=True)
            return x

        return _forward


Reduction.VALID_REDUCTION = _VALID_REDUCTION
