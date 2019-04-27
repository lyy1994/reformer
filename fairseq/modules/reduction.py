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

from fairseq.modules import (
    SeparableAttention,
)

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
            assert torch.isnan(prob).byte().any() == 0
            x = torch.sum(prob * x, dim=1)
            x = self.maybe_layer_norm(self.layer_norm, x, after=True)
            return x

        return _forward

    @register_to('multihead', _VALID_REDUCTION)
    def multihead(self, args):
        """
        Reduce the given dimension based on the distribution computed by a
        SE / Bahdanau attention / Multi-dimensional block.
        This version uses mean pooling to obtain global feature vector as query,
        and apply multiplicative attention vector-wise.
        :param args: Namespace
        :return: Callable
        """
        self.dropout = args.dropout
        self.self_attn = SeparableAttention(
            args.decoder_model_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
            tgt_attn=False,
        )

        def _forward(x, mask, incremental_state=None):
            """
            Customized forward function
            :param x: torch.FloatTensor, T x S x B x C
            :param mask: torch.ByteTensor, B x S, masked elements indicated by 1
            :param incremental_state: Dictionary
            :return: torch.FloatTensor, T x B x C
            """
            # Squeeze
            # mean / max pooling to obtain global features as query (here mean pooling)
            # TODO: mean pooling before normalization avoids variance shrink
            # TODO: ffn or bottleneck or just relu before average to compute query as well as dropout (before average and inside ffn) and residual
            x = self.maybe_layer_norm(self.layer_norm, x, before=True)
            # T x S x B x C
            g = x
            if mask is not None:
                lengths = (1. - mask.float()).sum(dim=1).unsqueeze(0).unsqueeze(-1)
                g = g.masked_fill(
                    mask.transpose(0, 1).unsqueeze(0).unsqueeze(-1),
                    0,
                )
            else:
                lengths = g.size(1)
            # T x 1 x B x C
            g = g.sum(dim=1, keepdim=True) / lengths
            # Excitation
            x, _ = self.self_attn(
                query=g,
                key=x,
                value=x,
                key_padding_mask=mask,
                incremental_state=None,
                need_weights=False,
                attn_mask=None,
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + g
            x = x.squeeze(1)
            x = self.maybe_layer_norm(self.layer_norm, x, after=True)
            return x

        return _forward


Reduction.VALID_REDUCTION = _VALID_REDUCTION
