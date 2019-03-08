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

from fairseq import utils

_VALID_REDUCER = {}


def register_reducer(name):
    def register_reducer_fn(fn):
        _VALID_REDUCER[name] = fn
        return fn

    return register_reducer_fn


class Reducer(nn.Module):
    """
    Dimension reducer that reduces source/target dimension of the input representation.
    """
    VALID_REDUCER = {}

    def __init__(self, method: str, reduce_src: bool, *args, **kwargs) -> None:
        super().__init__()
        self.method = method
        self.reduce_src = reduce_src
        self.specific_repr = None
        self.customize_forward = self.VALID_REDUCER[self.method](self, *args, **kwargs)

    def extra_repr(self):
        general_repr = 'method={}, reduce_src={},'.format(self.method, self.reduce_src)
        specific_repr = f' {self.specific_repr},' if self.specific_repr is not None else ''
        return general_repr + specific_repr

    def forward(self, *args, **kwargs):
        return self.customize_forward(*args, **kwargs)

    def _prepare_repr(self, store_dict: dict, incremental_state: dict=None):
        """
        Concatenate the previous representation and the current
        representation in the target dimension.
        :param store_dict: Dictionary
        :param incremental_state: Dictionary
        :return: Dictionary
        """
        out_dict = {}
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)

            for key, value in store_dict.items():
                v = value
                if key in saved_state:
                    v = torch.cat((saved_state[key], value), dim=0)
                saved_state[key] = v
                out_dict[key] = v

            self._set_input_buffer(incremental_state, saved_state)
        else:
            out_dict = store_dict
        return out_dict

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                # 2 is the Batch dim
                input_buffer[k] = input_buffer[k].index_select(2, new_order.to(input_buffer[k].device))
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(
            self,
            incremental_state,
            'repr_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(
            self,
            incremental_state,
            'repr_state',
            buffer,
        )

    @register_reducer('max')
    def max(self, *args, **kwargs):
        """
        Retain only the maximum elements over the given dimension.
        :param args: Tuple
        :param kwargs: Dictionary
        :return: Callable
        """

        def reduce_src(x, mask, incremental_state=None):
            """
            Customized forward function
            :param x: torch.FloatTensor, T x S x B x C
            :param mask: torch.ByteTensor, B x S, masked elements indicated by 1
            :param incremental_state: Dictionary
            :return: torch.FloatTensor, T x B x C
            """
            # T x S x B x C
            if mask is not None:
                mask = mask.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
                x = x.masked_fill(
                    mask,
                    float('-inf'),
                )
            # T x B x C
            x = x.max(dim=1)[0]
            return x

        def reduce_tgt(x, mask, incremental_state=None):
            """
            Customized forward function
            :param x: torch.FloatTensor, T x S x B x C
            :param mask: torch.ByteTensor, T x T, masked elements indicated by -inf
            :param incremental_state: Dictionary
            :return: torch.FloatTensor, T x S x B x C
            """
            raise NotImplementedError(f'{self.method} consumes too much memory')

        return reduce_src if self.reduce_src else reduce_tgt

    @register_reducer('attn')
    def attn(self, flags, *args, **kwargs):
        """
        Reduce the given dimension based on the distribution computed by an affine transformation
        (compute attention similar to multi-hop with nhop=model_dim
        but apply attention similar to multi-head with nhead=model_dim).
        :param flags: Namespace
        :param args: Tuple
        :param kwargs: Dictionary
        :return: Callable
        """
        self.weights = Parameter(torch.Tensor(flags.decoder_model_dim, flags.decoder_model_dim))
        nn.init.normal_(self.weights, mean=0, std=flags.decoder_model_dim ** -0.5)

        def reduce_src(x, mask, incremental_state=None):
            """
            Customized forward function
            :param x: torch.FloatTensor, T x S x B x C
            :param mask: torch.ByteTensor, B x S, masked elements indicated by 1
            :param incremental_state: Dictionary
            :return: torch.FloatTensor, T x B x C
            """
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
            return x

        def reduce_tgt(x, mask, incremental_state=None):
            """
            Customized forward function
            :param x: torch.FloatTensor, T x S x B x C
            :param mask: torch.ByteTensor, T x T, masked elements indicated by -inf
            :param incremental_state: Dictionary
            :return: torch.FloatTensor, T x S x B x C
            """
            raise NotImplementedError(f'{self.method} consumes too much memory')

        return reduce_src if self.reduce_src else reduce_tgt

    @register_reducer('softmax')
    def softmax(self, *args, **kwargs):
        """
        Retain the maximum elements over the given dimension via a soft way.
        :param args: Tuple
        :param kwargs: Dictionary
        :return: Callable
        """

        def reduce_src(x, mask, incremental_state=None):
            """
            Customized forward function
            :param x: torch.FloatTensor, T x S x B x C
            :param mask: torch.ByteTensor, B x S, masked elements indicated by 1
            :param incremental_state: Dictionary
            :return: torch.FloatTensor, T x B x C
            """
            weights = x
            # T x S x B x C
            if mask is not None:
                mask = mask.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
                weights = weights.masked_fill(
                    mask,
                    float('-inf'),
                )
            # T x S x B x C
            prob = F.softmax(weights, dim=1)
            assert torch.isnan(prob).byte().any() == 0
            # T x B x C
            x = torch.sum(prob * x, dim=1)
            assert torch.isnan(x).byte().any() == 0
            return x

        def reduce_tgt(x, mask, incremental_state=None):
            """
            Customized forward function
            :param x: torch.FloatTensor, T x S x B x C
            :param mask: torch.ByteTensor, T x T, masked elements indicated by -inf
            :param incremental_state: Dictionary
            :return: torch.FloatTensor, T x S x B x C
            """
            raise NotImplementedError(f'{self.method} consumes too much memory')

        return reduce_src if self.reduce_src else reduce_tgt

    @register_reducer('linear')
    def linear(self, flags, *args, **kwargs):
        """
        Reduce the given dimension based on the distribution computed by an affine transformation.
        :param flags: Namespace
        :param args: Tuple
        :param kwargs: Dictionary
        :return: Callable
        """
        self.weights = Parameter(torch.Tensor(1, flags.decoder_model_dim))
        nn.init.xavier_uniform_(self.weights)

        def reduce_src(x, mask, incremental_state=None):
            """
            Customized forward function
            :param x: torch.FloatTensor, T x S x B x C
            :param mask: torch.ByteTensor, B x S, masked elements indicated by 1
            :param incremental_state: Dictionary
            :return: torch.FloatTensor, T x B x C
            """
            T, S, B, C = x.size()
            # T x S x B x 1
            weights = F.linear(x, self.weights)
            if mask is not None:
                mask = mask.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
                weights = weights.masked_fill(
                    mask,
                    float('-inf'),
                )
            prob = F.softmax(weights, dim=1)
            assert torch.isnan(prob).byte().any() == 0
            prob = prob.transpose(1, 2).contiguous().view(T * B, S, 1)
            x = x.transpose(1, 2).contiguous().view(T * B, S, C)
            x = torch.bmm(prob.transpose(1, 2), x).squeeze(1).view(T, B, C)
            return x

        def reduce_tgt(x, mask, incremental_state=None):
            """
            Customized forward function
            :param x: torch.FloatTensor, T x S x B x C
            :param mask: torch.ByteTensor, T x T, masked elements indicated by -inf
            :param incremental_state: Dictionary
            :return: torch.FloatTensor, T x S x B x C
            """
            out_len = x.size(0)
            x = self._prepare_repr({'prev_repr': x}, incremental_state)['prev_repr']
            in_len, S, B, C = x.size()
            # in_len x S x B x C -> B * S x in_len x C
            x = x.transpose(0, 2).contiguous().view(B * S, in_len, C)
            # B * S x in_len x 1
            weights = F.linear(x, self.weights)
            # B * S x in_len x out_len
            weights = weights.repeat(1, 1, out_len)
            # B * S x out_len x in_len
            weights = weights.transpose(1, 2)
            if mask is not None:
                weights += mask.unsqueeze(0)
            prob = F.softmax(weights, dim=2)
            assert torch.isnan(prob).byte().any() == 0
            # B * S x in_len x C -> out_len x S x B x C
            x = torch.bmm(prob, x).view(B, S, out_len, C).transpose(0, 2)
            return x

        return reduce_src if self.reduce_src else reduce_tgt

    @register_reducer('ffn')
    def ffn(self, flags, *args, **kwargs):
        """
        Reduce the given dimension based on the distribution computed by an feedforward network.
        :param flags: Namespace
        :param args: Tuple
        :param kwargs: Dictionary
        :return: Callable
        """
        self.linear = nn.Linear(flags.decoder_model_dim, flags.decoder_model_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.)
        self.weights = Parameter(torch.Tensor(flags.decoder_model_dim))
        self.weights.data.fill_(1)

        def reduce_src(x, mask, incremental_state=None):
            """
            Customized forward function
            :param x: torch.FloatTensor, T x S x B x C
            :param mask: torch.ByteTensor, B x S, masked elements indicated by 1
            :param incremental_state: Dictionary
            :return: torch.FloatTensor, T x B x C
            """
            T, S, B, C = x.size()
            # T x S x B x 1
            weights = torch.sum(self.weights * torch.tanh(self.linear(x)), dim=-1, keepdim=True)
            if mask is not None:
                mask = mask.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
                weights = weights.masked_fill(
                    mask,
                    float('-inf'),
                )
            prob = F.softmax(weights, dim=1)
            assert torch.isnan(prob).byte().any() == 0
            prob = prob.transpose(1, 2).contiguous().view(T * B, S, 1)
            x = x.transpose(1, 2).contiguous().view(T * B, S, C)
            x = torch.bmm(prob.transpose(1, 2), x).squeeze(1).view(T, B, C)
            return x

        def reduce_tgt(x, mask, incremental_state=None):
            """
            Customized forward function
            :param x: torch.FloatTensor, T x S x B x C
            :param mask: torch.ByteTensor, T x T, masked elements indicated by -inf
            :param incremental_state: Dictionary
            :return: torch.FloatTensor, T x S x B x C
            """
            out_len = x.size(0)
            x = self._prepare_repr({'prev_repr': x}, incremental_state)['prev_repr']
            in_len, S, B, C = x.size()
            # in_len x S x B x C -> B * S x in_len x C
            x = x.transpose(0, 2).contiguous().view(B * S, in_len, C)
            # B * S x in_len x 1
            weights = torch.sum(self.weights * torch.tanh(self.linear(x)), dim=-1, keepdim=True)
            # B * S x in_len x out_len
            weights = weights.repeat(1, 1, out_len)
            # B * S x out_len x in_len
            weights = weights.transpose(1, 2)
            if mask is not None:
                weights += mask.unsqueeze(0)
            prob = F.softmax(weights, dim=2)
            assert torch.isnan(prob).byte().any() == 0
            # B * S x in_len x C -> out_len x S x B x C
            x = torch.bmm(prob, x).view(B, S, out_len, C).transpose(0, 2)
            return x

        return reduce_src if self.reduce_src else reduce_tgt

    @register_reducer('avg')
    def avg(self, *args, **kwargs):
        """
        Average all elements over the given dimension.
        :param args: Tuple
        :param kwargs: Dictionary
        :return: Callable
        """

        def reduce_src(x, mask, incremental_state=None):
            """
            Customized forward function
            :param x: torch.FloatTensor, T x S x B x C
            :param mask: torch.ByteTensor, B x S, masked elements indicated by 1
            :param incremental_state: Dictionary
            :return: torch.FloatTensor, T x B x C
            """
            # T x S x B x C
            if mask is not None:
                lengths = (1. - mask.float()).sum(dim=1).unsqueeze(0).unsqueeze(-1)
                mask = mask.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
                x = x.masked_fill(
                    mask,
                    0,
                )
            else:
                lengths = x.size(1)
            # T x B x C
            x = x.sum(dim=1) / lengths
            # common nan source: -inf * 0
            assert torch.isnan(x).byte().any() == 0
            return x

        def reduce_tgt(x, mask, incremental_state=None):
            """
            Customized forward function
            :param x: torch.FloatTensor, T x S x B x C
            :param mask: torch.ByteTensor, T x T, masked elements indicated by -inf
            :param incremental_state: Dictionary
            :return: torch.FloatTensor, T x S x B x C
            """
            raise NotImplementedError(f'{self.method} consumes too much memory')

        return reduce_src if self.reduce_src else reduce_tgt


Reducer.VALID_REDUCER = _VALID_REDUCER
