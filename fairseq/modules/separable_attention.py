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
from fairseq.modules import (
    Dropout1d,
)


class SeparableAttention(nn.Module):
    """Multi-headed attention with 2D inputs (src x tgt).
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 tgt_attn=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.tgt_attn = tgt_attn

        self.dropout1d = Dropout1d(p=dropout, dim=0)

        self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def extra_repr(self):
        return 'tgt_attn={}, num_heads={},'.format(self.tgt_attn, self.num_heads)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None, need_weights=True,
                attn_mask=None):
        """
        To perform decoder self-attention: tgt_attn=True, attn_mask is not None (train).
        To perform encoder self-attention: tgt_attn=False, key_padding_mask is not None, qkv_same=True.
        :param query: Output x Source x Batch x Channel
        :param key: Input x Source x Batch x Channel
        :param value: the same to key
        :param key_padding_mask: Batch x Source, required only tgt_attn=False
        :param incremental_state:
        :param need_weights:
        :param attn_mask: Output x Input, required only tgt_attn=True
        :return:
        """
        # since data_ptr() is used to detect whether q, k, v are the same be
        # careful with the case that q, k, v have different dims yet sliced
        # from one tensor
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        if not self.tgt_attn:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out_len, src_size, true_bsz, embed_dim = query.size()
        bsz = src_size * true_bsz
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [out_len, src_size, true_bsz, embed_dim]
        assert key.size() == value.size()

        # encoder self-attention does not need cache
        if incremental_state is not None and self.tgt_attn:
            saved_state = self._get_input_buffer(incremental_state)
        else:
            saved_state = None

        if qkv_same:
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        if saved_state is not None:

            if 'prev_key' in saved_state:
                k = torch.cat((saved_state['prev_key'], k), dim=0)
            if 'prev_value' in saved_state:
                v = torch.cat((saved_state['prev_value'], v), dim=0)
            saved_state['prev_key'] = k
            saved_state['prev_value'] = v

            self._set_input_buffer(incremental_state, saved_state)

        in_len = k.size(0)

        q = q.contiguous().view(out_len, src_size * true_bsz, embed_dim)
        k = k.contiguous().view(in_len, src_size * true_bsz, embed_dim)
        v = v.contiguous().view(in_len, src_size * true_bsz, embed_dim)

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == true_bsz
            assert key_padding_mask.size(1) == in_len

        q = q.contiguous().view(out_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(in_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(in_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if self.add_zero_attn:
            in_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, out_len, in_len]

        if attn_mask is not None:
            attn_weights += attn_mask.unsqueeze(0)
        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(src_size, true_bsz, self.num_heads, out_len, in_len)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).unsqueeze(0)
            attn_weights = attn_weights.float().masked_fill(
                key_padding_mask,
                float('-inf'),
            ).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(src_size * true_bsz * self.num_heads, out_len, in_len)

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = attn_weights.view(src_size, true_bsz, self.num_heads, out_len, in_len)
        attn_weights = self.dropout1d(attn_weights)
        attn_weights = attn_weights.view(bsz * self.num_heads, out_len, in_len)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, out_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(out_len, src_size, true_bsz, embed_dim)
        attn = self.out_proj(attn)

        if not self.tgt_attn:
            attn = attn.transpose(0, 1)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(src_size, true_bsz, self.num_heads, out_len, in_len)
            attn_weights = attn_weights.sum(dim=2) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        # Although encoder self-attention does not need cache, it still
        # possesses an empty slot due to the calling of this function
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
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )
