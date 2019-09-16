# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import collections
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from fairseq import options
from fairseq import utils

from fairseq.modules import (
    AdaptiveSoftmax, LearnedPositionalEmbedding, SeparableAttention,
    SinusoidalPositionalEmbedding, Reduction, Dropout1d, Dropout2d,
)

from fairseq.models.transformer import (
    TransformerEncoderLayer,
)

from . import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqModel, register_model,
    register_model_architecture,
)

VALID_INPUT_LAYER = {
    'cat': lambda encoder_embed_dim, decoder_embed_dim: encoder_embed_dim + decoder_embed_dim,
    'add': lambda encoder_embed_dim, decoder_embed_dim: decoder_embed_dim,
}

MODULE_DEVICE = collections.defaultdict(lambda: None)


@register_model('reformer')
class ReformerModel(FairseqModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (ReformerEncoder): the encoder
        decoder (ReformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')

        parser.add_argument('--decoder-input-layer', choices=VALID_INPUT_LAYER.keys(),
                            help='the method chosen to produce the 2D input')
        parser.add_argument('--decoder-output-layer', choices=Reduction.VALID_REDUCTION.keys(),
                            help='the method chosen to produce the 1D output')
        parser.add_argument('--src-tgt-embed', action='store_true',
                            help='use source and target embeddings')
        parser.add_argument('--layer-chain', type=str, metavar='STR',
                            help='specify the instruction of layers')
        parser.add_argument('--memory-efficient', default=False, action='store_true',
                            help='checkpoint all attentions, ~25% slower, ~38% less memory')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise RuntimeError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise RuntimeError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise RuntimeError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = ReformerEncoder(args, src_dict, encoder_embed_tokens)
        decoder = ReformerDecoder(args, tgt_dict, decoder_embed_tokens)
        return cls.model_parallelism(ReformerModel(encoder, decoder), args)

    @staticmethod
    def model_parallelism(model, args):
        if args.model_parallelism_world_size == 1:
            return model
        else:
            # we first put all parameters and buffers to a single GPU
            model.cuda()
            nsublayers = INCREMENTAL_MODULE_INSTANCE_ID[ReformerDecoderSubLayer.__name__]
            # starting from N implies making extra N sublayers space in the first device (embeddings, softmax, encoder)
            sublayer_id = args.pseudo_sublayers
            sublayers_per_device = math.ceil((nsublayers + sublayer_id) / args.model_parallelism_world_size)
            names, devices = ['module name'], ['device']
            for name, module in model.named_modules(prefix='reformer'):
                # we only distribute sublayers to different GPU
                if isinstance(module, ReformerDecoderSubLayer):
                    MODULE_DEVICE[module._id] = (sublayer_id // sublayers_per_device)
                    assert MODULE_DEVICE[module._id] < torch.cuda.device_count(), \
                        f'try to assign a sublayer {name} to a invalid device {MODULE_DEVICE[module._id]}\n' \
                        f'Fix bugs in the sublayer allocation strategy!!!'
                    module.cuda(MODULE_DEVICE[module._id])
                    names.append(name)
                    devices.append(f'cuda:{MODULE_DEVICE[module._id]}')
                    sublayer_id += 1
                # we push the dimension reduction to the last GPU (with potentially larger space)
                # to avoid communication overhead (as well as the last layer norm)
                if isinstance(module, ReformerOutputLayer):
                    module.cuda(0)
                    names.append(name)
                    devices.append(f'cuda:{0}')
                if '.decoder.layer_norm' in name:
                    module.cuda(0)
            if args.debug:
                name_width = max([len(e) for e in names])
                device_width = max([len(e) for e in devices])
                separate_line = '|' + '-' * (name_width + 2) + '|' + '-' * (device_width + 2) + '|'
                print(separate_line)
                for i, (name, device) in enumerate(zip(names, devices)):
                    if i == 0:
                        print(f"| {name: ^{name_width}} | {device: ^{device_width}} |")
                        print(separate_line)
                    else:
                        print(f"| {name: <{name_width}} | {device: <{device_width}} |")
                print(separate_line)
            return model


class ReformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``True``
    """

    def __init__(self, args, dictionary, embed_tokens, left_pad=True):
        super().__init__(dictionary)
        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.src_embed = nn.Parameter(nn.init.normal_(
            torch.Tensor(embed_dim),
            mean=0, std=embed_dim ** -0.5
        )) if args.src_tgt_embed else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for _ in range(args.encoder_layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        if self.src_embed is not None:
            x += self.src_embed.unsqueeze(0) * self.embed_scale
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.normalize:
            x = self.layer_norm(x)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class ReformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``False``
    """

    def __init__(self, args, dictionary, embed_tokens, left_pad=False, final_norm=True):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        output_embed_dim = args.decoder_output_dim

        model_dim = args.decoder_model_dim

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False,
                                     uniform=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.tgt_embed = nn.Parameter(nn.init.normal_(
            torch.Tensor(embed_dim),
            mean=0, std=embed_dim ** -0.5
        )) if args.src_tgt_embed else None

        self.input_layer = ReformerInputLayer(args)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            ReformerDecoderLayer(args)
            for _ in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(model_dim, output_embed_dim,
                                      bias=False, uniform=False) if model_dim != output_embed_dim else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary), output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=output_embed_dim ** -0.5)
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = LayerNorm(model_dim)

        # output_layer function to compress model output before softmax
        # Target x Source x Batch x Channel -> T x B x C
        self.output_layer = ReformerOutputLayer(args)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        if self.tgt_embed is not None:
            x += self.tgt_embed.unsqueeze(0) * self.embed_scale
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        src_input, tgt_input = encoder_out['encoder_out'], x
        # form a source-target 2D representation
        # T x B x C -> T x S x B x C
        x = self.input_layer(src_input, tgt_input)

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)

        # push the result to where softmax/embeddings hosted
        x = x.to(self.embed_tokens.weight.device)

        # T x S x B x C -> T x B x C
        # TODO: output_layer after project_out_dim
        x = self.output_layer(x, encoder_out['encoder_padding_mask'] if encoder_out is not None else None)

        if self.normalize:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = F.linear(x, self.embed_out)

        return x, {'attn': attn, 'inner_states': inner_states}

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]


class ReformerInputLayer(nn.Module):
    """
    Decoder input layer, transforms two sequential inputs to a 2D representation
    """

    def __init__(self, args):
        super().__init__()
        # TODO: more ways to form 2D representation
        self.input_layer = args.decoder_input_layer
        self.scaling = 0.5 if args.arch.startswith('reformer_v1') else 1.

    def extra_repr(self):
        return 'input_layer={}, scaling={}'.format(self.input_layer, self.scaling)

    def forward(self, src_embed, tgt_embed):
        x = None
        src_len = src_embed.size(0)
        tgt_len = tgt_embed.size(0)
        if self.input_layer == 'cat':
            x = torch.cat(
                (src_embed.unsqueeze(0).repeat(tgt_len, 1, 1, 1),
                 tgt_embed.unsqueeze(1).repeat(1, src_len, 1, 1)), -1)
        elif self.input_layer == 'add':
            assert src_embed.size(-1) == tgt_embed.size(-1), \
                f'source embedding dim ({src_embed.size(-1)}) must match target embedding dim({tgt_embed.size(-1)}) ' \
                f'when using input layer {self.input_layer}'
            x = src_embed.unsqueeze(0).repeat(tgt_len, 1, 1, 1) + tgt_embed.unsqueeze(1).repeat(1, src_len, 1, 1) * self.scaling
        return x


class ReformerOutputLayer(nn.Module):
    """
    Decoder output layer, transforms a 2D representation to a single sequential output
    """

    def __init__(self, args):
        super().__init__()
        self.reduction = Reduction(args.decoder_output_layer, args.decoder_normalize_before, args)

    def forward(self, x, encoder_padding_mask):
        # since reduction happens after layer_norm, additional layer_norm might be required after the
        # reduction, especially for those reduction variants that does not preserve output scale
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.to(x.device)
        x = self.reduction(x, encoder_padding_mask)
        return x


def register_to(name: str, mapping: dict):
    def wrapper(fn):
        mapping[name] = fn
        return fn

    return wrapper


class ReformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.layer_chain = args.layer_chain
        self.sublayers = nn.ModuleList([])
        # sublayer declaration order must match their computation order, which
        # helps to avoid potential extra communication cost due to auto-register
        for op, parsed_args in self.parse(self.layer_chain):
            self.sublayers.append(ReformerDecoderSubLayer(args, op, *parsed_args))

    @staticmethod
    def parse(layer_chain: str) -> list:
        if not hasattr(ReformerDecoderLayer.parse, 'str2arg'):
            ReformerDecoderLayer.parse.str2arg = {'enc': False, 'dec': True}
        ops = [op.split(':') for op in layer_chain.split('+')]
        ops_with_args = []
        for op_with_args in ops:
            op = op_with_args[0]
            args = []
            if len(op_with_args) == 2:
                for arg in op_with_args[1].split(','):
                    try:
                        args.append(ReformerDecoderLayer.parse.str2arg[arg])
                    except KeyError:
                        continue
            ops_with_args.append([op, args])
        return ops_with_args

    def forward(self, x, encoder_padding_mask, incremental_state,
                self_attn_mask=None, self_attn_padding_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        attn = None
        for sublayer in self.sublayers:
            x, attn = sublayer(x, encoder_padding_mask, incremental_state,
                               self_attn_mask, self_attn_padding_mask)
        return x, attn


INCREMENTAL_MODULE_INSTANCE_ID = collections.defaultdict(lambda: 0)


def register_module(init_fn):
    @functools.wraps(init_fn)
    def wrapper(self, *args, **kwargs):
        module_name = self.__class__.__name__
        # assign a unique id to each module instance
        if not hasattr(self, '_id'):
            INCREMENTAL_MODULE_INSTANCE_ID[module_name] += 1
            self._id = INCREMENTAL_MODULE_INSTANCE_ID[module_name]
        return init_fn(self, *args, **kwargs)

    return wrapper


def fetch_input(forward_fn):
    @functools.wraps(forward_fn)
    def wrapper(self, *args, **kwargs):
        args = [item.cuda(MODULE_DEVICE[self._id])
                if isinstance(item, torch.Tensor) and MODULE_DEVICE[self._id] is not None else item
                for item in args]
        kwargs = {key: value.cuda(MODULE_DEVICE[self._id])
                  if isinstance(value, torch.Tensor) and MODULE_DEVICE[self._id] is not None else value
                  for key, value in kwargs.items()}
        return forward_fn(self, *args, **kwargs)

    return wrapper


VALID_SUBLAYER = {}


class ReformerDecoderSubLayer(nn.Module):
    """
    Decoder sublayer, performs only one attention/ffn followed with add residual
    and layer normalization
    """

    @register_module
    def __init__(self, args, layer_type, decoder_attn=True):
        super().__init__()
        self.decoder_attn = decoder_attn
        assert layer_type in VALID_SUBLAYER.keys()
        self.layer_type = layer_type
        self.embed_dim = args.decoder_model_dim
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.decoder_normalize_before

        self.customize_forward = VALID_SUBLAYER[self.layer_type](self, args)

        self.layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

    def extra_repr(self):
        return 'layer_type={}, normalize_before={},'.format(self.layer_type, self.normalize_before)

    @fetch_input
    def forward(self, x, encoder_padding_mask, incremental_state,
                self_attn_mask=None, self_attn_padding_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        return self.customize_forward(x, encoder_padding_mask, incremental_state,
                                      self_attn_mask, self_attn_padding_mask)

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    @register_to('ffn2d', VALID_SUBLAYER)
    def ffn2d(self, args):
        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.relu_dropout2d = Dropout2d(p=self.relu_dropout)
        self.dropout2d = Dropout2d(p=self.dropout)

        def forward(x, encoder_padding_mask, incremental_state,
                    self_attn_mask, self_attn_padding_mask):
            residual = x
            x = self.maybe_layer_norm(self.layer_norm, x, before=True)

            x = F.relu(self.fc1(x))
            x = self.relu_dropout2d(x)
            x = self.fc2(x)

            x = self.dropout2d(x)
            x = residual + x
            x = self.maybe_layer_norm(self.layer_norm, x, after=True)
            return x, None

        return forward

    @register_to('attn1d', VALID_SUBLAYER)
    def attn1d(self, args):
        self.self_attn = SeparableAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
            tgt_attn=self.decoder_attn,
        )
        self.dropout1d = Dropout1d(p=self.dropout, dim=1 if self.decoder_attn else 0)

        def forward(x, encoder_padding_mask, incremental_state,
                    self_attn_mask, self_attn_padding_mask):
            residual = x
            x = self.maybe_layer_norm(self.layer_norm, x, before=True)
            if not args.memory_efficient:
                x, attn = self.self_attn(
                    query=x,
                    key=x,
                    value=x,
                    key_padding_mask=self_attn_padding_mask if self.decoder_attn else encoder_padding_mask,
                    incremental_state=incremental_state,
                    need_weights=(not self.training and self.need_attn),
                    attn_mask=self_attn_mask if self.decoder_attn else None,
                )
            else:
                x, attn = ckpt.checkpoint(
                    lambda h: self.self_attn(
                        query=h,
                        key=h,
                        value=h,
                        key_padding_mask=self_attn_padding_mask if self.decoder_attn else encoder_padding_mask,
                        incremental_state=incremental_state,
                        need_weights=True,
                        attn_mask=self_attn_mask if self.decoder_attn else None,
                    ),
                    x,
                )
            x = self.dropout1d(x)
            x = residual + x
            x = self.maybe_layer_norm(self.layer_norm, x, after=True)
            return x, attn

        return forward


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LayerNorm(embedding_dim, affine=True):
    m = nn.LayerNorm(embedding_dim, elementwise_affine=affine)
    return m


def Linear(in_features, out_features, bias=True, uniform=True):
    m = nn.Linear(in_features, out_features, bias)
    if uniform:
        nn.init.xavier_uniform_(m.weight)
    else:
        nn.init.xavier_normal_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings + padding_idx + 1, embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, num_embeddings + padding_idx + 1)
    return m


@register_model_architecture('reformer', 'reformer')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 5)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', args.encoder_layers)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', args.encoder_attention_heads)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', args.encoder_normalize_before)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', args.encoder_learned_pos)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)

    args.decoder_input_layer = getattr(args, 'decoder_input_layer', 'add')
    args.decoder_output_layer = getattr(args, 'decoder_output_layer', 'attn')

    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)
    args.decoder_model_dim = getattr(args, 'decoder_model_dim',
                                     VALID_INPUT_LAYER[args.decoder_input_layer](
                                         args.encoder_embed_dim, args.decoder_embed_dim))
    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_model_dim)

    args.src_tgt_embed = getattr(args, 'src_tgt_embed', False)
    args.layer_chain = getattr(args, 'layer_chain', 'attn1d:dec+ffn2d:dec+attn1d:enc+ffn2d:enc')
    args.memory_efficient = getattr(args, 'memory_efficient', False)


@register_model_architecture('reformer', 'reformer_base_iwslt_de_en')
def reformer_base_iwslt_de_en(args):
    args.dropout = getattr(args, 'dropout', 0.3)
    args.encoder_layers = getattr(args, 'encoder_layers', 0)
    args.decoder_layers = getattr(args, 'decoder_layers', 7)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    base_architecture(args)


@register_model_architecture('reformer', 'reformer_fast_iwslt_de_en')
def reformer_fast_iwslt_de_en(args):
    args.dropout = getattr(args, 'dropout', 0.3)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    base_architecture(args)


@register_model_architecture('reformer', 'reformer_fast_iwslt_de_en_scaling')
def reformer_fast_iwslt_de_en_scaling(args):
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1536)
    args.encoder_layers = getattr(args, 'encoder_layers', 7)
    reformer_fast_iwslt_de_en(args)


@register_model_architecture('reformer', 'reformer_base_nist_zh_en')
def reformer_base_nist_zh_en(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 0)
    args.decoder_layers = getattr(args, 'decoder_layers', 7)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    base_architecture(args)


@register_model_architecture('reformer', 'reformer_fast_nist_zh_en')
def reformer_fast_nist_zh_en(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    base_architecture(args)


@register_model_architecture('reformer', 'reformer_fast_nist_zh_en_scaling')
def reformer_fast_nist_zh_en_scaling(args):
    args.dropout = getattr(args, 'dropout', 0.2)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_layers = getattr(args, 'encoder_layers', 7)
    reformer_fast_nist_zh_en(args)
