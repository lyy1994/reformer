# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options
from fairseq import utils

from fairseq.modules import (
    AdaptiveSoftmax, LearnedPositionalEmbedding, MultiheadAttention2D,
    SinusoidalPositionalEmbedding
)

from . import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqModel, register_model,
    register_model_architecture,
)


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

        parser.add_argument('--src-tgt-embed', action='store_true',
                            help='use source and target embeddings')
        parser.add_argument('--non-parametric-normalize', action='store_true',
                            help='layer normalization without element-wise affine transformation')
        parser.add_argument('--decoder-sublayer-before', action='store_true',
                            help='apply decoder self-attention before encoder self-attention')
        parser.add_argument('--encoder-ffn', action='store_true',
                            help='apply ffn after encoder self-attention')
        parser.add_argument('--encoder-single-relu', action='store_true',
                            help='apply relu if there is no ffn after encoder self-attention')
        parser.add_argument('--encoder-sublayers', type=int, metavar='N',
                            help='num encoder sublayers within one block')
        parser.add_argument('--decoder-ffn', action='store_true',
                            help='apply ffn after decoder self-attention')
        parser.add_argument('--decoder-single-relu', action='store_true',
                            help='apply relu if there is no ffn after decoder self-attention')
        parser.add_argument('--decoder-sublayers', type=int, metavar='N',
                            help='num decoder sublayers within one block')

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
        if not args.model_parallelism:
            return model
        else:
            # we first put all parameters and buffers to a single GPU
            model.cuda()
            encoder_sublayers = args.encoder_sublayers if not args.encoder_ffn else args.encoder_sublayers + 1
            decoder_sublayers = args.decoder_sublayers if not args.decoder_ffn else args.decoder_sublayers + 1
            nsublayers = args.decoder_layers * (encoder_sublayers + decoder_sublayers)
            # starting from N implies making extra N sublayers space for embeddings and softmax in the first device
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
            if args.model_parallelism_debug:
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

        self.register_buffer('version', torch.Tensor([2]))

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
            x += self.src_embed.unsqueeze(0)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

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

    def upgrade_state_dict(self, state_dict):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'encoder.embed_positions.weights' in state_dict:
                del state_dict['encoder.embed_positions.weights']
            state_dict['encoder.embed_positions._float_tensor'] = torch.FloatTensor(1)
        if utils.item(state_dict.get('encoder.version', torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict['encoder.version'] = torch.Tensor([1])
        return state_dict


class ReformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``False``
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False, final_norm=True):
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

        self.layers = nn.ModuleList([])
        self.layers.extend([
            ReformerDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        # reduce function to compress model output before softmax
        # Target x Source x Batch x Channel -> T x B x C
        # TODO: more reduction functions
        self.reduce = lambda x: x.max(dim=1)[0]

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
            self.layer_norm = LayerNorm(model_dim, not args.non_parametric_normalize)

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
            x += self.tgt_embed.unsqueeze(0)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        # form a source-target 2D representation
        # Target x Source x Batch x 2*Channel
        # TODO: more ways to form 2D representation
        src_len = encoder_out['encoder_out'].size(0)
        tgt_len = x.size(0)
        x = torch.cat(
            (encoder_out['encoder_out'].unsqueeze(0).repeat(tgt_len, 1, 1, 1),
             x.unsqueeze(1).repeat(1, src_len, 1, 1)), -1)

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

        if self.normalize:
            x = self.layer_norm(x)

        # reduce the Source dim
        # TODO: reduce after project_out_dim
        x = self.reduce(x)

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
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
    """

    def __init__(self, args, no_encoder_attn=False):
        super().__init__()
        self.no_encoder_attn = no_encoder_attn
        self.decoder_sublayer_before = args.decoder_sublayer_before
        # sublayer declaration order must match their computation order, which
        # helps to avoid potential extra communication cost due to auto-register
        self.maybe_declare('decoder', args, before=True)
        if not self.no_encoder_attn:
            self.declare('encoder', args)
        self.maybe_declare('decoder', args, after=True)

    def maybe_declare(self, sublayer_type, args, before=False, after=False):
        assert before ^ after
        if after ^ getattr(self, f'{sublayer_type}_sublayer_before'):
            self.declare(sublayer_type, args)

    def declare(self, sublayer_type, args):
        assert sublayer_type in ['encoder', 'decoder']
        decoder_attn = True if sublayer_type == 'decoder' else False
        nsublayers = getattr(args, f'{sublayer_type}_sublayers')
        sublayers = nn.ModuleList([])
        for _ in range(nsublayers):
            # add self-attention layer
            sublayers.append(ReformerDecoderSubLayer(args, decoder_attn=decoder_attn, is_ffn=False))
            # add ffn layer
            if getattr(args, f'{sublayer_type}_ffn'):
                sublayers.append(ReformerDecoderSubLayer(args, decoder_attn=decoder_attn, is_ffn=True))
            elif getattr(args, f'{sublayer_type}_single_relu'):
                sublayers.append(nn.ReLU())
        setattr(self, f'{sublayer_type}_sublayers', sublayers)

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
        x, dec_attn = self.maybe_run('decoder', x, encoder_padding_mask, incremental_state,
                                     self_attn_mask, self_attn_padding_mask, before=True)

        enc_attn = None
        if not self.no_encoder_attn:
            x, enc_attn = self.run('encoder', x, encoder_padding_mask, incremental_state,
                                   self_attn_mask=self_attn_mask,
                                   self_attn_padding_mask=self_attn_padding_mask)

        x, dec_attn = self.maybe_run('decoder', x, encoder_padding_mask, incremental_state,
                                     self_attn_mask, self_attn_padding_mask, after=True)

        return x, enc_attn

    def maybe_run(self, sublayer_type, x, encoder_padding_mask, incremental_state, self_attn_mask,
                  self_attn_padding_mask, before=False, after=False):
        assert before ^ after
        attn = None
        if after ^ getattr(self, f'{sublayer_type}_sublayer_before'):
            x, attn = self.run(sublayer_type, x, encoder_padding_mask, incremental_state,
                               self_attn_mask=self_attn_mask,
                               self_attn_padding_mask=self_attn_padding_mask)
        return x, attn

    def run(self, sublayer_type, x, encoder_padding_mask, incremental_state, self_attn_mask, self_attn_padding_mask):
        assert sublayer_type in ['encoder', 'decoder']
        attn = None
        for layer in getattr(self, f'{sublayer_type}_sublayers'):
            if isinstance(layer, ReformerDecoderSubLayer):
                x, attn = layer(x, encoder_padding_mask, incremental_state,
                                self_attn_mask=self_attn_mask,
                                self_attn_padding_mask=self_attn_padding_mask)
            else:
                x = layer(x)
        return x, attn


INCREMENTAL_MODULE_INSTANCE_ID = collections.defaultdict(lambda: 0)


def register_module(init_fn):
    def register(self, *args, **kwargs):
        module_name = self.__class__.__name__
        # assign a unique id to each module instance
        if not hasattr(self, '_id'):
            INCREMENTAL_MODULE_INSTANCE_ID[module_name] += 1
            self._id = INCREMENTAL_MODULE_INSTANCE_ID[module_name]
        return init_fn(self, *args, **kwargs)
    return register


def fetch_input(forward_fn):
    def _forward_fn(self, *args, **kwargs):
        args = [item.cuda(MODULE_DEVICE[self._id])
                if isinstance(item, torch.Tensor) and MODULE_DEVICE[self._id] is not None else item
                for item in args]
        kwargs = {key: value.cuda(MODULE_DEVICE[self._id])
                  if isinstance(value, torch.Tensor) and MODULE_DEVICE[self._id] is not None else value
                  for key, value in kwargs.items()}
        return forward_fn(self, *args, **kwargs)
    return _forward_fn


class ReformerDecoderSubLayer(nn.Module):
    """
    Decoder sublayer, performs only one attention/ffn followed with add residual
    and layer normalization
    """

    @register_module
    def __init__(self, args, decoder_attn=True, is_ffn=True):
        super().__init__()
        self.decoder_attn = decoder_attn
        self.is_ffn = is_ffn
        self.embed_dim = args.decoder_model_dim
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.decoder_normalize_before

        if self.is_ffn:
            self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
            self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)
        else:
            self.self_attn = MultiheadAttention2D(
                self.embed_dim, args.decoder_attention_heads,
                dropout=args.attention_dropout,
                tgt_attn=self.decoder_attn,
            )
            self.need_attn = True

        self.layer_norm = LayerNorm(self.embed_dim, not args.non_parametric_normalize)

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
        attn = None

        residual = x
        x = self.maybe_layer_norm(self.layer_norm, x, before=True)
        if self.is_ffn:
            x = F.relu(self.fc1(x))
            x = F.dropout(x, p=self.relu_dropout, training=self.training)
            x = self.fc2(x)
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask if self.decoder_attn else encoder_padding_mask,
                incremental_state=incremental_state,
                need_weights=(not self.training and self.need_attn),
                attn_mask=self_attn_mask if self.decoder_attn else None,
            )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.layer_norm, x, after=True)
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


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
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)

    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)
    args.decoder_model_dim = getattr(args, 'decoder_model_dim', args.encoder_embed_dim + args.decoder_embed_dim)
    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_model_dim)

    args.src_tgt_embed = getattr(args, 'src_tgt_embed', False)
    args.non_parametric_normalize = getattr(args, 'non_parametric_normalize', False)
    args.decoder_sublayer_before = getattr(args, 'decoder_sublayer_before', False)
    args.encoder_ffn = getattr(args, 'encoder_ffn', False)
    args.encoder_single_relu = getattr(args, 'encoder_single_relu', False)
    args.encoder_sublayers = getattr(args, 'encoder_sublayers', 1)
    args.decoder_ffn = getattr(args, 'decoder_ffn', False)
    args.decoder_single_relu = getattr(args, 'decoder_single_relu', False)
    args.decoder_sublayers = getattr(args, 'decoder_sublayers', 1)


@register_model_architecture('reformer', 'reformer_iwslt_de_en')
def reformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 2)
    base_architecture(args)


@register_model_architecture('reformer', 'reformer_wmt_en_de')
def reformer_wmt_en_de(args):
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani, et al, 2017)
@register_model_architecture('reformer', 'reformer_vaswani_wmt_en_de_big')
def reformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.dropout = getattr(args, 'dropout', 0.3)
    base_architecture(args)


@register_model_architecture('reformer', 'reformer_vaswani_wmt_en_fr_big')
def reformer_vaswani_wmt_en_fr_big(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    reformer_vaswani_wmt_en_de_big(args)


@register_model_architecture('reformer', 'reformer_wmt_en_de_big')
def reformer_wmt_en_de_big(args):
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    reformer_vaswani_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture('reformer', 'reformer_wmt_en_de_big_t2t')
def reformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    reformer_vaswani_wmt_en_de_big(args)
