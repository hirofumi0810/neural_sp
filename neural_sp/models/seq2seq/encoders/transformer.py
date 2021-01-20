# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer encoder."""

import copy
from distutils.version import LooseVersion
import logging
import math
import numpy as np
import random
import torch
import torch.nn as nn

from neural_sp.models.modules.positional_embedding import (
    PositionalEncoding,
    XLPositionalEmbedding
)
from neural_sp.models.seq2seq.encoders.conv import ConvEncoder
from neural_sp.models.seq2seq.encoders.encoder_base import EncoderBase
from neural_sp.models.seq2seq.encoders.subsampling import (
    AddSubsampler,
    ConcatSubsampler,
    Conv1dSubsampler,
    DropSubsampler,
    MaxpoolSubsampler
)
from neural_sp.models.seq2seq.encoders.transformer_block import TransformerEncoderBlock
from neural_sp.models.seq2seq.encoders.utils import chunkwise
from neural_sp.models.torch_utils import (
    make_pad_mask,
    tensor2np
)

random.seed(1)

logger = logging.getLogger(__name__)

torch_12_plus = LooseVersion("1.3") > LooseVersion(torch.__version__) >= LooseVersion("1.2")


class TransformerEncoder(EncoderBase):
    """Transformer encoder.

    Args:
        input_dim (int): dimension of input features (freq * channel)
        enc_type (str): type of encoder
        n_heads (int): number of heads for multi-head attention
        n_layers (int): number of blocks
        n_layers_sub1 (int): number of layers in the 1st auxiliary task
        n_layers_sub2 (int): number of layers in the 2nd auxiliary task
        d_model (int): dimension of MultiheadAttentionMechanism
        d_ff (int): dimension of PositionwiseFeedForward
        ffn_bottleneck_dim (int): bottleneck dimension for the light-weight FFN layer
        ffn_activation (str): nonlinear function for PositionwiseFeedForward
        pe_type (str): type of positional encoding
        layer_norm_eps (float): epsilon value for layer normalization
        last_proj_dim (int): dimension of the last projection layer
        dropout_in (float): dropout probability for input-hidden connection
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        dropout_layer (float): LayerDrop probability for layers
        subsample (list): subsample in the corresponding Transformer layers
            ex.) [1, 2, 2, 1] means that subsample is conducted in the 2nd and 3rd layers.
        subsample_type (str): drop/concat/max_pool/1dconv
        n_stacks (int): number of frames to stack
        n_splices (int): frames to splice. Default is 1 frame.
        conv_in_channel (int): number of channels of input features
        conv_channels (int): number of channels in CNN blocks
        conv_kernel_sizes (list): size of kernels in CNN blocks
        conv_strides (list): number of strides in CNN blocks
        conv_poolings (list): size of poolings in CNN blocks
        conv_batch_norm (bool): apply batch normalization only in CNN blocks
        conv_layer_norm (bool): apply layer normalization only in CNN blocks
        conv_bottleneck_dim (int): dimension of the bottleneck layer between CNN and self-attention layers
        conv_param_init (float): only for CNN layers before Transformer layers
        task_specific_layer (bool): add a task specific layer for each sub task
        param_init (str): parameter initialization method
        clamp_len (int): maximum length for relative positional encoding
        lookahead (int): lookahead frames per layer for unidirectional Transformer encoder
        chunk_size_left (int): left chunk size for latency-controlled Transformer encoder
        chunk_size_current (int): current chunk size for latency-controlled Transformer encoder
        chunk_size_right (int): right chunk size for latency-controlled Transformer encoder
        streaming_type (str): implementation methods of latency-controlled Transformer encoder

    """

    def __init__(self, input_dim, enc_type, n_heads,
                 n_layers, n_layers_sub1, n_layers_sub2,
                 d_model, d_ff, ffn_bottleneck_dim, ffn_activation,
                 pe_type, layer_norm_eps, last_proj_dim,
                 dropout_in, dropout, dropout_att, dropout_layer,
                 subsample, subsample_type, n_stacks, n_splices,
                 conv_in_channel, conv_channels, conv_kernel_sizes, conv_strides, conv_poolings,
                 conv_batch_norm, conv_layer_norm, conv_bottleneck_dim, conv_param_init,
                 task_specific_layer, param_init, clamp_len,
                 lookahead, chunk_size_left, chunk_size_current, chunk_size_right, streaming_type):

        super(TransformerEncoder, self).__init__()

        # parse subsample
        subsamples = [1] * n_layers
        for lth, s in enumerate(list(map(int, subsample.split('_')[:n_layers]))):
            subsamples[lth] = s
        # parse lookahead
        lookaheads = [0] * n_layers
        for lth, s in enumerate(list(map(int, lookahead.split('_')[:n_layers]))):
            lookaheads[lth] = s

        if len(subsamples) > 0 and len(subsamples) != n_layers:
            raise ValueError('subsample must be the same size as n_layers. n_layers: %d, subsample: %s' %
                             (n_layers, subsamples))
        if n_layers_sub1 < 0 or (n_layers_sub1 > 1 and n_layers < n_layers_sub1):
            raise Warning('Set n_layers_sub1 between 1 to n_layers. n_layers: %d, n_layers_sub1: %d' %
                          (n_layers, n_layers_sub1))
        if n_layers_sub2 < 0 or (n_layers_sub2 > 1 and n_layers_sub1 < n_layers_sub2):
            raise Warning('Set n_layers_sub2 between 1 to n_layers_sub1. n_layers_sub1: %d, n_layers_sub2: %d' %
                          (n_layers_sub1, n_layers_sub2))

        self.enc_type = enc_type
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pe_type = pe_type
        self.scale = math.sqrt(d_model)

        # for compatibility
        chunk_size_left = str(chunk_size_left)
        chunk_size_current = str(chunk_size_current)
        chunk_size_right = str(chunk_size_right)

        # for streaming encoder
        self.unidir = 'uni' in enc_type
        self.lookaheads = lookaheads
        if sum(lookaheads) > 0:
            assert self.unidir
        self.chunk_size_left = int(chunk_size_left.split('_')[-1]) // n_stacks
        self.chunk_size_current = int(chunk_size_current.split('_')[-1]) // n_stacks
        self.chunk_size_right = int(chunk_size_right.split('_')[-1]) // n_stacks
        self.lc_bidir = self.chunk_size_current > 0 and enc_type != 'conv' and 'uni' not in enc_type
        self.cnn_lookahead = self.unidir or enc_type == 'conv'
        self.streaming_type = streaming_type if self.lc_bidir else ''
        # -: past context
        # *: current context
        # +: future context
        # reshape) overlapped windowing. additional redundant computation is introduced.
        # During inference, caching is not applied. However, considering (N_l+N_c+N_r) is very short
        # and independent on layer depth, the overhead is negligible.
        # chunk1: |**|++
        # chunk2:  --|**|++
        # chunk3:     --|**|++
        # chunk4:        --|**|++
        # chunk5:           --|**|++
        # mask) chunkwise masking. future context is restricted within the current chunk
        # to avoid accumuration of future context depending on the layer depth.
        # chunk1: |**|
        # chunk2:  --|**|
        # chunk3:  -- --|**|
        # chunk4:     -- --|**|
        # chunk5:        -- --|**|
        if self.unidir:
            assert self.chunk_size_left == self.chunk_size_current == self.chunk_size_right == 0
        if self.streaming_type == 'mask':
            assert self.chunk_size_right == 0
            assert self.chunk_size_left == self.chunk_size_current
            # NOTE: this is important to cache CNN output at each chunk
        if self.lc_bidir:
            assert n_layers_sub1 == 0
            assert n_layers_sub2 == 0
            assert not self.unidir

        # for hierarchical encoder
        self.n_layers_sub1 = n_layers_sub1
        self.n_layers_sub2 = n_layers_sub2
        self.task_specific_layer = task_specific_layer

        # for bridge layers
        self.bridge = None
        self.bridge_sub1 = None
        self.bridge_sub2 = None

        # for attention plot
        self.aws_dict = {}
        self.data_dict = {}

        # Setting for CNNs
        if 'conv' in enc_type:
            assert conv_channels
            assert n_stacks == 1 and n_splices == 1
            self.conv = ConvEncoder(input_dim,
                                    in_channel=conv_in_channel,
                                    channels=conv_channels,
                                    kernel_sizes=conv_kernel_sizes,
                                    strides=conv_strides,
                                    poolings=conv_poolings,
                                    dropout=0.,
                                    batch_norm=conv_batch_norm,
                                    layer_norm=conv_layer_norm,
                                    layer_norm_eps=layer_norm_eps,
                                    residual=False,
                                    bottleneck_dim=d_model,
                                    param_init=conv_param_init)
            self._odim = self.conv.output_dim
        else:
            self.conv = None
            self._odim = input_dim * n_splices * n_stacks
            self.embed = nn.Linear(self._odim, d_model)

        # calculate subsampling factor
        self._factor = 1
        if self.conv is not None:
            self._factor *= self.conv.subsampling_factor
        self.subsample = None
        if np.prod(subsamples) > 1:
            self._factor *= np.prod(subsamples)
            if subsample_type == 'max_pool':
                self.subsample = nn.ModuleList([MaxpoolSubsampler(factor)
                                                for factor in subsamples])
            elif subsample_type == 'concat':
                self.subsample = nn.ModuleList([ConcatSubsampler(factor, self._odim)
                                                for factor in subsamples])
            elif subsample_type == 'drop':
                self.subsample = nn.ModuleList([DropSubsampler(factor)
                                                for factor in subsamples])
            elif subsample_type == '1dconv':
                self.subsample = nn.ModuleList([Conv1dSubsampler(factor, self._odim)
                                                for factor in subsamples])
            elif subsample_type == 'add':
                self.subsample = nn.ModuleList([AddSubsampler(factor)
                                                for factor in subsamples])

        if self.chunk_size_left > 0:
            assert self.chunk_size_left % self._factor == 0
        if self.chunk_size_current > 0:
            assert self.chunk_size_current % self._factor == 0
        if self.chunk_size_right > 0:
            assert self.chunk_size_right % self._factor == 0

        self.pos_enc, self.pos_emb = None, None
        self.u_bias, self.v_bias = None, None
        if pe_type in ['relative', 'relative_xl']:
            self.pos_emb = XLPositionalEmbedding(d_model, dropout)
            if pe_type == 'relative_xl':
                self.u_bias = nn.Parameter(torch.Tensor(n_heads, d_model // n_heads))
                self.v_bias = nn.Parameter(torch.Tensor(n_heads, d_model // n_heads))
                # NOTE: u_bias and v_bias are global parameters shared in the whole model
        else:
            self.pos_enc = PositionalEncoding(d_model, dropout_in, pe_type, param_init)

        self.layers = nn.ModuleList([copy.deepcopy(TransformerEncoderBlock(
            d_model, d_ff, n_heads,
            dropout, dropout_att, dropout_layer,
            layer_norm_eps, ffn_activation, param_init,
            pe_type, clamp_len, ffn_bottleneck_dim))
            for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self._odim = d_model

        if n_layers_sub1 > 0:
            if task_specific_layer:
                self.layer_sub1 = TransformerEncoderBlock(
                    d_model, d_ff, n_heads,
                    dropout, dropout_att, dropout_layer,
                    layer_norm_eps, ffn_activation, param_init,
                    pe_type, clamp_len, ffn_bottleneck_dim)
            odim_sub1 = d_model
            if last_proj_dim > 0 and last_proj_dim != self.output_dim:
                self.bridge_sub1 = nn.Linear(self._odim, last_proj_dim)
                odim_sub1 = last_proj_dim
            if n_layers_sub1 == n_layers:
                self.norm_out_sub1 = None
            else:
                self.norm_out_sub1 = nn.LayerNorm(odim_sub1, eps=layer_norm_eps)

        if n_layers_sub2 > 0:
            if task_specific_layer:
                self.layer_sub2 = TransformerEncoderBlock(
                    d_model, d_ff, n_heads,
                    dropout, dropout_att, dropout_layer,
                    layer_norm_eps, ffn_activation, param_init,
                    pe_type, clamp_len, ffn_bottleneck_dim)
            odim_sub2 = d_model
            if last_proj_dim > 0 and last_proj_dim != self.output_dim:
                self.bridge_sub2 = nn.Linear(self._odim, last_proj_dim)
                odim_sub2 = last_proj_dim
            if n_layers_sub2 == n_layers:
                self.norm_out_sub2 = None
            else:
                self.norm_out_sub2 = nn.LayerNorm(odim_sub2, eps=layer_norm_eps)

        if last_proj_dim > 0 and last_proj_dim != self.output_dim:
            self.bridge = nn.Linear(self._odim, last_proj_dim)
            self._odim = last_proj_dim

        self.reset_parameters(param_init)

        # for streaming inference
        self.reset_cache()

    @staticmethod
    def add_args(parser, args):
        """Add arguments."""
        group = parser.add_argument_group("Transformer encoder")
        if 'conv' in args.enc_type:
            parser = ConvEncoder.add_args(parser, args)
        # Transformer common
        if not hasattr(args, 'transformer_layer_norm_eps'):
            group.add_argument('--transformer_ffn_bottleneck_dim', type=int, default=0,
                               help='bottleneck dimension in the FFN layer')
            group.add_argument('--transformer_input_bottleneck_dim', type=int, default=0,
                               help='bottleneck dimension in the FFN layer')
            group.add_argument('--transformer_layer_norm_eps', type=float, default=1e-12,
                               help='epsilon value for layer normalization')
            group.add_argument('--transformer_ffn_activation', type=str, default='relu',
                               choices=['relu', 'gelu', 'gelu_accurate', 'glu', 'swish'],
                               help='nonlinear activation for the FFN layer')
            group.add_argument('--transformer_param_init', type=str, default='xavier_uniform',
                               choices=['xavier_uniform', 'pytorch'],
                               help='parameter initialization')

        # Transformer encoder specific
        group.add_argument('--transformer_enc_d_model', type=int, default=256,
                           help='number of units in the MHA layer for Transformer encoder')
        group.add_argument('--transformer_enc_d_ff', type=int, default=2048,
                           help='number of units in the FFN layer for Transformer encoder')
        group.add_argument('--transformer_enc_n_heads', type=int, default=4,
                           help='number of heads in the MHA layer for Transformer encoder')
        group.add_argument('--transformer_enc_pe_type', type=str, default='add',
                           choices=['add', 'none', 'relative', 'relative_xl'],
                           help='type of positional encoding for Transformer encoder')
        group.add_argument('--dropout_enc_layer', type=float, default=0.0,
                           help='LayerDrop probability for Transformer encoder layers')
        group.add_argument('--transformer_enc_clamp_len', type=int, default=-1,
                           help='maximum length for relative positional encoding. -1 means infinite length.')
        # streaming
        group.add_argument('--transformer_enc_lookaheads', type=str, default="0_0_0_0_0_0_0_0_0_0_0_0",
                           help='lookahead frames per layer for unidirectional Transformer encoder')
        group.add_argument('--lc_chunk_size_left', type=str, default="0",
                           help='left chunk size for latency-controlled Transformer encoder')
        group.add_argument('--lc_chunk_size_current', type=str, default="0",
                           help='current chunk size (and hop size) for latency-controlled Transformer encoder')
        group.add_argument('--lc_chunk_size_right', type=str, default="0",
                           help='right chunk size for latency-controlled Transformer encoder')
        group.add_argument('--lc_type', type=str, default='reshape',
                           choices=['reshape', 'mask'],
                           help='implementation methods of latency-controlled Transformer encoder')
        return parser

    @staticmethod
    def define_name(dir_name, args):
        if 'conv' in args.enc_type:
            dir_name = ConvEncoder.define_name(dir_name, args)

        dir_name += str(args.transformer_enc_d_model) + 'dmodel'
        dir_name += str(args.transformer_enc_d_ff) + 'dff'
        if args.transformer_ffn_bottleneck_dim > 0:
            dir_name += str(args.transformer_ffn_bottleneck_dim) + 'bn'
        dir_name += str(args.enc_n_layers) + 'L'
        dir_name += str(args.transformer_enc_n_heads) + 'H'
        dir_name += 'pe' + str(args.transformer_enc_pe_type)
        if args.transformer_enc_clamp_len > 0:
            dir_name += '_clamp' + str(args.transformer_enc_clamp_len)
        if args.dropout_enc_layer > 0:
            dir_name += '_LD' + str(args.dropout_enc_layer)
        if int(args.lc_chunk_size_left.split('_')[-1]) > 0 or int(args.lc_chunk_size_current.split('_')[-1]) > 0 \
                or int(args.lc_chunk_size_right.split('_')[-1]) > 0:
            dir_name += '_chunkL' + args.lc_chunk_size_left + 'C' + \
                args.lc_chunk_size_current + 'R' + args.lc_chunk_size_right
            dir_name += '_' + args.lc_type
        elif sum(list(map(int, args.transformer_enc_lookaheads.split('_')))) > 0:
            dir_name += '_LA' + str(sum(list(map(int, args.transformer_enc_lookaheads.split('_')))))
        return dir_name

    def reset_parameters(self, param_init):
        """Initialize parameters."""
        if param_init == 'xavier_uniform':
            logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
            if self.conv is None:
                nn.init.xavier_uniform_(self.embed.weight)
                nn.init.constant_(self.embed.bias, 0.)
            if self.bridge is not None:
                nn.init.xavier_uniform_(self.bridge.weight)
                nn.init.constant_(self.bridge.bias, 0.)
            if self.bridge_sub1 is not None:
                nn.init.xavier_uniform_(self.bridge_sub1.weight)
                nn.init.constant_(self.bridge_sub1.bias, 0.)
            if self.bridge_sub2 is not None:
                nn.init.xavier_uniform_(self.bridge_sub2.weight)
                nn.init.constant_(self.bridge_sub2.bias, 0.)
            if self.pe_type == 'relative_xl':
                nn.init.xavier_uniform_(self.u_bias)
                nn.init.xavier_uniform_(self.v_bias)

    def reset_cache(self):
        self.frontend_cache = None  # TODO
        self.cache = [None] * self.n_layers
        logger.debug('Reset cache.')

    def forward(self, xs, xlens, task, streaming=False,
                lookback=False, lookahead=False):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, input_dim]`
            xlens (InteTensor): `[B]` (on CPU)
            task (str): ys/ys_sub1/ys_sub2
            streaming (bool): streaming encoding
            lookback (bool): truncate leftmost frames for lookback in CNN context
            lookahead (bool): truncate rightmost frames for lookahead in CNN context
        Returns:
            eouts (dict):
                xs (FloatTensor): `[B, T, d_model]`
                xlens (InteTensor): `[B]` (on CPU)

        """
        eouts = {'ys': {'xs': None, 'xlens': None},
                 'ys_sub1': {'xs': None, 'xlens': None},
                 'ys_sub2': {'xs': None, 'xlens': None}}

        bs, xmax = xs.size()[:2]
        n_chunks = 0
        unidir = self.unidir
        lc_bidir = self.lc_bidir
        N_l, N_c, N_r = self.chunk_size_left, self.chunk_size_current, self.chunk_size_right

        if streaming and self.streaming_type == 'mask':
            assert xmax <= N_c
        elif streaming and self.streaming_type == 'reshape':
            assert xmax <= (N_l + N_c + N_r)

        if lc_bidir:
            if self.streaming_type == 'mask' and not streaming:
                xs = chunkwise(xs, 0, N_c, 0, padding=True)  # `[B * n_chunks, N_c, idim]`
                # NOTE: CNN consumes inputs in the current chunk to avoid extra lookahead latency
                # That is, CNN outputs are independent on chunk boundary
            elif self.streaming_type == 'reshape':
                xs = chunkwise(xs, N_l, N_c, N_r, padding=not streaming)  # `[B * n_chunks, N_l+N_c+N_r, idim]`
            n_chunks = xs.size(0) // bs
            assert bs * n_chunks == xs.size(0)
            if streaming:
                assert n_chunks == 1, xs.size()

        if self.conv is None:
            xs = self.embed(xs)
        else:
            # Path through CNN blocks
            xs, xlens = self.conv(xs, xlens,
                                  lookback=False if lc_bidir else lookback,
                                  lookahead=False if lc_bidir else lookahead)
            # NOTE: CNN lookahead surpassing a chunk is not allowed in chunkwise processing
            N_l = max(0, N_l // self.conv.subsampling_factor)
            N_c = N_c // self.conv.subsampling_factor
            N_r = N_r // self.conv.subsampling_factor

        if lc_bidir:
            # Do nothing in the streaming mode
            if self.streaming_type == 'mask' and not streaming:
                # back to the original shape (during training only)
                xs = xs.contiguous().view(bs, -1, xs.size(2))[:, :xlens.max()]  # `[B, emax, d_model]`
        elif streaming:
            xs = xs[:, :xlens.max()]  # for unidirectional

        if self.enc_type == 'conv':
            eouts['ys']['xs'] = xs
            eouts['ys']['xlens'] = xlens
            return eouts

        if not streaming:
            self.reset_cache()
        n_hist = self.cache[0]['input_san'].size(1) if streaming and self.cache[0] is not None else 0

        # positional encoding
        if self.pe_type in ['relative', 'relative_xl']:
            xs = xs * self.scale  # NOTE: first layer only
            rel_pos_embs = self.pos_emb(xs, mlen=n_hist)
        else:
            xs = self.pos_enc(xs, scale=True, offset=max(0, n_hist))
            rel_pos_embs = None

        new_cache = [None] * self.n_layers
        if lc_bidir:
            # chunkwise streaming encoder
            if self.streaming_type == 'reshape':
                xx_mask = None  # NOTE: no mask to avoid masking all frames in a chunk
            elif self.streaming_type == 'mask':
                if streaming:
                    n_chunks = math.ceil((xlens.max().item() + n_hist) / N_c)
                xx_mask = make_chunkwise_san_mask(xs, xlens + n_hist, N_l, N_c, n_chunks)

            for lth, layer in enumerate(self.layers):
                xs, cache = layer(xs, xx_mask, cache=self.cache[lth],
                                  pos_embs=rel_pos_embs, u_bias=self.u_bias, v_bias=self.v_bias)
                if self.streaming_type == 'mask':
                    new_cache[lth] = cache
                if not self.training and not streaming:
                    if self.streaming_type == 'reshape':
                        n_heads = layer.xx_aws.size(1)
                        xx_aws = layer.xx_aws[:, :, N_l:N_l + N_c, N_l:N_l + N_c]
                        xx_aws = xx_aws.view(bs, n_chunks, n_heads, N_c, N_c)
                        emax = xlens.max().item()
                        xx_aws_center = xx_aws.new_zeros(bs, n_heads, emax, emax)
                        for chunk_idx in range(n_chunks):
                            offset = chunk_idx * N_c
                            emax_chunk = xx_aws_center[:, :, offset:offset + N_c].size(2)
                            xx_aws_chunk = xx_aws[:, chunk_idx, :, :emax_chunk, :emax_chunk]
                            xx_aws_center[:, :, offset:offset + N_c, offset:offset + N_c] = xx_aws_chunk
                        self.aws_dict['xx_aws_layer%d' % lth] = tensor2np(xx_aws_center)
                    elif self.streaming_type == 'mask':
                        self.aws_dict['xx_aws_layer%d' % lth] = tensor2np(layer.xx_aws)
                    self.data_dict['elens%d' % lth] = tensor2np(xlens)

                if self.subsample is not None:
                    xs, xlens = self.subsample[lth](xs, xlens)
                    N_l = max(0, N_l // self.subsample[lth].factor)
                    N_c = N_c // self.subsample[lth].factor
                    N_r = N_r // self.subsample[lth].factor
                    if self.pe_type in ['relative', 'relative_xl']:
                        rel_pos_embs = self.pos_emb(xs)
                    if self.streaming_type == 'mask':
                        xx_mask = make_chunkwise_san_mask(xs, xlens, N_l, N_c, n_chunks)

            # Extract the center region
            if self.streaming_type == 'reshape':
                xs = xs[:, N_l:N_l + N_c]  # `[B * n_chunks, N_c, d_model]`
                xs = xs.contiguous().view(bs, -1, xs.size(2))
                xs = xs[:, :xlens.max()]

        else:
            xx_mask = make_san_mask(xs, xlens + n_hist, unidir, self.lookaheads[0])
            for lth, layer in enumerate(self.layers):
                xs, cache = layer(xs, xx_mask, cache=self.cache[lth],
                                  pos_embs=rel_pos_embs, u_bias=self.u_bias, v_bias=self.v_bias)
                new_cache[lth] = cache
                if not self.training and not streaming:
                    self.aws_dict['xx_aws_layer%d' % lth] = tensor2np(layer.xx_aws)
                    self.data_dict['elens%d' % lth] = tensor2np(xlens)

                # Pick up outputs in the sub task before the projection layer
                if lth == self.n_layers_sub1 - 1:
                    xs_sub1 = self.sub_module(xs, xx_mask, lth, rel_pos_embs, 'sub1')
                    xlens_sub1 = xlens.clone()
                    if task == 'ys_sub1':
                        eouts[task]['xs'], eouts[task]['xlens'] = xs_sub1, xlens_sub1
                        return eouts
                if lth == self.n_layers_sub2 - 1:
                    xs_sub2 = self.sub_module(xs, xx_mask, lth, rel_pos_embs, 'sub2')
                    xlens_sub2 = xlens.clone()
                    if task == 'ys_sub2':
                        eouts[task]['xs'], eouts[task]['xlens'] = xs_sub2, xlens_sub2
                        return eouts

                if lth < len(self.layers) - 1:
                    if self.subsample is not None and self.subsample[lth].factor > 1:
                        xs, xlens = self.subsample[lth](xs, xlens)
                        n_hist = self.cache[lth + 1]['input_san'].size(
                            1) if streaming and self.cache[lth + 1] is not None else 0
                        if self.pe_type in ['relative', 'relative_xl']:
                            rel_pos_embs = self.pos_emb(xs, mlen=n_hist)
                        xx_mask = make_san_mask(xs, xlens + n_hist, unidir, self.lookaheads[lth + 1])
                    elif self.lookaheads[lth] != self.lookaheads[lth + 1]:
                        xx_mask = make_san_mask(xs, xlens + n_hist, unidir, self.lookaheads[lth + 1])

        xs = self.norm_out(xs)

        if streaming:
            self.cache = new_cache

        # Bridge layer
        if self.bridge is not None:
            xs = self.bridge(xs)

        if task in ['all', 'ys']:
            eouts['ys']['xs'], eouts['ys']['xlens'] = xs, xlens
        if self.n_layers_sub1 >= 1 and task == 'all':
            eouts['ys_sub1']['xs'], eouts['ys_sub1']['xlens'] = xs_sub1, xlens_sub1
        if self.n_layers_sub2 >= 1 and task == 'all':
            eouts['ys_sub2']['xs'], eouts['ys_sub2']['xlens'] = xs_sub2, xlens_sub2
        return eouts

    def sub_module(self, xs, xx_mask, lth, pos_embs=None, module='sub1'):
        if self.task_specific_layer:
            xs_sub, cache = getattr(self, 'layer_' + module)(xs, xx_mask, pos_embs=pos_embs)
        else:
            xs_sub = xs.clone()
        if getattr(self, 'bridge_' + module) is not None:
            xs_sub = getattr(self, 'bridge_' + module)(xs_sub)
        if getattr(self, 'norm_out_' + module) is not None:
            xs_sub = getattr(self, 'norm_out_' + module)(xs_sub)
        if not self.training:
            self.aws_dict['xx_aws_%s_layer%d' % (module, lth)] = tensor2np(getattr(self, 'layer_' + module).xx_aws)
        return xs_sub


def make_san_mask(xs, xlens, unidirectional=False, lookahead=0):
    """Mask self-attention mask.

    Args:
        xs (FloatTensor): `[B, T, d_model]`
        xlens (InteTensor): `[B]` (on CPU)
        unidirectional (bool): pad future context
        lookahead (int): lookahead frame
    Returns:
        xx_mask (ByteTensor): `[B, T (query), T (key)]`

    """
    xx_mask = make_pad_mask(xlens.to(xs.device))
    xx_mask = xx_mask.unsqueeze(1).repeat([1, xlens.max(), 1])  # `[B, emax (query), emax (key)]`
    if unidirectional:
        xx_mask = causal(xx_mask, lookahead)
    return xx_mask


def causal(xx_mask, lookahead):
    """Causal masking.

    Args:
        xx_mask (ByteTensor): `[B, T (query), T (key)]`
        lookahead (int): lookahead frame
    Returns:
        xx_mask (ByteTensor): `[B, T (query), T (key)]`

    """
    causal_mask = xx_mask.new_ones(xx_mask.size(1), xx_mask.size(1), dtype=xx_mask.dtype)
    if torch_12_plus:
        causal_mask = causal_mask.byte()
    causal_mask = torch.tril(causal_mask, diagonal=lookahead, out=causal_mask).unsqueeze(0)
    xx_mask = xx_mask & causal_mask  # `[B, L (query), L (key)]`
    return xx_mask


def make_chunkwise_san_mask(xs, xlens, N_l, N_c, n_chunks):
    """Mask self-attention mask for chunkwise processing.

    Args:
        xs (FloatTensor): `[B, T, d_model]`
        xlens (InteTensor): `[B]` (on CPU)
        N_l (int): number of frames for left context
        N_c (int): number of frames for current context
        n_chunks (int): number of chunks
    Returns:
        xx_mask (ByteTensor): `[B, T (query), T (key)]`

    """
    xx_mask = make_san_mask(xs, xlens)
    for chunk_idx in range(n_chunks):
        offset = chunk_idx * N_c
        xx_mask[:, offset:offset + N_c, :max(0, offset - N_l)] = 0
        xx_mask[:, offset:offset + N_c, offset + N_c:] = 0
    return xx_mask
