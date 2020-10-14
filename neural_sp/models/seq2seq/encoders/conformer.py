# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer encoder."""

import copy
import logging
import random
import torch.nn as nn

from neural_sp.models.modules.conformer_convolution import ConformerConvBlock
from neural_sp.models.modules.positionwise_feed_forward import PositionwiseFeedForward as FFN
from neural_sp.models.modules.relative_multihead_attention import RelativeMultiheadAttentionMechanism as RelMHA
from neural_sp.models.seq2seq.encoders.conv import ConvEncoder
from neural_sp.models.seq2seq.encoders.transformer import TransformerEncoder

random.seed(1)

logger = logging.getLogger(__name__)


class ConformerEncoder(TransformerEncoder):
    """Conformer encoder.

    Args:
        input_dim (int): dimension of input features (freq * channel)
        enc_type (str): type of encoder
        n_heads (int): number of heads for multi-head attention
        kernel_size (int): kernel size for depthwise convolution in convolution module
        n_layers (int): number of blocks
        n_layers_sub1 (int): number of layers in the 1st auxiliary task
        n_layers_sub2 (int): number of layers in the 2nd auxiliary task
        d_model (int): dimension of MultiheadAttentionMechanism
        d_ff (int): dimension of PositionwiseFeedForward
        ffn_bottleneck_dim (int): bottleneck dimension for the light-weight FFN layer
        ffn_activation (str): nonolinear function for PositionwiseFeedForward
        pe_type (str): type of positional encoding
        layer_norm_eps (float): epsilon value for layer normalization
        last_proj_dim (int): dimension of the last projection layer
        dropout_in (float): dropout probability for input-hidden connection
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        dropout_layer (float): LayerDrop probability for layers
        subsample (list): subsample in the corresponding Conformer layers
            ex.) [1, 2, 2, 1] means that subsample is conducted in the 2nd and 3rd layers.
        subsample_type (str): drop/concat/max_pool/1dconv
        n_stacks (int): number of frames to stack
        n_splices (int): frames to splice. Default is 1 frame.
        conv_in_channel (int): number of channels of input features
        conv_channels (int): number of channles in CNN blocks
        conv_kernel_sizes (list): size of kernels in CNN blocks
        conv_strides (list): number of strides in CNN blocks
        conv_poolings (list): size of poolings in CNN blocks
        conv_batch_norm (bool): apply batch normalization only in CNN blocks
        conv_layer_norm (bool): apply layer normalization only in CNN blocks
        conv_bottleneck_dim (int): dimension of the bottleneck layer between CNN and self-attention layers
        conv_param_init (float): only for CNN layers before Conformer layers
        task_specific_layer (bool): add a task specific layer for each sub task
        param_init (str): parameter initialization method
        clamp_len (int): maximum length for relative positional encoding
        lookahead (int): lookahead frames per layer for unidirectional Transformer encoder
        chunk_size_left (int): left chunk size for latency-controlled Conformer encoder
        chunk_size_current (int): current chunk size for latency-controlled Conformer encoder
        chunk_size_right (int): right chunk size for latency-controlled Conformer encoder
        streaming_type (str): implementation methods of latency-controlled Conformer encoder

    """

    def __init__(self, input_dim, enc_type, n_heads, kernel_size,
                 n_layers, n_layers_sub1, n_layers_sub2,
                 d_model, d_ff, ffn_bottleneck_dim, ffn_activation,
                 pe_type, layer_norm_eps, last_proj_dim,
                 dropout_in, dropout, dropout_att, dropout_layer,
                 subsample, subsample_type, n_stacks, n_splices,
                 conv_in_channel, conv_channels, conv_kernel_sizes, conv_strides, conv_poolings,
                 conv_batch_norm, conv_layer_norm, conv_bottleneck_dim, conv_param_init,
                 task_specific_layer, param_init, clamp_len,
                 lookahead, chunk_size_left, chunk_size_current, chunk_size_right, streaming_type):

        super(ConformerEncoder, self).__init__(
            input_dim, enc_type, n_heads,
            n_layers, n_layers_sub1, n_layers_sub2,
            d_model, d_ff, ffn_bottleneck_dim, ffn_activation,
            pe_type, layer_norm_eps, last_proj_dim,
            dropout_in, dropout, dropout_att, dropout_layer,
            subsample, subsample_type, n_stacks, n_splices,
            conv_in_channel, conv_channels, conv_kernel_sizes, conv_strides, conv_poolings,
            conv_batch_norm, conv_layer_norm, conv_bottleneck_dim, conv_param_init,
            task_specific_layer, param_init, clamp_len,
            lookahead, chunk_size_left, chunk_size_current, chunk_size_right, streaming_type)

        self.layers = nn.ModuleList([copy.deepcopy(ConformerEncoderBlock(
            d_model, d_ff, n_heads, kernel_size, dropout, dropout_att, dropout_layer,
            layer_norm_eps, ffn_activation, param_init, pe_type,
            ffn_bottleneck_dim, self.unidir))
            for _ in range(n_layers)])

        if n_layers_sub1 > 0:
            if task_specific_layer:
                self.layer_sub1 = ConformerEncoderBlock(
                    d_model, d_ff, n_heads, kernel_size, dropout, dropout_att, dropout_layer,
                    layer_norm_eps, ffn_activation, param_init, pe_type,
                    ffn_bottleneck_dim, self.unidir)

        if n_layers_sub2 > 0:
            if task_specific_layer:
                self.layer_sub2 = ConformerEncoderBlock(
                    d_model, d_ff, n_heads, kernel_size, dropout, dropout_att, dropout_layer,
                    layer_norm_eps, ffn_activation, param_init, pe_type,
                    ffn_bottleneck_dim, self.unidir)

        self.reset_parameters(param_init)

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
            group.add_argument('--transformer_layer_norm_eps', type=float, default=1e-12,
                               help='epsilon value for layer normalization')
            group.add_argument('--transformer_ffn_activation', type=str, default='relu',
                               choices=['relu', 'gelu', 'gelu_accurate', 'glu', 'swish'],
                               help='nonlinear activation for the FFN layer')
            group.add_argument('--transformer_param_init', type=str, default='xavier_uniform',
                               choices=['xavier_uniform', 'pytorch'],
                               help='parameter initialization')
        # NOTE: These checks are important to avoid conflict with args in Transformer decoder

        # Conformer encoder specific
        group.add_argument('--transformer_enc_d_model', type=int, default=256,
                           help='number of units in the MHA layer for Conformer encoder')
        group.add_argument('--transformer_enc_d_ff', type=int, default=2048,
                           help='number of units in the FFN layer for Conformer encoder')
        group.add_argument('--transformer_enc_n_heads', type=int, default=4,
                           help='number of heads in the MHA layer for Conformer encoder')
        group.add_argument('--transformer_enc_pe_type', type=str, default='relative',
                           choices=['relative', 'relative_xl'],
                           help='type of positional encoding for Conformer encoder')
        group.add_argument('--conformer_kernel_size', type=int, default=31,
                           help='kernel size for depthwise convolution in convolution module for Conformer encoder')
        group.add_argument('--dropout_enc_layer', type=float, default=0.0,
                           help='LayerDrop probability for Conformer encoder layers')
        group.add_argument('--transformer_enc_clamp_len', type=int, default=-1,
                           help='maximum length for relative positional encoding. -1 means infinite length.')
        # streaming
        group.add_argument('--transformer_enc_lookaheads', type=str, default="0_0_0_0_0_0_0_0_0_0_0_0",
                           help='lookahead frames per layer for unidirectional Conformer encoder')
        group.add_argument('--lc_chunk_size_left', type=str, default="0",
                           help='left chunk size for latency-controlled Conformer encoder')
        group.add_argument('--lc_chunk_size_current', type=str, default="0",
                           help='current chunk size (and hop size) for latency-controlled Conformer encoder')
        group.add_argument('--lc_chunk_size_right', type=str, default="0",
                           help='right chunk size for latency-controlled Conformer encoder')
        group.add_argument('--lc_type', type=str, default='reshape',
                           choices=['reshape', 'mask'],
                           help='implementation methods of latency-controlled Conformer encoder')
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
        dir_name += 'kernel' + str(args.conformer_kernel_size)
        if args.transformer_enc_clamp_len > 0:
            dir_name += '_clamp' + str(args.transformer_enc_clamp_len)
        if args.dropout_enc_layer > 0:
            dir_name += 'droplayer' + str(args.dropout_enc_layer)
        if int(args.lc_chunk_size_left.split('_')[-1]) > 0 or int(args.lc_chunk_size_current.split('_')[-1]) > 0 \
                or int(args.lc_chunk_size_right.split('_')[-1]) > 0:
            dir_name += '_chunkL' + args.lc_chunk_size_left + 'C' + \
                args.lc_chunk_size_current + 'R' + args.lc_chunk_size_right
            dir_name += '_' + args.lc_type
        elif sum(list(map(int, args.transformer_enc_lookaheads.split('_')))) > 0:
            dir_name += '_LA' + str(sum(list(map(int, args.transformer_enc_lookaheads.split('_')))))

        return dir_name


class ConformerEncoderBlock(nn.Module):
    """A single layer of the Conformer encoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        n_heads (int): number of heads for multi-head attention
        kernel_size (int): kernel size for depthwise convolution in convolution module
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        dropout_layer (float): LayerDrop probability
        layer_norm_eps (float): epsilon parameter for layer normalization
        ffn_activation (str): nonolinear function for PositionwiseFeedForward
        param_init (str): parameter initialization method
        pe_type (str): type of positional encoding
        ffn_bottleneck_dim (int): bottleneck dimension for the light-weight FFN layer
        unidirectional (bool): pad right context for unidirectional encoding

    """

    def __init__(self, d_model, d_ff, n_heads, kernel_size,
                 dropout, dropout_att, dropout_layer,
                 layer_norm_eps, ffn_activation, param_init, pe_type,
                 ffn_bottleneck_dim, unidirectional):
        super(ConformerEncoderBlock, self).__init__()

        self.n_heads = n_heads
        self.fc_factor = 0.5

        # first half position-wise feed-forward
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward_macaron = FFN(d_model, d_ff, dropout, ffn_activation, param_init,
                                        ffn_bottleneck_dim)

        # self-attention
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.self_attn = RelMHA(kdim=d_model,
                                qdim=d_model,
                                adim=d_model,
                                odim=d_model,
                                n_heads=n_heads,
                                dropout=dropout_att,
                                param_init=param_init,
                                xl_like=pe_type == 'relative_xl')

        # conv module
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.conv = ConformerConvBlock(d_model, kernel_size, param_init,
                                       causal=unidirectional)

        # second half position-wise feed-forward
        self.norm4 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = FFN(d_model, d_ff, dropout, ffn_activation, param_init,
                                ffn_bottleneck_dim)
        self.norm5 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout = nn.Dropout(dropout)
        self.dropout_layer = dropout_layer

        self.reset_visualization()

    @property
    def xx_aws(self):
        return self._xx_aws

    def reset_visualization(self):
        self._xx_aws = None

    def forward(self, xs, xx_mask=None, pos_embs=None, u_bias=None, v_bias=None):
        """Conformer encoder layer definition.

        Args:
            xs (FloatTensor): `[B, T, d_model]`
            xx_mask (ByteTensor): `[B, T (query), T (key)]`
            pos_embs (LongTensor): `[L, 1, d_model]`
            u_bias (FloatTensor): global parameter for relative positional encoding
            v_bias (FloatTensor): global parameter for relative positional encoding
        Returns:
            xs (FloatTensor): `[B, T, d_model]`

        """
        self.reset_visualization()

        # LayerDrop
        if self.dropout_layer > 0 and self.training and random.random() < self.dropout_layer:
            return xs

        # first half FFN
        residual = xs
        xs = self.norm1(xs)
        xs = self.feed_forward_macaron(xs)
        xs = self.fc_factor * self.dropout(xs) + residual  # Macaron FFN

        # self-attention w/ relative positional encoding
        residual = xs
        xs = self.norm2(xs)
        xs, self._xx_aws = self.self_attn(xs, xs, pos_embs, xx_mask, u_bias, v_bias)
        xs = self.dropout(xs) + residual

        # conv
        residual = xs
        xs = self.norm3(xs)
        xs = self.conv(xs)
        xs = self.dropout(xs) + residual

        # second half FFN
        residual = xs
        xs = self.norm4(xs)
        xs = self.feed_forward(xs)
        xs = self.fc_factor * self.dropout(xs) + residual  # Macaron FFN
        xs = self.norm5(xs)  # this is important for performance

        return xs
