# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer encoder block."""

import logging
import random
import torch
import torch.nn as nn

from neural_sp.models.modules.conformer_convolution import ConformerConvBlock
from neural_sp.models.modules.positionwise_feed_forward import PositionwiseFeedForward as FFN
from neural_sp.models.modules.relative_multihead_attention import RelativeMultiheadAttentionMechanism as RelMHA

random.seed(1)

logger = logging.getLogger(__name__)


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
        ffn_activation (str): nonlinear function for PositionwiseFeedForward
        param_init (str): parameter initialization method
        pe_type (str): type of positional encoding
        clamp_len (int): maximum relative distance from each position
        ffn_bottleneck_dim (int): bottleneck dimension for the light-weight FFN layer
        unidirectional (bool): pad right context for unidirectional encoding
        normalization (str): batch_norm/group_norm/layer_norm

    """

    def __init__(self, d_model, d_ff, n_heads, kernel_size,
                 dropout, dropout_att, dropout_layer,
                 layer_norm_eps, ffn_activation, param_init,
                 pe_type, clamp_len, ffn_bottleneck_dim, unidirectional,
                 normalization='batch_norm'):
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
                                xl_like=pe_type == 'relative_xl',
                                clamp_len=clamp_len)

        # conv module
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.conv = ConformerConvBlock(d_model, kernel_size, param_init, normalization,
                                       causal=unidirectional)
        self.conv_context = kernel_size

        # second half position-wise feed-forward
        self.norm4 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = FFN(d_model, d_ff, dropout, ffn_activation, param_init,
                                ffn_bottleneck_dim)

        self.norm5 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout = nn.Dropout(dropout)
        self.dropout_layer = dropout_layer  # probability to skip
        logger.info('Stochastic depth prob: %.3f' % dropout_layer)

        self.reset_visualization()

    @property
    def xx_aws(self):
        return self._xx_aws

    def reset_visualization(self):
        self._xx_aws = None

    def forward(self, xs, xx_mask=None, cache=None,
                pos_embs=None, u_bias=None, v_bias=None):
        """Conformer encoder layer definition.

        Args:
            xs (FloatTensor): `[B, T (query), d_model]`
            xx_mask (ByteTensor): `[B, T (query), T (key)]`
            cache (dict):
                input_san: `[B, n_hist, d_model]`
                input_conv: `[B, n_hist, d_model]`
                output: `[B, n_hist, d_model]`
            pos_embs (LongTensor): `[T (query), 1, d_model]`
            u_bias (FloatTensor): global parameter for relative positional encoding
            v_bias (FloatTensor): global parameter for relative positional encoding
        Returns:
            xs (FloatTensor): `[B, T (query), d_model]`
            new_cache (dict):
                input_san: `[B, n_hist+T, d_model]`
                input_conv: `[B, n_hist+T, d_model]`
                output: `[B, T (query), d_model]`

        """
        self.reset_visualization()
        new_cache = {}
        qlen = xs.size(1)

        # LayerDrop
        if self.dropout_layer > 0:
            if self.training and random.random() < self.dropout_layer:
                return xs, new_cache
            else:
                xs = xs / (1 - self.dropout_layer)

        ##################################################
        # first half FFN
        ##################################################
        residual = xs  # `[B, qlen, d_model]`
        xs = self.norm1(xs)  # pre-norm
        xs = self.feed_forward_macaron(xs)
        xs = self.fc_factor * self.dropout(xs) + residual  # Macaron FFN

        ##################################################
        # self-attention w/ relative positional encoding
        ##################################################
        residual = xs  # `[B, qlen, d_model]`
        xs = self.norm2(xs)  # pre-norm

        # cache for self-attention
        if cache is not None:
            xs = torch.cat([cache['input_san'], xs], dim=1)
        new_cache['input_san'] = xs

        xs_kv = xs
        if cache is not None:
            xs = xs[:, -qlen:]
            residual = residual[:, -qlen:]  # `[B, qlen, d_model]`
            xx_mask = xx_mask[:, -qlen:]

        xs, self._xx_aws = self.self_attn(xs_kv, xs, pos_embs, xx_mask, u_bias, v_bias)  # k/q/m
        xs = self.dropout(xs) + residual

        ##################################################
        # conv
        ##################################################
        residual = xs  # `[B, qlen, d_model]`
        xs = self.norm3(xs)  # pre-norm

        # cache for convolution
        if cache is not None:
            xs = torch.cat([cache['input_conv'], xs], dim=1)
        new_cache['input_conv'] = xs
        # restrict to kernel size
        if cache is not None:
            xs = xs[:, -(self.conv_context + qlen - 1):]

        xs = self.conv(xs)
        if cache is not None:
            xs = xs[:, -qlen:]
        xs = self.dropout(xs) + residual

        ##################################################
        # second half FFN
        ##################################################
        residual = xs  # `[B, qlen, d_model]`
        xs = self.norm4(xs)  # pre-norm
        xs = self.feed_forward(xs)
        xs = self.fc_factor * self.dropout(xs) + residual  # Macaron FFN
        xs = self.norm5(xs)  # this is important for performance

        new_cache['output'] = xs

        return xs, new_cache
