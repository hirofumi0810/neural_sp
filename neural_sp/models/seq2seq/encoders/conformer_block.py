# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer encoder block."""

import logging
import random
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

    def forward(self, xs, xx_mask=None,
                pos_embs=None, u_bias=None, v_bias=None):
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
        xs, self._xx_aws = self.self_attn(xs, xs, pos_embs, xx_mask, u_bias, v_bias)  # k/q/m
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
