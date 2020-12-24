# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Gated Linear Units (GLU) block."""

from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F


class LinearGLUBlock(nn.Module):
    """A linear GLU block.

    Args:
        idim (int): input and output dimension

    """

    def __init__(self, idim):

        super().__init__()

        self.fc = nn.Linear(idim, idim * 2)

    def forward(self, xs):
        return F.glu(self.fc(xs), dim=-1)


class ConvGLUBlock(nn.Module):
    """A convolutional GLU block.

    Args:
        kernel_size (int): kernel size
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        bottlececk_dim (int): dimension of the bottleneck layers for computational efficiency
        dropout (float): dropout probability

    """

    def __init__(self, kernel_size, in_ch, out_ch, bottlececk_dim=0, dropout=0.):

        super().__init__()

        self.conv_residual = None
        if in_ch != out_ch:
            self.conv_residual = nn.utils.weight_norm(
                nn.Conv2d(in_channels=in_ch,
                          out_channels=out_ch,
                          kernel_size=(1, 1)), name='weight', dim=0)
            self.dropout_residual = nn.Dropout(p=dropout)

        self.pad_left = nn.ConstantPad2d((0, 0, kernel_size - 1, 0), 0)

        layers = OrderedDict()
        if bottlececk_dim == 0:
            layers['conv'] = nn.utils.weight_norm(
                nn.Conv2d(in_channels=in_ch,
                          out_channels=out_ch * 2,
                          kernel_size=(kernel_size, 1)), name='weight', dim=0)
            # TODO(hirofumi0810): padding?
            layers['dropout'] = nn.Dropout(p=dropout)
            layers['glu'] = nn.GLU()

        elif bottlececk_dim > 0:
            layers['conv_in'] = nn.utils.weight_norm(
                nn.Conv2d(in_channels=in_ch,
                          out_channels=bottlececk_dim,
                          kernel_size=(1, 1)), name='weight', dim=0)
            layers['dropout_in'] = nn.Dropout(p=dropout)
            layers['conv_bottleneck'] = nn.utils.weight_norm(
                nn.Conv2d(in_channels=bottlececk_dim,
                          out_channels=bottlececk_dim,
                          kernel_size=(kernel_size, 1)), name='weight', dim=0)
            layers['dropout'] = nn.Dropout(p=dropout)
            layers['glu'] = nn.GLU()
            layers['conv_out'] = nn.utils.weight_norm(
                nn.Conv2d(in_channels=bottlececk_dim,
                          out_channels=out_ch * 2,
                          kernel_size=(1, 1)), name='weight', dim=0)
            layers['dropout_out'] = nn.Dropout(p=dropout)

        self.layers = nn.Sequential(layers)

    def forward(self, xs):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, in_ch, T, feat_dim]`
        Returns:
            out (FloatTensor): `[B, out_ch, T, feat_dim]`

        """
        residual = xs
        if self.conv_residual is not None:
            residual = self.dropout_residual(self.conv_residual(residual))
        xs = self.pad_left(xs)  # `[B, embed_dim, T+kernel-1, 1]`
        xs = self.layers(xs)  # `[B, out_ch * 2, T ,1]`
        xs = xs + residual
        return xs
