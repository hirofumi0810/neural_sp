# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Convolution block for Conformer encoder."""

import logging
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.modules.initialization import init_with_lecun_normal
from neural_sp.models.modules.initialization import init_with_xavier_uniform
from neural_sp.models.modules.swish import Swish

logger = logging.getLogger(__name__)


class ConformerConvBlock(nn.Module):
    """A single convolution block for the Conformer encoder.

    Args:
        d_model (int): input/output dimension
        kernel_size (int): kernel size in depthwise convolution
        param_init (str): parameter initialization method
        normalization (str): "batch_norm" or "group_norm"

    """

    def __init__(self, d_model, kernel_size, param_init, causal=False,
                 normalization='batch_norm'):

        super().__init__()

        assert (kernel_size - 1) % 2 == 0, 'kernel_size must be the odd number.'

        self.kernel_size = kernel_size
        if causal:
            self.causal = nn.ConstantPad1d((kernel_size - 1, 0), 0)
        else:
            self.causal = None

        self.pointwise_conv1 = nn.Conv1d(in_channels=d_model,
                                         out_channels=d_model * 2,  # for GLU
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        self.depthwise_conv = nn.Conv1d(in_channels=d_model,
                                        out_channels=d_model,
                                        kernel_size=kernel_size,
                                        stride=1,
                                        padding=(kernel_size - 1) // 2,
                                        groups=d_model)  # depthwise
        if normalization == 'batch_norm':
            self.norm = nn.BatchNorm1d(d_model)
        elif normalization == 'group_norm':
            self.norm = nn.GroupNorm(num_groups=max(1, d_model // 16),
                                     num_channels=d_model)
        else:
            raise NotImplementedError
        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(in_channels=d_model,
                                         out_channels=d_model,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)

        if param_init == 'xavier_uniform':
            self.reset_parameters_xavier_uniform()
        elif param_init == 'lecun':
            self.reset_parameters_lecun()
        else:
            logger.info('Parameter initialization is skipped.')

    def reset_parameters_xavier_uniform(self):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
        for layer in [self.pointwise_conv1, self.pointwise_conv2, self.depthwise_conv]:
            for n, p in layer.named_parameters():
                init_with_xavier_uniform(n, p)

    def reset_parameters_lecun(self, param_init=0.1):
        """Initialize parameters with lecun style.."""
        logger.info('===== Initialize %s with lecun style =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            init_with_lecun_normal(n, p, param_init)

    def forward(self, xs):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, d_model]`
        Returns:
            xs (FloatTensor): `[B, T, d_model]`

        """
        xs = xs.transpose(2, 1).contiguous()  # `[B, C, T]`
        if self.causal is not None:
            xs = self.causal(xs)
            xs = xs[:, :, :-(self.kernel_size - 1)]
        xs = self.pointwise_conv1(xs)  # `[B, 2 * C, T]`
        xs = xs.transpose(2, 1)  # `[B, T, 2 * C]`
        xs = F.glu(xs)  # `[B, T, C]`
        xs = xs.transpose(2, 1).contiguous()  # `[B, C, T]`
        xs = self.depthwise_conv(xs)  # `[B, C, T]`

        xs = self.norm(xs)
        xs = self.activation(xs)
        xs = self.pointwise_conv2(xs)  # `[B, C, T]`

        xs = xs.transpose(2, 1).contiguous()  # `[B, T, C]`
        return xs
