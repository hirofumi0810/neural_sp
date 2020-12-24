# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Dilated causal convolution."""

import logging
import torch.nn as nn

from neural_sp.models.modules.initialization import init_with_lecun_normal
from neural_sp.models.modules.initialization import init_with_xavier_uniform

logger = logging.getLogger(__name__)


class CausalConv1d(nn.Module):
    """1D dilated causal convolution.

    Args:
        in_channels (int): input channel size
        out_channels (int): output channel size
        kernel_size (int): kernel size
        dilation (int): deletion rate
        param_init (str): parameter initialization method

    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                 param_init=''):

        super().__init__()

        self.padding = (kernel_size - 1) * dilation
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size,
                                padding=self.padding, dilation=dilation)

        if param_init == 'xavier_uniform':
            self.reset_parameters_xavier_uniform()
        elif param_init == 'lecun':
            self.reset_parameters_lecun()
        else:
            logger.info('Parameter initialization is skipped.')

    def reset_parameters_xavier_uniform(self):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            init_with_xavier_uniform(n, p)

    def reset_parameters_lecun(self, param_init=0.1):
        """Initialize parameters with lecun style.."""
        logger.info('===== Initialize %s with lecun style =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            init_with_lecun_normal(n, p, param_init)

    def forward(self, xs):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, C_in]`
        Returns:
            xs (FloatTensor): `[B, T, C_out]`

        """
        xs = xs.transpose(2, 1)
        xs = self.conv1d(xs)
        if self.padding != 0:
            xs = xs[:, :, :-self.padding]
        xs = xs.transpose(2, 1).contiguous()
        return xs
