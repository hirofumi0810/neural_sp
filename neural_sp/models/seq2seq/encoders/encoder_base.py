#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for encoders."""

import logging
import os
import shutil
import torch

from neural_sp.models.base import ModelBase

import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)


class EncoderBase(ModelBase):
    """Base class for encoders."""

    def __init__(self):

        super(ModelBase, self).__init__()
        logger.info('Overriding EncoderBase class.')

    @property
    def output_dim(self):
        return self._odim

    @property
    def subsampling_factor(self):
        return self._factor

    def forward(self, xs, xlens, task):
        raise NotImplementedError

    def reset_cache(self):
        raise NotImplementedError

    def turn_on_ceil_mode(self, encoder):
        if isinstance(encoder, torch.nn.Module):
            for name, module in encoder.named_children():
                if isinstance(module, torch.nn.MaxPool2d):
                    module.ceil_mode = True
                    logging.debug('Turn ON ceil_mode in %s.' % name)
                else:
                    self.turn_on_ceil_mode(module)

    def turn_off_ceil_mode(self, encoder):
        if isinstance(encoder, torch.nn.Module):
            for name, module in encoder.named_children():
                if isinstance(module, torch.nn.MaxPool2d):
                    module.ceil_mode = False
                    logging.debug('Turn OFF ceil_mode in %s.' % name)
                else:
                    self.turn_off_ceil_mode(module)

    def _plot_attention(self, save_path=None, n_cols=2):
        """Plot attention for each head in all encoder layers."""
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator

        # Clean directory
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)

        if not hasattr(self, 'aws_dict'):
            return

        for k, aw in self.aws_dict.items():
            lth = k.split('_')[-1].replace('layer', '')
            elens_l = self.data_dict['elens' + lth]

            plt.clf()
            n_heads = aw.shape[1]
            n_cols_tmp = 1 if n_heads == 1 else n_cols
            fig, axes = plt.subplots(max(1, n_heads // n_cols_tmp), n_cols_tmp,
                                     figsize=(20, 8), squeeze=False)
            for h in range(n_heads):
                ax = axes[h // n_cols_tmp, h % n_cols_tmp]
                ax.imshow(aw[-1, h, :elens_l[-1], :elens_l[-1]], aspect="auto")
                ax.grid(False)
                ax.set_xlabel("Input (head%d)" % h)
                ax.set_ylabel("Output (head%d)" % h)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            fig.tight_layout()
            if save_path is not None:
                fig.savefig(os.path.join(save_path, '%s.png' % k), dvi=500)
            plt.close()
