#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for encoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import shutil
import torch

from neural_sp.models.base import ModelBase
from neural_sp.utils import mkdir_join

import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)


class EncoderBase(ModelBase):
    """Base class for encoders."""

    def __init__(self):

        super(ModelBase, self).__init__()
        logger.info('Overriding EncoderBase class.')

    @property
    def device_id(self):
        return torch.cuda.device_of(next(self.parameters()).data).idx

    @property
    def output_dim(self):
        return self._odim

    @property
    def subsampling_factor(self):
        return self._factor

    def reset_parameters(self, param_init):
        raise NotImplementedError

    def forward(self, xs, xlens, task):
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

    def _plot_attention(self, save_path, n_cols=2):
        """Plot attention for each head in all encoder layers."""
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator

        _save_path = mkdir_join(save_path, 'enc_att_weights')

        # Clean directory
        if _save_path is not None and os.path.isdir(_save_path):
            shutil.rmtree(_save_path)
            os.mkdir(_save_path)

        elens = self.data_dict['elens']

        for k, aw in self.aws_dict.items():
            plt.clf()
            n_heads = aw.shape[1]
            n_cols_tmp = 1 if n_heads == 1 else n_cols
            fig, axes = plt.subplots(max(1, n_heads // n_cols_tmp), n_cols_tmp,
                                     figsize=(20, 8), squeeze=False)
            for h in range(n_heads):
                ax = axes[h // n_cols_tmp, h % n_cols_tmp]
                ax.imshow(aw[-1, h, :elens[-1], :elens[-1]], aspect="auto")
                ax.grid(False)
                ax.set_xlabel("Input (head%d)" % h)
                ax.set_ylabel("Output (head%d)" % h)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            fig.tight_layout()
            fig.savefig(os.path.join(_save_path, '%s.png' % k), dvi=500)
            plt.close()


def blockwise(xs, N_l, N_c, N_r):
    bs, xmax, idim = xs.size()

    n_blocks = xmax // N_c
    if xmax % N_c != 0:
        n_blocks += 1
    xs_tmp = xs.new_zeros(bs, n_blocks, N_l + N_c + N_r, idim)
    xs_pad = torch.cat([xs.new_zeros(bs, N_l, idim),
                        xs,
                        xs.new_zeros(bs, N_r, idim)], dim=1)
    for blc_id, t in enumerate(range(N_l, N_l + xmax, N_c)):
        xs_chunk = xs_pad[:, t - N_l:t + (N_c + N_r)]
        xs_tmp[:, blc_id, :xs_chunk.size(1), :] = xs_chunk
    xs = xs_tmp.view(bs * n_blocks, N_l + N_c + N_r, idim)

    return xs
