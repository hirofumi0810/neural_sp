#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for all models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch
import torch.nn as nn

np.random.seed(1)

logger = logging.getLogger(__name__)


class ModelBase(nn.Module):
    """A base class for all models. All models have to inherit this class."""

    def __init__(self, *args, **kwargs):

        super().__init__()
        logger.info('Overriding ModelBase class.')

    @property
    def torch_version(self):
        return float('.'.join(torch.__version__.split('.')[:2]))

    @property
    def num_params_dict(self):
        if not hasattr(self, '_nparams_dict'):
            self._nparams_dict = {}
            for n, p in self.named_parameters():
                self._nparams_dict[n] = p.view(-1).size(0)
        return self._nparams_dict

    @property
    def total_parameters(self):
        if not hasattr(self, '_nparams'):
            self._nparams = 0
            for n, p in self.named_parameters():
                self._nparams += p.view(-1).size(0)
        return self._nparams

    @property
    def use_cuda(self):
        return torch.cuda.is_available()

    @property
    def device_id(self):
        return torch.cuda.device_of(next(self.parameters())).idx

    def init_forget_gate_bias_with_one(self):
        """Initialize bias in forget gate with 1. See detail in

            https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745

        """
        for n, p in self.named_parameters():
            if p.dim() == 1 and 'bias_ih' in n:
                dim = p.size(0)
                start, end = dim // 4, dim // 2
                p.data[start:end].fill_(1.)
                logger.info('Initialize %s with 1 (bias in forget gate)' % (n))

    def set_cuda(self, deterministic=True, benchmark=False):
        """Set model to the GPU version.

        Args:
            deterministic (bool):
            benchmark (bool):

        """
        if self.use_cuda:
            if benchmark:
                torch.backends.cudnn.benchmark = True
                logger.info('GPU mode (benchmark)')
            elif deterministic:
                logger.info('GPU deterministic mode (no cudnn)')
                torch.backends.cudnn.enabled = False
                # NOTE: this is slower than GPU mode.
            else:
                logger.info('GPU mode')
            self = self.cuda(self.device_id)
        else:
            logger.warning('CPU mode')
