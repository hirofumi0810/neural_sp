#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for decoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch

from neural_sp.models.base import ModelBase


class DecoderBase(ModelBase):
    """Base class for decoders."""

    def __init__(self):

        super(ModelBase, self).__init__()
        logger = logging.getLogger('training')
        logger.info('Overriding DecoderBase class.')

    @property
    def device_id(self):
        return torch.cuda.device_of(next(self.parameters()).data).idx

    def reset_parameters(self, param_init):
        raise NotImplementedError

    def greedy(self):
        raise NotImplementedError

    def beam_search(self):
        raise NotImplementedError

    def _plot_attention(self):
        raise NotImplementedError
