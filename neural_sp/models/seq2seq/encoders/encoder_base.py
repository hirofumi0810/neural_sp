#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for encoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch

from neural_sp.models.base import ModelBase

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

    def reset_parameters(self, param_init):
        raise NotImplementedError

    def forward(self, xs, xlens, task):
        raise NotImplementedError
