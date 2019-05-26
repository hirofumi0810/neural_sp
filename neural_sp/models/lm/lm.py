#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Select a language model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from neural_sp.models.lm.gated_convlm import GatedConvLM
from neural_sp.models.lm.rnnlm import RNNLM
from neural_sp.models.lm.transformerlm import TransformerLM


def select_lm(args, save_path=None):
    if args.lm_type == 'gated_cnn':
        lm = GatedConvLM(args, save_path)
    elif args.lm_type == 'transformer':
        lm = TransformerLM(args, save_path)
    else:
        lm = RNNLM(args, save_path)
    return lm
