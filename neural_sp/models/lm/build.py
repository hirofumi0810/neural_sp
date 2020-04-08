#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Select a language model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from neural_sp.models.lm.gated_convlm import GatedConvLM
from neural_sp.models.lm.rnnlm import RNNLM
from neural_sp.models.lm.transformerlm import TransformerLM
# from neural_sp.models.lm.wordlm import LookAheadWordLM


def build_lm(args, save_path=None, wordlm=False, lm_dict_path=None, asr_dict_path=None):
    """Select LM class.

    Args:
        args ():
        save_path (str):
        wordlm (bool):
        lm_dict_path (dict):
        asr_dict_path (dict):
    Returns:
        lm ():

    """
    if 'gated_conv' in args.lm_type:
        lm = GatedConvLM(args, save_path)
    elif args.lm_type == 'transformer':
        lm = TransformerLM(args, save_path)
    else:
        lm = RNNLM(args, save_path)

        # Word-level RNNLM
        # if wordlm:
        #     lm = LookAheadWordLM(lm, lm_dict_path, asr_dict_path)

    return lm
