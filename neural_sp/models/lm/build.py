#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Select a language model"""


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
        from neural_sp.models.lm.gated_convlm import GatedConvLM
        lm = GatedConvLM(args, save_path)
    elif args.lm_type == 'transformer':
        from neural_sp.models.lm.transformerlm import TransformerLM
        lm = TransformerLM(args, save_path)
    elif args.lm_type == 'transformer_xl':
        from neural_sp.models.lm.transformer_xl import TransformerXL
        lm = TransformerXL(args, save_path)
    else:
        from neural_sp.models.lm.rnnlm import RNNLM
        lm = RNNLM(args, save_path)

    return lm
