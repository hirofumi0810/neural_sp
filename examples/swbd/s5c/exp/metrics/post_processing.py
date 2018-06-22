#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Fix transcription for evaluation (Switchboard corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from utils.evaluation.normalization import normalize

LAUGHTER = 'L'
NOISE = 'N'
VOCALIZED_NOISE = 'V'
HESITATION = '%hesitation'


def fix_trans(trans, glm):
    """Fix transcription.
    Args:
        trans (string):
    Returns:
        trans (string):
    """
    # Remove consecutive spaces
    trans = re.sub(r'[_]+', '_', trans)

    # Fix abbreviation, hesitation
    trans = glm(trans)
    # TODO: 省略は元に戻すのではなく，逆に全てを省略形にする方が良い？？

    # Replace OOV temporaly
    trans = trans.replace('OOV', '@')

    # Remove noisy labels
    trans = normalize(trans, remove_tokens=[
                      NOISE, LAUGHTER, VOCALIZED_NOISE, '>'])
    # trans = trans.replace(HESITATION, '')
    trans = trans.replace('-', '')
    # trans = trans.replace('\'', '')
    # trans = trans.replace('.', '')

    # Replace back OOV
    trans = trans.replace('@', 'OOV')

    return trans
