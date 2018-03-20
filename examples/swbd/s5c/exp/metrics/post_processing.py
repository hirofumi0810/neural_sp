#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Fix transcription for evaluation (Switchboard corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

LAUGHTER = 'LA'
NOISE = 'NZ'
VOCALIZED_NOISE = 'VN'
HESITATION = '%hesitation'


def fix_trans(transcript, glm):
    """Fix transcription.
    Args:
        transcript (string):
    Returns:
        transcript (string):
    """
    # Remove consecutive spaces
    transcript = re.sub(r'[_]+', '_', transcript)

    # Fix abbreviation, hesitation
    transcript = glm.fix_trans(transcript)
    # TODO: 省略は元に戻すのではなく，逆に全てを省略形にする方が良い？？

    # Remove NOISE, LAUGHTER, VOCALIZED-NOISE, HESITATION
    transcript = transcript.replace(NOISE, '')
    transcript = transcript.replace(LAUGHTER, '')
    transcript = transcript.replace(VOCALIZED_NOISE, '')
    # transcript = transcript.replace(HESITATION, '')

    # Remove garbage labels
    transcript = re.sub(r'[<>]+', '', transcript)
    transcript = transcript.replace('-', '')
    # transcript = transcript.replace('\'', '')

    # Convert acronyms to character
    transcript = transcript.replace('.', '')

    # Remove consecutive spaces again
    transcript = re.sub(r'[_]+', '_', transcript)

    # Remove the first and last space
    if len(transcript) > 0 and transcript[0] == '_':
        transcript = transcript[1:]
    if len(transcript) > 0 and transcript[-1] == '_':
        transcript = transcript[: -1]

    return transcript
