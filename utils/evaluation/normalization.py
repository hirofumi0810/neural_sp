#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re


def normalize(text, remove_tokens=[]):

    if len(remove_tokens) > 0:
        for token in remove_tokens:
            text = text.replace(token, '')

    # Remove consecutive spaces
    text = re.sub(r'[_]+', '_', text)

    text = text.replace('_>', '').replace('>', '')

    # Remove first and last space
    if len(text) > 0:
        if text[0] == '_':
            text = text[1:]

        if text[-1] == '_':
            text = text[: -1]

    return text
