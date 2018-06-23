#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm


def wrap_iterator(iterator, progressbar):
    if progressbar:
        iterator = tqdm(iterator)
    return iterator


def wrap_generator(generator, progressbar, total):
    if progressbar:
        generator = tqdm(generator, total=total)
    return generator
