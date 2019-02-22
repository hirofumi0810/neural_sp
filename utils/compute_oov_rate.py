#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Compute OOV (out-of-vocabylary) rate."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import codecs

parser = argparse.ArgumentParser()
parser.add_argument('word_count', type=str,
                    help='word count file')
parser.add_argument('dict', type=str,
                    help='dictionary file')
parser.add_argument('set', type=str,
                    help='dataset')
args = parser.parse_args()


def main():

    token_set = set([])
    with codecs.open(args.dict, 'r', encoding="utf-8") as f:
        token_set = set([])
        for line in f:
            token, id = line.strip().split(' ')
            token_set.add(token)

    oov_count = 0
    num_words = 0
    with codecs.open(args.word_count, 'r', encoding="utf-8") as f:
        for line in f:
            count, w = line.strip().split(' ')

            # For swbd
            if w == '(%hesitation)':
                continue

            num_words += int(count)
            if w not in token_set:
                oov_count += int(count)

    oov_rate = float(oov_count * 100) / float(num_words)
    print("%s: %.3f%%" % (args.set, oov_rate))


if __name__ == '__main__':
    main()
