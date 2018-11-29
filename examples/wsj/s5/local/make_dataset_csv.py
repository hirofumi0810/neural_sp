#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make dataset CSV files (WSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile
import sys
import argparse
from tqdm import tqdm
import pandas as pd
import pickle
import codecs
import re
from distutils.util import strtobool

sys.path.append('../../../')
from src.utils.io.labels.phone import Phone2idx
from src.utils.io.labels.character import Char2idx
from src.utils.io.labels.word import Word2idx
from src.utils.directory import mkdir_join

parser = argparse.ArgumentParser()
parser.add_argument('--data_save_path', type=str,
                    help='path to save dataset')
parser.add_argument('--tool', type=str,
                    choices=['htk', 'python_speech_features', 'librosa'])
parser.add_argument('--has_extended_text', nargs='?', type=strtobool, default=False,
                    help='')
args = parser.parse_args()

DOUBLE_LETTERS = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj',
                  'kk', 'll', 'mm', 'nn', 'oo', 'pp', 'qq', 'rr', 'ss', 'tt',
                  'uu', 'vv', 'ww', 'xx', 'yy', 'zz']
SPACE = '_'
NOISE = '@'
OOV = 'OOV'
WORD_BOUNDARY = 'wb'


def read_text(text_path, vocab_save_path, data_type, lexicon_path):

    with open(text_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if data_type == 'train_lm':
                utt_idx = i
                trans = line.lower()

                trans = re.sub(r'[*\\$^]+', '', trans)

                # Replace digits
                trans = ' '.join([w for w in trans.split(
                    ' ') if re.search(r'[0-9]+st', w) is None])
                trans = ' '.join([w for w in trans.split(
                    ' ') if re.search(r'[0-9]+nd', w) is None])
                trans = ' '.join([w for w in trans.split(
                    ' ') if re.search(r'[0-9]+rd', w) is None])
                trans = ' '.join([w for w in trans.split(
                    ' ') if re.search(r'[0-9]+th', w) is None])
                trans = trans.replace('\'87', 'eighty seven')
                trans = trans.replace(' 1 ', ' one ')
                trans = trans.replace(' 20 ', ' twenty ')
                trans = trans.replace(' 5 ', ' five ')
                trans = trans.replace(' 6 ', ' six ')
                trans = trans.replace(' 2 ', ' two ')
                trans = trans.replace('6000', 'six thousand')
                trans = trans.replace('7000', 'seven thousand')
                trans = trans.replace('8000', 'eight thousand')
                trans = trans.replace('i2d2', 'i two d two')
                trans = trans.replace(' 8 ', ' eight ')
                trans = trans.replace(' 10 ', ' ten ')

                # Remove the first and last spaces
                # if trans[0] == ' ':
                #     trans = trans[1:]
                if trans[-1] == ' ':
                    trans = trans[:-1]

                trans_phone = ''
                # TODO:
            else:
                utt_idx = line.split(' ')[0]
                trans = ' '.join(line.split(' ')[1:]).lower()

                # Phoneme
                trans_phone = ''
                if 'eval' not in data_type:
                    words = trans.split(' ')
                    for i, w in enumerate(words):
                        if w in word2phone.keys():
                            trans_phone += word2phone[w].replace(' ', SPACE)
                        else:
                            trans_phone += word2phone['<unk>']
                        if i != len(words) - 1:
                            trans_phone += SPACE + WORD_BOUNDARY + SPACE

                trans = trans.replace('<noise>', NOISE)
                trans = trans.replace('.period', 'period')
                trans = trans.replace('\'single-quote', 'single-quote')
                trans = trans.replace('-hyphen', 'hyphen')
                trans = trans.replace('`', '\'')  # 47rc020w (typo)
                trans = re.sub(r'[(){}*,?!":;&/~]+', '', trans)
                trans = re.sub(r'<.*>', '', trans)

            trans = re.sub(r'[\s]+', ' ', trans)
            trans = trans.replace(' ', SPACE)

            if len(trans) == 0:
                continue
                # NOTE: utterances such as ~~
                # 46uc030b
                # 47hc0418

            trans_capital = ''
            for w in trans.split(SPACE):
                # Count word frequency
                if w not in word_dict.keys():
                    word_dict[w] = 1
                else:
                    word_dict[w] += 1

                word_set.add(w)
                char_set |= set(list(w))

                # Capital-divided
                if len(w) == 1:
                    char_capital_set.add(w.upper())
                    trans_capital += w.upper()
                else:
                    # Replace the first character with the capital letter
                    try:
                        w = w[0].upper() + w[1:]
                    except:
                        print(trans)
                        raise ValueError
                    char_capital_set.add(w[0])

                    # Check double-letters
                    for i in range(0, len(w) - 1, 1):
                        if w[i:i + 2] in DOUBLE_LETTERS:
                            char_capital_set.add(w[i:i + 2])
                        else:
                            char_capital_set.add(w[i])
                    trans_capital += w


if __name__ == '__main__':

    main()
