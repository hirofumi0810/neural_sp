#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make dataset CSV files (LibriSpeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile
import sys
import argparse
from tqdm import tqdm
import pandas as pd
import pickle

sys.path.append('../../../')
from utils.io.labels.phone import Phone2idx
from utils.io.labels.character import Char2idx
from utils.io.labels.word import Word2idx
from utils.directory import mkdir_join

parser = argparse.ArgumentParser()
parser.add_argument('--data_save_path', type=str,
                    help='path to save dataset')
parser.add_argument('--data_size', type=str,
                    choices=['100', '460', '960'])
parser.add_argument('--tool', type=str,
                    choices=['htk', 'python_speech_features', 'librosa'])

args = parser.parse_args()

DOUBLE_LETTERS = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj',
                  'kk', 'll', 'mm', 'nn', 'oo', 'pp', 'qq', 'rr', 'ss', 'tt',
                  'uu', 'vv', 'ww', 'xx', 'yy', 'zz']
SPACE = '_'
HYPHEN = '-'
APOSTROPHE = '\''
OOV = 'OOV'


def main():

    for data_type in ['train_' + args.data_size, 'dev_clean', 'dev_other', 'test_clean', 'test_other']:
        print('=' * 50)
        print(' ' * 20 + data_type)
        print('=' * 50)

        # Convert transcript to index
        print('=> Processing transcripts...')
        trans_dict = read_text(
            text_path=join(args.data_save_path, data_type, 'text'),
            vocab_save_path=mkdir_join(
                args.data_save_path, 'vocab', args.data_size),
            data_type=data_type)

        # Make dataset file (.csv)
        print('=> Saving dataset files...')
        csv_save_path = mkdir_join(
            args.data_save_path, 'dataset', args.tool, args.data_size)

        df_columns = ['frame_num', 'input_path', 'transcript']
        df_word = pd.DataFrame([], columns=df_columns)
        df_char = pd.DataFrame([], columns=df_columns)
        df_char_capital = pd.DataFrame([], columns=df_columns)

        with open(join(args.data_save_path, 'feature', args.tool, args.data_size, data_type, 'frame_num.pickle'), 'rb') as f:
            frame_num_dict = pickle.load(f)

        utt_count = 0
        df_word_list = []
        df_char_list, df_char_capital_list = [], []
        for utt_name, trans in tqdm(trans_dict.items()):
            speaker, chapter, utt_idx = utt_name.split('-')
            feat_utt_save_path = mkdir_join(
                args.data_save_path, 'feature', args.tool, args.data_size, data_type, speaker, chapter, utt_name + '.npy')
            frame_num = frame_num_dict[utt_name]

            if not isfile(feat_utt_save_path):
                raise ValueError('There is no file: %s' % feat_utt_save_path)

            df_word = add_element(
                df_word, [frame_num, feat_utt_save_path, trans['word']])
            df_char = add_element(
                df_char, [frame_num, feat_utt_save_path, trans['char']])
            df_char_capital = add_element(
                df_char_capital, [frame_num, feat_utt_save_path, trans['char_capital']])
            utt_count += 1

            # Reset
            if utt_count == 10000:
                df_word_list.append(df_word)
                df_char_list.append(df_char)
                df_char_capital_list.append(df_char_capital)

                df_word = pd.DataFrame([], columns=df_columns)
                df_char = pd.DataFrame([], columns=df_columns)
                df_char_capital = pd.DataFrame([], columns=df_columns)
                utt_count = 0

        # Last dataframe
        df_word_list.append(df_word)
        df_char_list.append(df_char)
        df_char_capital_list.append(df_char_capital)

        # Concatenate all dataframes
        df_word = df_word_list[0]
        df_char = df_char_list[0]
        df_char_capital = df_char_capital_list[0]

        for i in df_word_list[1:]:
            df_word = pd.concat([df_word, i], axis=0)
        for i in df_char_list[1:]:
            df_char = pd.concat([df_char, i], axis=0)
        for i in df_char_capital_list[1:]:
            df_char_capital = pd.concat([df_char_capital, i], axis=0)

        df_word.to_csv(join(csv_save_path, 'word.csv'))
        df_char.to_csv(join(csv_save_path, 'character.csv'))
        df_char_capital.to_csv(
            join(csv_save_path, 'character_capital_divide.csv'))


def add_element(df, elem_list):
    series = pd.Series(elem_list, index=df.columns)
    df = df.append(series, ignore_index=True)
    return df


def read_text(text_path, vocab_save_path, data_type):
    """Read transcripts (.sdb) & save files (.npy).
    Args:
        text_path (string): path to a text file of kaldi
        vocab_save_path (string): path to save vocabulary files
        data_type (string): train or dev-clean or dev-other or test-clean or test-other
    Returns:
        speaker_dict (dict): the dictionary of utterances of each speaker
            key (string) => speaker
            value (dict) => the dictionary of utterance information of each speaker
                key (string) => utterance index
                value (dict) =>
                    key => label type
                    value => indices
    """
    # Make vocabulary files
    word_vocab_path = mkdir_join(vocab_save_path, 'word.txt')
    char_vocab_path = mkdir_join(vocab_save_path, 'character.txt')
    char_capital_vocab_path = mkdir_join(
        vocab_save_path, 'character_capital_divide.txt')

    trans_dict = {}
    char_set = set([])
    char_capital_set = set([])
    word_set = set([])
    word_dict = {}
    with open(text_path, 'r') as f:
        for line in f:
            line = line.strip()
            utt_idx = line.split(' ')[0]
            trans = ' '.join(line.split(' ')[1:]).lower()

            # text normalization
            trans = trans.replace(' ', SPACE)

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
                    # Replace the first character with the capital
                    # letter
                    w = w[0].upper() + w[1:]
                    char_capital_set.add(w[0])

                    # Check double-letters
                    for i in range(0, len(w) - 1, 1):
                        if w[i:i + 2] in DOUBLE_LETTERS:
                            char_capital_set.add(w[i:i + 2])
                        else:
                            char_capital_set.add(w[i])
                    trans_capital += w

            trans_dict[utt_idx] = [trans, trans_capital]

    # Save vocabulary files
    if 'train' in data_type:
        # word-level (threshold == 5)
        with open(word_vocab_path, 'w') as f:
            word_list = sorted([word for word, freq in list(word_dict.items())
                                if freq >= 5]) + [OOV]
            for w in word_list:
                f.write('%s\n' % w)

        # character-level
        with open(char_vocab_path, 'w') as f:
            char_list = sorted(list(char_set)) + [SPACE]
            for char in char_list:
                f.write('%s\n' % char)

        # character-level (capital-divided)
        with open(char_capital_vocab_path, 'w') as f:
            char_capital_list = sorted(list(char_capital_set))
            for char in char_capital_list:
                f.write('%s\n' % char)

    # Compute OOV rate
    with open(mkdir_join(vocab_save_path, 'oov', data_type + '.txt'), 'w') as f:
        # word-level (threshold == 5)
        oov_rate = compute_oov_rate(word_dict, word_vocab_path)
        f.write('Word (freq5):\n')
        f.write('  OOV rate (test): %f %%\n' % oov_rate)

    # Convert to index
    print('=====> Convert to index...')
    word2idx = Word2idx(word_vocab_path)
    char2idx = Char2idx(char_vocab_path)
    char2idx_capital = Char2idx(char_capital_vocab_path, capital_divide=True)

    for utt_idx, [trans, trans_pos] in tqdm(trans_dict.items()):
        if 'test' in data_type:
            trans_dict[utt_idx] = {"word": trans,
                                   "char": trans,
                                   "char_capital": trans}
            # NOTE: save as it is
        else:
            word_indices = word2idx(trans)
            char_indices = char2idx(trans)
            char_capital_indices = char2idx_capital(trans)

            word_indices = ' '.join(list(map(str, word_indices)))
            char_indices = ' '.join(list(map(str, char_indices)))
            char_capital_indices = ' '.join(
                list(map(str, char_capital_indices)))

            trans_dict[utt_idx] = {"word": word_indices,
                                   "char": char_indices,
                                   "char_capital": char_capital_indices}

    return trans_dict


def compute_oov_rate(word_dict, vocab_path):

    with open(vocab_path, 'r') as f:
        vocab_set = set([])
        for line in f:
            word = line.strip()
            vocab_set.add(word)

    oov_count = 0
    word_num = 0
    for word, freq in word_dict.items():
        word_num += freq
        if word not in vocab_set:
            oov_count += freq

    oov_rate = oov_count * 100 / word_num
    return oov_rate


if __name__ == '__main__':

    main()
