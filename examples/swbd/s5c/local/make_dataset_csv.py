#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make dataset CSV files (Switchboard corpus)."""

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
# from utils.io.labels.phone import Phone2idx
from utils.io.labels.character import Char2idx
from utils.io.labels.word import Word2idx
from utils.directory import mkdir_join
# from utils.feature_extraction.wav_split import split_wav

parser = argparse.ArgumentParser()
parser.add_argument('--data_save_path', type=str,
                    help='path to save dataset')
parser.add_argument('--tool', type=str,
                    choices=['wav', 'htk', 'python_speech_features', 'librosa'])

args = parser.parse_args()

DOUBLE_LETTERS = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj',
                  'kk', 'll', 'mm', 'nn', 'oo', 'pp', 'qq', 'rr', 'ss', 'tt',
                  'uu', 'vv', 'ww', 'xx', 'yy', 'zz']
SPACE = '_'
LAUGHTER = 'LA'
NOISE = 'NZ'
VOCALIZED_NOISE = 'VN'
OOV = 'OOV'


def main():

    for data_type in ['train', 'dev', 'eval2000_swbd', 'eval2000_ch']:
        print('=' * 50)
        print(' ' * 20 + data_type)
        print('=' * 50)

        # Convert transcript to index
        print('=> Processing transcripts...')
        trans_dict = read_text(
            text_path=join(args.data_save_path, data_type, 'text'),
            vocab_save_path=mkdir_join(args.data_save_path, 'vocab'),
            data_type=data_type,
            lexicon_path=None)

        # Make dataset file (.csv)
        print('=> Saving dataset files...')
        csv_save_path = mkdir_join(
            args.data_save_path, 'dataset', args.tool, data_type)

        df_columns = ['frame_num', 'input_path', 'transcript']
        df_word1 = pd.DataFrame([], columns=df_columns)
        df_word5 = pd.DataFrame([], columns=df_columns)
        df_word10 = pd.DataFrame([], columns=df_columns)
        df_word15 = pd.DataFrame([], columns=df_columns)
        df_char = pd.DataFrame([], columns=df_columns)
        df_char_capital = pd.DataFrame([], columns=df_columns)

        with open(join(args.data_save_path, 'feature', args.tool, data_type,
                       'frame_num.pickle'), 'rb') as f:
            frame_num_dict = pickle.load(f)

        utt_count = 0
        df_word1_list, df_word5_list = [], []
        df_word10_list, df_word15_list = [], []
        df_char_list, df_char_capital_list = [], []
        for utt_idx, trans_list in tqdm(trans_dict.items()):
            word1_indices, word5_indices = trans_list[:2]
            word10_indices, word15_indices = trans_list[2:4]
            char_indices, char_capital_indices = trans_list[4:6]

            speaker = utt_idx.split('_')[0]

            if args.tool == 'wav':
                raise NotImplementedError
                feat_utt_save_path = join(
                    args.data_save_path, 'feature', args.tool, data_type,
                    speaker, utt_idx + '.wav')
                # frame_num =
            else:
                feat_utt_save_path = join(
                    args.data_save_path, 'feature', args.tool, data_type,
                    speaker, utt_idx + '.npy')
                frame_num = frame_num_dict[utt_idx]

            if not isfile(feat_utt_save_path):
                raise ValueError('There is no file: %s' % feat_utt_save_path)

            df_word1 = add_element(
                df_word1, [frame_num, feat_utt_save_path, word1_indices])
            df_word5 = add_element(
                df_word5, [frame_num, feat_utt_save_path, word5_indices])
            df_word10 = add_element(
                df_word10, [frame_num, feat_utt_save_path, word10_indices])
            df_word15 = add_element(
                df_word15, [frame_num, feat_utt_save_path, word15_indices])
            df_char = add_element(
                df_char, [frame_num, feat_utt_save_path, char_indices])
            df_char_capital = add_element(
                df_char_capital, [frame_num, feat_utt_save_path, char_capital_indices])
            utt_count += 1

            # Reset
            if utt_count == 10000:
                df_word1_list.append(df_word1)
                df_word5_list.append(df_word5)
                df_word10_list.append(df_word10)
                df_word15_list.append(df_word15)
                df_char_list.append(df_char)
                df_char_capital_list.append(df_char_capital)

                df_word1 = pd.DataFrame([], columns=df_columns)
                df_word5 = pd.DataFrame([], columns=df_columns)
                df_word10 = pd.DataFrame([], columns=df_columns)
                df_word15 = pd.DataFrame([], columns=df_columns)
                df_char = pd.DataFrame([], columns=df_columns)
                df_char_capital = pd.DataFrame([], columns=df_columns)
                utt_count = 0

        # Last dataframe
        df_word1_list.append(df_word1)
        df_word5_list.append(df_word5)
        df_word10_list.append(df_word10)
        df_word15_list.append(df_word15)
        df_char_list.append(df_char)
        df_char_capital_list.append(df_char_capital)

        # Concatenate all dataframes
        df_word1 = df_word1_list[0]
        df_word5 = df_word5_list[0]
        df_word10 = df_word10_list[0]
        df_word15 = df_word15_list[0]
        df_char = df_char_list[0]
        df_char_capital = df_char_capital_list[0]

        for i in df_word1_list[1:]:
            df_word1 = pd.concat([df_word1, i], axis=0)
        for i in df_word5_list[1:]:
            df_word5 = pd.concat([df_word5, i], axis=0)
        for i in df_word10_list[1:]:
            df_word10 = pd.concat([df_word10, i], axis=0)
        for i in df_word15_list[1:]:
            df_word15 = pd.concat([df_word15, i], axis=0)
        for i in df_char_list[1:]:
            df_char = pd.concat([df_char, i], axis=0)
        for i in df_char_capital_list[1:]:
            df_char_capital = pd.concat([df_char_capital, i], axis=0)

        df_word1.to_csv(join(csv_save_path, 'word1.csv'))
        df_word5.to_csv(join(csv_save_path, 'word5.csv'))
        df_word10.to_csv(join(csv_save_path, 'word10.csv'))
        df_word15.to_csv(join(csv_save_path, 'word15.csv'))
        df_char.to_csv(join(csv_save_path, 'character.csv'))
        df_char_capital.to_csv(
            join(csv_save_path, 'character_capital_divide.csv'))


def add_element(df, elem_list):
    series = pd.Series(elem_list, index=df.columns)
    df = df.append(series, ignore_index=True)
    return df


def read_text(text_path, vocab_save_path, data_type, lexicon_path=None):
    """Read transcripts (.sdb) & save files (.npy).
    Args:
        text_path (string): path to a text file of kaldi
        vocab_save_path (string): path to save vocabulary files
        data_type (string): train or dev or eval2000_swbd or eval2000_ch
        lexicon_path (string, optional):
    Returns:
        speaker_dict (dict): the dictionary of utterances of each speaker
            key (string) => speaker
            value (dict) => the dictionary of utterance information of each speaker
                key (string) => utterance index
                value (list) => list of
                    [word1_indices, word5_indices,
                     word10_indices, word15_indices
                     char_indices, char_capital_indices]
    """
    # Make vocabulary files
    word1_vocab_path = mkdir_join(vocab_save_path, 'word1.txt')
    word5_vocab_path = mkdir_join(vocab_save_path, 'word5.txt')
    word10_vocab_path = mkdir_join(vocab_save_path, 'word10.txt')
    word15_vocab_path = mkdir_join(vocab_save_path, 'word15.txt')
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
            trans = ' '.join(line.split(' ')[1:])

            if data_type == 'eval2000_swbd' and utt_idx[:2] == 'en':
                continue
            if data_type == 'eval2000_ch' and utt_idx[:2] == 'sw':
                continue

            # text normalization
            trans = trans.replace('[laughter]', LAUGHTER)
            trans = trans.replace('[noise]', NOISE)
            trans = trans.replace('[vocalized-noise]', VOCALIZED_NOISE)
            trans = trans.replace(' ', SPACE)

            trans_dict[utt_idx] = trans

            trans_capital = ''
            for word in trans.split(SPACE):
                # Count word frequency
                if word not in word_dict.keys():
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1

                word_set.add(word)
                char_set |= set(list(word))

                # Capital-divided
                if len(word) == 1:
                    char_capital_set.add(word)
                    trans_capital += word
                else:
                    # Replace the first character with the capital
                    # letter
                    word = word[0].upper() + word[1:]

                    # Check double-letters
                    for i in range(0, len(word) - 1, 1):
                        if word[i:i + 2] in DOUBLE_LETTERS:
                            char_capital_set.add(word[i:i + 2])
                        else:
                            char_capital_set.add(word[i])
                    trans_capital += word

    # Reserve some indices
    char_set.discard('L')
    char_set.discard('A')
    char_set.discard('N')
    char_set.discard('Z')
    char_set.discard('V')

    # Save vocabulary files
    if data_type == 'train':
        # word-level (threshold == 1)
        with open(word1_vocab_path, 'w') as f:
            word_list = sorted(list(word_set)) + [OOV]
            for word in word_list:
                f.write('%s\n' % word)
            # NOTE: OOV index is reserved for the dev set

        # word-level (threshold == 5)
        with open(word5_vocab_path, 'w') as f:
            word_list = sorted([word for word, freq in list(word_dict.items())
                                if freq >= 5]) + [OOV]
            for word in word_list:
                f.write('%s\n' % word)

        # word-level (threshold == 10)
        with open(word10_vocab_path, 'w') as f:
            word_list = sorted([word for word, freq in list(word_dict.items())
                                if freq >= 10]) + [OOV]
            for word in word_list:
                f.write('%s\n' % word)

        # word-level (threshold == 15)
        with open(word15_vocab_path, 'w') as f:
            word_list = sorted([word for word, freq in list(word_dict.items())
                                if freq >= 15]) + [OOV]
            for word in word_list:
                f.write('%s\n' % word)

        # character-level
        with open(char_vocab_path, 'w') as f:
            char_list = sorted(list(char_set)) + \
                [SPACE, LAUGHTER, NOISE, VOCALIZED_NOISE]
            for char in char_list:
                f.write('%s\n' % char)

        # character-level (capital-divided)
        with open(char_capital_vocab_path, 'w') as f:
            char_capital_list = sorted(list(char_capital_set)) + \
                [LAUGHTER, NOISE, VOCALIZED_NOISE]
            for char in char_capital_list:
                f.write('%s\n' % char)

    # Compute OOV rate
    if data_type != 'train':
        with open(mkdir_join(vocab_save_path, 'oov', data_type + '.txt'), 'w') as f:

            # word-level (threshold == 1)
            oov_rate = compute_oov_rate(word_dict, word1_vocab_path)
            f.write('Word (freq1):\n')
            f.write('  OOV rate (test): %f %%\n' % oov_rate)

            # word-level (threshold == 5)
            oov_rate = compute_oov_rate(word_dict, word5_vocab_path)
            f.write('Word (freq5):\n')
            f.write('  OOV rate (test): %f %%\n' % oov_rate)

            # word-level (threshold == 10)
            oov_rate = compute_oov_rate(word_dict, word10_vocab_path)
            f.write('Word (freq10):\n')
            f.write('  OOV rate (test): %f %%\n' % oov_rate)

            # word-level (threshold == 15)
            oov_rate = compute_oov_rate(word_dict, word15_vocab_path)
            f.write('Word (freq15):\n')
            f.write('  OOV rate (test): %f %%\n' % oov_rate)

    # Convert to index
    print('=====> Convert to index...')
    word2idx_freq1 = Word2idx(word1_vocab_path)
    word2idx_freq5 = Word2idx(word5_vocab_path)
    word2idx_freq10 = Word2idx(word10_vocab_path)
    word2idx_freq15 = Word2idx(word15_vocab_path)
    char2idx = Char2idx(char_vocab_path, double_letter=True)
    char2idx_capital = Char2idx(char_capital_vocab_path, capital_divide=True)

    for utt_idx, trans in tqdm(trans_dict.items()):
        if 'eval' in data_type:
            pass
            # trans_dict[utt_idx] = trans
            # NOTE: save as it is
        else:
            word1_indices = word2idx_freq1(trans)
            word5_indices = word2idx_freq5(trans)
            word10_indices = word2idx_freq10(trans)
            word15_indices = word2idx_freq15(trans)
            char_indices = char2idx(trans)
            char_capital_indices = char2idx_capital(trans)

            word1_indices = ' '.join(
                list(map(str, word1_indices.tolist())))
            word5_indices = ' '.join(
                list(map(str, word5_indices.tolist())))
            word10_indices = ' '.join(
                list(map(str, word10_indices.tolist())))
            word15_indices = ' '.join(
                list(map(str, word15_indices.tolist())))
            char_indices = ' '.join(
                list(map(str, char_indices.tolist())))
            char_capital_indices = ' '.join(
                list(map(str, char_capital_indices.tolist())))

            trans_dict[utt_idx] = [
                word1_indices, word5_indices, word10_indices, word15_indices,
                char_indices, char_capital_indices]

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
