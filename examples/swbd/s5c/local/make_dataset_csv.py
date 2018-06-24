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
import re
from distutils.util import strtobool

sys.path.append('../../../')
# from utils.io.labels.phone import Phone2idx
from utils.io.labels.character import Char2idx
from utils.io.labels.word import Word2idx
from utils.directory import mkdir_join

parser = argparse.ArgumentParser()
parser.add_argument('--data_save_path', type=str,
                    help='path to save dataset')
parser.add_argument('--tool', type=str,
                    choices=['htk', 'python_speech_features', 'librosa'])
parser.add_argument('--has_fisher', nargs='?', type=strtobool, default=False,
                    help='')

DOUBLE_LETTERS = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj',
                  'kk', 'll', 'mm', 'nn', 'oo', 'pp', 'qq', 'rr', 'ss', 'tt',
                  'uu', 'vv', 'ww', 'xx', 'yy', 'zz']
SPACE = '_'
LAUGHTER = 'L'
NOISE = 'N'
VOCALIZED_NOISE = 'V'
OOV = 'OOV'

SOF = 'F'
EOF = 'G'
SOD = 'D'
EOD = 'E'
SOB = 'B'
EOB = 'C'

HESITATIONS = ['UH', 'UM', 'EH', 'MM',  'HM', 'AH',
               'HUH', 'HA', 'ER', 'OOF', 'HEE', 'ACH', 'EEE' 'EW']
BACKCHANNELS = ['UH-HUH', 'UM-HUM', 'MHM', 'MMHM', 'MM-HM', 'MM-HUH', 'HUH-UH']


def main():

    args = parser.parse_args()

    data_types = ['train', 'dev', 'eval2000_swbd', 'eval2000_ch']
    if args.has_fisher:
        data_types += ['train_fisher']

    # for data_type in data_types:
    for data_type in ['train_fisher']:
        print('=' * 50)
        print(' ' * 20 + data_type)
        print('=' * 50)

        if 'eval' in data_type:
            data_type_tmp = 'eval2000'
        else:
            data_type_tmp = data_type

        # Convert transcript to index
        print('=> Processing transcripts...')
        if data_type == 'train_fisher':
            trans_dict = read_text(
                text_path=join(args.data_save_path, 'train', 'text'),
                vocab_save_path=mkdir_join(args.data_save_path, 'vocab'),
                data_type=data_type,
                lexicon_path=None)
            trans_dict_fisher = read_text(
                text_path=join(args.data_save_path, 'train_fisher', 'text'),
                vocab_save_path=mkdir_join(args.data_save_path, 'vocab'),
                data_type=data_type,
                lexicon_path=None)
            trans_dict.update(trans_dict_fisher)
        else:
            trans_dict = read_text(
                text_path=join(args.data_save_path, data_type_tmp, 'text'),
                vocab_save_path=mkdir_join(args.data_save_path, 'vocab'),
                data_type=data_type,
                lexicon_path=None)

        # Make dataset file (.csv)
        print('=> Saving dataset files...')
        csv_save_path = mkdir_join(
            args.data_save_path, 'dataset', args.tool, data_type)

        df_columns = ['frame_num', 'input_path', 'transcript']
        df_word = pd.DataFrame([], columns=df_columns)
        df_char = pd.DataFrame([], columns=df_columns)
        df_char_capital = pd.DataFrame([], columns=df_columns)
        df_char_left = pd.DataFrame([], columns=df_columns)
        df_char_right = pd.DataFrame([], columns=df_columns)
        df_char_both = pd.DataFrame([], columns=df_columns)
        df_char_remove = pd.DataFrame([], columns=df_columns)

        if data_type != 'train_fisher':
            with open(join(args.data_save_path, 'feature', args.tool, data_type,
                           'frame_num.pickle'), 'rb') as f:
                frame_num_dict = pickle.load(f)

        utt_count = 0
        df_word_list = []
        df_char_list, df_char_capital_list = [], []
        df_char_left_list, df_char_right_list, df_char_both_list, df_char_remove_list = [], [], [], []
        for utt_idx, trans in tqdm(trans_dict.items()):
            if data_type == 'train_fisher':
                df_word = add_element(
                    df_word, [len(trans['word'].split('_')), '', trans['word']])
                df_char = add_element(
                    df_char, [len(trans['char'].split('_')), '', trans['char']])
                df_char_capital = add_element(
                    df_char_capital, [len(trans['char_capital'].split('_')), '', trans['char_capital']])
                df_char_left = add_element(
                    df_char_left, [len(trans['char_left'].split('_')), '', trans['char_left']])
                df_char_right = add_element(
                    df_char_right, [len(trans['char_right'].split('_')), '', trans['char_right']])
                df_char_both = add_element(
                    df_char_both, [len(trans['char_both'].split('_')), '', trans['char_both']])
                df_char_remove = add_element(
                    df_char_remove, [len(trans['char_remove'].split('_')), '', trans['char_remove']])
            else:
                if 'eval2000' in data_type:
                    speaker = '_'.join(utt_idx.split('_')[:2])
                else:
                    speaker = utt_idx.split('_')[0]
                feat_utt_save_path = join(
                    args.data_save_path, 'feature', args.tool, data_type,
                    speaker, utt_idx + '.npy')
                frame_num = frame_num_dict[utt_idx]

                if not isfile(feat_utt_save_path):
                    raise ValueError('There is no file: %s' %
                                     feat_utt_save_path)

                df_word = add_element(
                    df_word, [frame_num, feat_utt_save_path, trans['word']])
                df_char = add_element(
                    df_char, [frame_num, feat_utt_save_path, trans['char']])
                df_char_capital = add_element(
                    df_char_capital, [frame_num, feat_utt_save_path, trans['char_capital']])
                df_char_left = add_element(
                    df_char_left, [frame_num, feat_utt_save_path, trans['char_left']])
                df_char_right = add_element(
                    df_char_right, [frame_num, feat_utt_save_path, trans['char_right']])
                df_char_both = add_element(
                    df_char_both, [frame_num, feat_utt_save_path, trans['char_both']])
                df_char_remove = add_element(
                    df_char_remove, [frame_num, feat_utt_save_path, trans['char_remove']])
            utt_count += 1

            # Reset
            if utt_count == 10000:
                df_word_list.append(df_word)
                df_char_list.append(df_char)
                df_char_capital_list.append(df_char_capital)
                df_char_left_list.append(df_char_left)
                df_char_right_list.append(df_char_right)
                df_char_both_list.append(df_char_both)
                df_char_remove_list.append(df_char_remove)

                df_word = pd.DataFrame([], columns=df_columns)
                df_char = pd.DataFrame([], columns=df_columns)
                df_char_capital = pd.DataFrame([], columns=df_columns)
                df_char_left = pd.DataFrame([], columns=df_columns)
                df_char_right = pd.DataFrame([], columns=df_columns)
                df_char_both = pd.DataFrame([], columns=df_columns)
                df_char_remove = pd.DataFrame([], columns=df_columns)

                utt_count = 0

        # Last dataframe
        df_word_list.append(df_word)
        df_char_list.append(df_char)
        df_char_capital_list.append(df_char_capital)
        df_char_left_list.append(df_char_left)
        df_char_right_list.append(df_char_right)
        df_char_both_list.append(df_char_both)
        df_char_remove_list.append(df_char_remove)

        # Concatenate all dataframes
        df_word = df_word_list[0]
        df_char = df_char_list[0]
        df_char_capital = df_char_capital_list[0]
        df_char_left = df_char_left_list[0]
        df_char_right = df_char_right_list[0]
        df_char_both = df_char_both_list[0]
        df_char_remove = df_char_remove_list[0]

        for i in df_word_list[1:]:
            df_word = pd.concat([df_word, i], axis=0)
        for i in df_char_list[1:]:
            df_char = pd.concat([df_char, i], axis=0)
        for i in df_char_capital_list[1:]:
            df_char_capital = pd.concat([df_char_capital, i], axis=0)
        for i in df_char_left_list[1:]:
            df_char_left = pd.concat([df_char_left, i], axis=0)
        for i in df_char_right_list[1:]:
            df_char_right = pd.concat([df_char_right, i], axis=0)
        for i in df_char_both_list[1:]:
            df_char_both = pd.concat([df_char_both, i], axis=0)
        for i in df_char_remove_list[1:]:
            df_char_remove = pd.concat([df_char_remove, i], axis=0)

        df_word.to_csv(join(csv_save_path, 'word.csv'), encoding='utf-8')
        df_char.to_csv(join(csv_save_path, 'character.csv'), encoding='utf-8')
        df_char_capital.to_csv(
            join(csv_save_path, 'character_capital_divide.csv'), encoding='utf-8')
        df_char_left.to_csv(
            join(csv_save_path, 'char_left.csv'), encoding='utf-8')
        df_char_right.to_csv(
            join(csv_save_path, 'char_right.csv'), encoding='utf-8')
        df_char_both.to_csv(
            join(csv_save_path, 'char_both.csv'), encoding='utf-8')
        df_char_remove.to_csv(
            join(csv_save_path, 'char_remove.csv'), encoding='utf-8')


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
                value (dict) => the dictionary of utterance information of each speaker
                    key (string) => utterance index
                    value (dict)
                        key => label type
                        value => indices
    """
    # Make vocabulary files
    word_vocab_path = mkdir_join(vocab_save_path, 'word.txt')
    char_vocab_path = mkdir_join(vocab_save_path, 'character.txt')
    char_capital_vocab_path = mkdir_join(
        vocab_save_path, 'character_capital_divide.txt')
    char_left_vocab_path = mkdir_join(vocab_save_path, 'character_left.txt')
    char_right_vocab_path = mkdir_join(vocab_save_path, 'character_right.txt')
    char_both_vocab_path = mkdir_join(vocab_save_path, 'character_both.txt')
    char_remove_vocab_path = mkdir_join(
        vocab_save_path, 'character_remove.txt')

    trans_dict = {}
    char_set = set([])
    char_capital_set = set([])
    char_set_remove = set([])
    word_set = set([])
    word_dict = {}
    with open(text_path, 'r') as f:
        for line in f:
            line = line.strip()
            utt_idx = line.split(' ')[0]
            trans = ' '.join(line.split(' ')[1:]).lower()

            if data_type == 'eval2000_swbd' and utt_idx[:2] == 'en':
                continue
            if data_type == 'eval2000_ch' and utt_idx[:2] == 'sw':
                continue

            # text normalization
            trans = trans.replace('[laughter]', LAUGHTER)
            trans = trans.replace('[noise]', NOISE)
            trans = trans.replace('[vocalized-noise]', VOCALIZED_NOISE)

            if data_type == 'train_fisher':
                trans = trans.replace("[[skip]]", "")
                trans = trans.replace("[pause]", "")
                trans = trans.replace("[laugh]", LAUGHTER)
                trans = trans.replace("[sigh]", NOISE)
                trans = trans.replace("[cough]", NOISE)
                trans = trans.replace("[mn]", NOISE)
                trans = trans.replace("[breath]", NOISE)
                trans = trans.replace("[lipsmack]", NOISE)
                trans = trans.replace("[sneeze]", NOISE)
                trans = re.sub(r'[,?*~]', '', trans)

                # trans = trans.replace('_', '')
                # trans = trans.replace('-', '')
                # trans = trans.replace('%.', '')
                # trans = trans.replace('%s+', ' ')
                # trans = trans.replace('^%s+', '')
                # trans = trans.replace('%s+$', '')
                # trans = trans.replace('%s', '|').replace('(.)', '%1 ').replace(' $', '')

            if 'eval2000' in data_type:
                trans = trans.replace('<b_aside>', '')
                trans = trans.replace('<e_aside>', '')
                trans = re.sub(r'[()]+', '', trans)

            if len(trans) == 0:
                continue

            # Remove consecutive spaces
            trans = re.sub(r'[\s]+', ' ', trans)

            # Remove the first and last spaces
            if trans[0] == ' ':
                trans = trans[1:]
            if trans[-1] == ' ':
                trans = trans[:-1]

            ###################################
            # with filler and disfluency
            ###################################
            trans_left_list, trans_right_list, trans_both_list, trans_remove_list = [], [], [], []
            for w in trans.split(' '):
                if w[-1] == '-':
                    w_left = SOD + w[:-1]
                    w_right = w[:-1] + SOD
                    w_both = SOD + w[:-1] + EOD
                elif w[0] == '-':
                    w_left = SOD + w[1:]
                    w_right = w[1:] + SOD
                    w_both = SOD + w[1:] + EOD
                elif w.upper() in HESITATIONS:
                    w_left = SOF + w
                    w_right = w + SOF
                    w_both = SOF + w + EOF
                elif w.upper() in BACKCHANNELS:
                    w_left = SOB + w
                    w_right = w + SOB
                    w_both = SOB + w + EOB
                else:
                    w_left = w
                    w_right = w
                    w_both = w
                    trans_remove_list.append(w)
                trans_left_list.append(w_left)
                trans_right_list.append(w_right)
                trans_both_list.append(w_both)
            trans_left = SPACE.join(trans_left_list)
            trans_right = SPACE.join(trans_right_list)
            trans_both = SPACE.join(trans_both_list)
            trans_remove = SPACE.join(trans_remove_list)
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

                for w in trans_remove.split(SPACE):
                    char_set_remove |= set(list(w))

            trans_dict[utt_idx] = [trans, trans_capital,
                                   trans_left, trans_right, trans_both, trans_remove]

    # Reserve some indices
    char_set.discard('L')
    char_set.discard('N')
    char_set.discard('V')
    char_set.discard('B')
    char_set.discard('C')
    char_set.discard('D')
    char_set.discard('E')
    char_set.discard('F')
    char_set.discard('G')

    # Save vocabulary files
    if data_type == 'train':
        # word-level (threshold == 5)
        with open(word_vocab_path, 'w') as f:
            word_list = sorted([w for w, freq in list(word_dict.items())
                                if freq >= 5]) + [OOV]
            for w in word_list:
                f.write('%s\n' % w)

        # character-level
        with open(char_vocab_path, 'w') as f:
            char_list = sorted(list(char_set)) + \
                [SPACE, LAUGHTER, NOISE, VOCALIZED_NOISE]
            for c in char_list:
                f.write('%s\n' % c)

        # character-level (capital-divided)
        with open(char_capital_vocab_path, 'w') as f:
            char_capital_list = sorted(list(char_capital_set)) + \
                [LAUGHTER, NOISE, VOCALIZED_NOISE]
            for c in char_capital_list:
                f.write('%s\n' % c)

        # character-level (char_wb + left, right, both, remove)
        with open(char_left_vocab_path, 'w') as f:
            for c in char_list + [SPACE, LAUGHTER, NOISE, VOCALIZED_NOISE, SOF, SOD, SOB]:
                f.write('%s\n' % c)
        with open(char_right_vocab_path, 'w') as f:
            for c in char_list + [SPACE, LAUGHTER, NOISE, VOCALIZED_NOISE, SOF, SOD, SOB]:
                f.write('%s\n' % c)
        with open(char_both_vocab_path, 'w') as f:
            for c in char_list + [SPACE, LAUGHTER, NOISE, VOCALIZED_NOISE, SOF, SOD, SOB, EOF, EOD, EOB]:
                f.write('%s\n' % c)
        with open(char_remove_vocab_path, 'w') as f:
            char_list_remove = sorted(list(char_set_remove))
            for c in char_list_remove + [SPACE, LAUGHTER, NOISE, VOCALIZED_NOISE]:
                f.write('%s\n' % c)

    # Compute OOV rate
    if data_type != 'train':
        with open(mkdir_join(vocab_save_path, 'oov', data_type + '.txt'), 'w') as f:
            # word-level (threshold == 5)
            oov_rate = compute_oov_rate(word_dict, word_vocab_path)
            f.write('Word (freq5):\n')
            f.write('  OOV rate: %f %%\n' % oov_rate)

    # Convert to index
    print('=====> Convert to index...')
    word2idx = Word2idx(word_vocab_path)
    char2idx = Char2idx(char_vocab_path)
    char2idx_capital = Char2idx(char_capital_vocab_path, capital_divide=True)
    char2idx_left = Char2idx(char_left_vocab_path)
    char2idx_right = Char2idx(char_right_vocab_path)
    char2idx_both = Char2idx(char_both_vocab_path)
    char2idx_remove = Char2idx(char_remove_vocab_path)

    for utt_idx, [trans, trans_capital, trans_left, trans_right, trans_both, trans_remove] in tqdm(trans_dict.items()):
        if 'eval' in data_type:
            trans_dict[utt_idx] = {
                "word": trans,
                "char": trans,
                "char_capital": trans,
                "char_left": trans,
                "char_right": trans,
                "char_both": trans,
                "char_remove": trans_remove,
            }
            # NOTE: save as it is
        else:
            word_indices = word2idx(trans)
            char_indices = char2idx(trans)
            char_capital_indices = char2idx_capital(trans_capital)
            char_left_indices = char2idx_left(trans_left)
            char_right_indices = char2idx_right(trans_right)
            char_both_indices = char2idx_both(trans_both)
            char_remove_indices = char2idx_remove(trans_remove)

            word_indices = ' '.join(list(map(str, word_indices)))
            char_indices = ' '.join(list(map(str, char_indices)))
            char_capital_indices = ' '.join(
                list(map(str, char_capital_indices)))
            char_left_indices = ' '.join(list(map(str, char_left_indices)))
            char_right_indices = ' '.join(list(map(str, char_right_indices)))
            char_both_indices = ' '.join(list(map(str, char_both_indices)))
            char_remove_indices = ' '.join(list(map(str, char_remove_indices)))

            trans_dict[utt_idx] = {
                "word": word_indices,
                "char": char_indices,
                "char_capital": char_capital_indices,
                "char_left": char_left_indices,
                "char_right": char_right_indices,
                "char_both": char_both_indices,
                "char_remove": char_remove_indices,
            }

    return trans_dict


def compute_oov_rate(word_dict, vocab_path):

    with open(vocab_path, 'r') as f:
        vocab_set = set([])
        for line in f:
            word = line.strip()

            # Convert acronyms to character
            if word[-1] == '.':
                word = word.replace('.', '')

            vocab_set.add(word)

    oov_count = 0
    word_num = 0
    for word, freq in word_dict.items():

        if word == '%hesitation':
            continue

        word_num += freq
        # if word not in vocab_set and word.replace('-', '') not in vocab_set:
        if word not in vocab_set:
            oov_count += freq

    oov_rate = oov_count * 100 / word_num
    return oov_rate


if __name__ == '__main__':

    main()
