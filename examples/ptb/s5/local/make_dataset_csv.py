#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make dataset CSV files (PTB corpus)."""

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

sys.path.append('../../../')
from utils.io.labels.character import Char2idx
from utils.io.labels.word import Word2idx
from utils.directory import mkdir_join

parser = argparse.ArgumentParser()
parser.add_argument('--data_save_path', type=str,
                    help='path to save dataset')

args = parser.parse_args()

SPACE = '_'
SIL = 'sil'
OOV = 'OOV'
SHORT_PAUSE = '@'


def main():

    for data_type in ['train' 'dev', 'test']:
        print('=' * 50)
        print(' ' * 20 + data_type)
        print('=' * 50)

        # Convert transcript to index
        print('=> Processing transcripts...')
        trans_dict = read_text(
            text_path=join(args.data_save_path, data_type, 'text'),
            vocab_save_path=mkdir_join(args.data_save_path, 'vocab'),
            data_type=data_type)

        # Make dataset file (.csv)
        print('=> Saving dataset files...')
        csv_save_path = mkdir_join(
            args.data_save_path, 'dataset', args.data_size, data_type)

        df_columns = ['frame_num', 'input_path', 'transcript']
        df_word = pd.DataFrame([], columns=df_columns)
        df_char = pd.DataFrame([], columns=df_columns)

        utt_count = 0
        df_word_list = []
        df_char_list, df_char_wb_list = [], []
        for utt_idx, trans in tqdm(trans_dict.items()):
            speaker = utt_idx.split('_')[0]
            feat_utt_save_path = join(
                args.data_save_path, 'feature', args.data_size, data_type, speaker, utt_idx + '.npy')

            if not isfile(feat_utt_save_path):
                raise ValueError('There is no file: %s' % feat_utt_save_path)

            df_word = add_element(
                df_word, [frame_num, feat_utt_save_path, trans['word']])
            df_char = add_element(
                df_char, [frame_num, feat_utt_save_path, trans['char']])
            utt_count += 1

            # Reset
            if utt_count == 10000:
                df_word_list.append(df_word)
                df_char_list.append(df_char)
                df_char_wb_list.append(df_char_wb)
                df_char_wb_left_list.append(df_char_wb_left)
                df_char_wb_right_list.append(df_char_wb_right)
                df_char_wb_both_list.append(df_char_wb_both)
                df_char_wb_remove_list.append(df_char_wb_remove)
                # df_phone_list.append(df_phone)
                # df_phone_wb_list.append(df_phone_wb)
                df_pos_list.append(df_pos)

                df_word = pd.DataFrame([], columns=df_columns)
                df_char = pd.DataFrame([], columns=df_columns)
                df_char_wb = pd.DataFrame([], columns=df_columns)
                df_char_wb_left = pd.DataFrame([], columns=df_columns)
                df_char_wb_right = pd.DataFrame([], columns=df_columns)
                df_char_wb_both = pd.DataFrame([], columns=df_columns)
                df_char_wb_remove = pd.DataFrame([], columns=df_columns)
                # df_phone = pd.DataFrame([], columns=df_columns)
                # df_phone_wb = pd.DataFrame([], columns=df_columns)
                df_pos = pd.DataFrame([], columns=df_columns)
                utt_count = 0

        # Last dataframe
        df_word_list.append(df_word)
        df_char_list.append(df_char)
        df_char_wb_list.append(df_char_wb)
        df_char_wb_left_list.append(df_char_wb_left)
        df_char_wb_right_list.append(df_char_wb_right)
        df_char_wb_both_list.append(df_char_wb_both)
        df_char_wb_remove_list.append(df_char_wb_remove)
        # df_phone_list.append(df_phone)
        # df_phone_wb_list.append(df_phone_wb)
        df_pos_list.append(df_pos)

        # Concatenate all dataframes
        df_word = df_word_list[0]
        df_char = df_char_list[0]
        df_char_wb = df_char_wb_list[0]
        df_char_wb_left = df_char_wb_left_list[0]
        df_char_wb_right = df_char_wb_right_list[0]
        df_char_wb_both = df_char_wb_both_list[0]
        df_char_wb_remove = df_char_wb_remove_list[0]
        # df_phone = df_phone_list[0]
        # df_phone_wb = df_phone_wb_list[0]
        df_pos = df_pos_list[0]

        for i in df_word_list[1:]:
            df_word = pd.concat([df_word, i], axis=0)
        for i in df_char_list[1:]:
            df_char = pd.concat([df_char, i], axis=0)
        for i in df_char_wb_list[1:]:
            df_char_wb = pd.concat([df_char_wb, i], axis=0)
        for i in df_char_wb_left_list[1:]:
            df_char_wb_left = pd.concat([df_char_wb_left, i], axis=0)
        for i in df_char_wb_right_list[1:]:
            df_char_wb_right = pd.concat([df_char_wb_right, i], axis=0)
        for i in df_char_wb_both_list[1:]:
            df_char_wb_both = pd.concat([df_char_wb_both, i], axis=0)
        for i in df_char_wb_remove_list[1:]:
            df_char_wb_remove = pd.concat([df_char_wb_remove, i], axis=0)
        # for i in df_phone_list[1:]:
        #     df_phone = pd.concat([df_phone, i], axis=0)
        # for i in df_phone_wb_list[1:]:
        #     df_phone_wb = pd.concat([df_phone_wb, i], axis=0)
        for i in df_pos_list[1:]:
            df_pos = pd.concat([df_pos, i], axis=0)

        df_word.to_csv(join(csv_save_path, 'word.csv'), encoding='utf-8')
        df_char.to_csv(join(csv_save_path, 'character.csv'), encoding='utf-8')
        df_char_wb.to_csv(
            join(csv_save_path, 'character_wb.csv'), encoding='utf-8')
        df_char_wb_left.to_csv(
            join(csv_save_path, 'character_wb_left.csv'), encoding='utf-8')
        df_char_wb_right.to_csv(
            join(csv_save_path, 'character_wb_right.csv'), encoding='utf-8')
        df_char_wb_both.to_csv(
            join(csv_save_path, 'character_wb_both.csv'), encoding='utf-8')
        df_char_wb_remove.to_csv(
            join(csv_save_path, 'character_wb_remove.csv'), encoding='utf-8')
        # df_phone.to_csv(join(csv_save_path, 'phone.csv'), encoding='utf-8')
        # df_phone_wb.to_csv(join(csv_save_path, 'phone_wb.csv'), encoding='utf-8')
        df_pos.to_csv(join(csv_save_path, 'pos.csv'), encoding='utf-8')

        # TODO: word5でremove


def add_element(df, elem_list):
    series = pd.Series(elem_list, index=df.columns)
    df = df.append(series, ignore_index=True)
    return df


def read_text(text_path, vocab_save_path, data_type,
              kana2phone_path, lexicon_path=None):
    """Read transcripts (.sdb) & save files (.npy).
    Args:
        text_path (string): path to a text file of kaldi
        vocab_save_path (string): path to save vocabulary files
        data_type (string): train or dev or eval1 or eval2 or eval3
        kana2phone_path (string):
        lexicon_path (string):
    Returns:
        speaker_dict (dict): the dictionary of utterances of each speaker
            key (string) => speaker
            value (dict) => the dictionary of utterance information of each speaker
                key (string) => utterance index
                value (dict)
                    key => label type
                    value => indices
    """
    # Make vocabulary files
    word_vocab_path = mkdir_join(vocab_save_path, 'word.txt')
    char_vocab_path = mkdir_join(vocab_save_path, 'character.txt')

    trans_dict = {}
    char_set = set([])
    word_set = set([])
    word_dict = {}
    with codecs.open(text_path, 'r', 'utf-8') as f:
        for line in f:
            line = line.strip()
            utt_idx, trans_w_pos = line.split('  ')
            trans_w_pos = trans_w_pos.replace('<sp>', SHORT_PAUSE)
            trans = SPACE.join([w.split('+')[0]
                                for w in trans_w_pos.split(' ')])
            trans_pos = SPACE.join([w.split('+')[1].split('/')[0] if '+' in w else SHORT_PAUSE
                                    for w in trans_w_pos.split(' ')])
            # NOTE: word and POS sequence are the same length

            trans_dict[utt_idx] = trans

            for w in trans.split(SPACE):
                # Count word frequency
                if w not in word_dict.keys():
                    word_dict[w] = 1
                else:
                    word_dict[w] += 1
                word_set.add(w)
                char_set |= set(list(w))

    # Save vocabulary files
    if data_type == 'train':
        # word-level (threshold == 5)
        with codecs.open(word_vocab_path, 'w', 'utf-8') as f:
            word_list = sorted([w for w, freq in list(word_dict.items())
                                if freq >= 5]) + [OOV]
            for w in word_list:
                f.write('%s\n' % w)

        # character-level (char, char_wb)
        char_list = sorted(list(char_set))
        with codecs.open(char_vocab_path, 'w', 'utf-8') as f:
            for c in char_list + [OOV]:
                f.write('%s\n' % c)

    # Compute OOV rate
    with codecs.open(mkdir_join(vocab_save_path, 'oov', data_type + '.txt'), 'w', 'utf-8') as f:
        # word-level (threshold == 5)
        oov_rate = compute_oov_rate(word_dict, word_vocab_path)
        f.write('Word (freq5):\n')
        f.write('  OOV rate: %f %%\n' % oov_rate)

    # Convert to index
    print('=====> Convert to index...')
    word2idx = Word2idx(word_vocab_path)
    char2idx = Char2idx(char_vocab_path)

    for utt_idx, trans in tqdm(trans_dict.items()):
        if 'eval' in data_type:
            trans_dict[utt_idx] = {
                "word": trans,
                "char": trans.replace(SPACE, ''),
            }
            # NOTE: save as it is
        else:
            word_indices = word2idx(trans)
            char_indices = char2idx(trans.replace(SPACE, ''))

            word_indices = ' '.join(
                list(map(str, word_indices.tolist())))
            char_indices = ' '.join(
                list(map(str, char_indices.tolist())))

            trans_dict[utt_idx] = {"word": word_indices,
                                   "char": char_indices,
                                   }

    return trans_dict


def compute_oov_rate(word_dict, vocab_path):

    with codecs.open(vocab_path, 'r', 'utf-8') as f:
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
