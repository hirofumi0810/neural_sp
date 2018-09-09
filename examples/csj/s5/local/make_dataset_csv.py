#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make dataset CSV files (CSJ corpus)."""

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
from src.utils.io.labels.phone import Phone2idx
from src.utils.io.labels.character import Char2idx
from src.utils.io.labels.word import Word2idx
from src.utils.directory import mkdir_join

parser = argparse.ArgumentParser()
parser.add_argument('--data_save_path', type=str,
                    help='path to save dataset')
parser.add_argument('--data_size', type=str,
                    choices=['aps_other', 'aps', 'all_except_dialog', 'all'])
parser.add_argument('--tool', type=str,
                    choices=['htk', 'python_speech_features', 'librosa'])
args = parser.parse_args()

SPACE = '_'
SIL = 'sil'
OOV = 'OOV'
SHORT_PAUSE = '@'
WORD_BOUNDARY = 'wb'

SOF = 'F'
EOF = 'f'
SOD = 'D'
EOD = 'd'


def main():

    for data_type in ['train_' + args.data_size, 'dev', 'eval1', 'eval2', 'eval3']:
        # Convert transcript to index
        print('=> Processing transcripts...')
        trans_dict = read_text(
            text_path=join(args.data_save_path, data_type, 'text'),
            vocab_path=mkdir_join(
                args.data_save_path, 'vocab', args.data_size),
            data_type=data_type,
            kana2phone_path='./local/csj_make_trans/kana2phone',
            lexicon_path=join(args.data_save_path, 'local/dict_nosp/lexicon.txt'))

        # Make dataset file (.csv)
        print('=> Saving dataset files...')
        csv_save_path = mkdir_join(
            args.data_save_path, 'dataset', args.tool, args.data_size, data_type.split('_')[0])

        df_columns = ['frame_num', 'input_path', 'transcript']
        df_char = pd.DataFrame([], columns=df_columns)

        with open(join(args.data_save_path, 'feature', args.tool, args.data_size, data_type.split('_')[0], 'frame_num.pickle'), 'rb') as f:
            frame_num_dict = pickle.load(f)

        utt_count = 0
        df_char_list, df_char_nowb_list = [], []
        for utt_idx, trans in tqdm(trans_dict.items()):
            speaker = utt_idx.split('_')[0]
            feat_utt_save_path = join(
                args.data_save_path, 'feature', args.tool, args.data_size, data_type.split('_')[0], speaker, utt_idx + '.npy')
            x_len = frame_num_dict[utt_idx]

            if not isfile(feat_utt_save_path):
                raise ValueError('There is no file: %s' % feat_utt_save_path)

            df_char = add_element(
                df_char, [x_len, feat_utt_save_path, trans['char']])
            utt_count += 1

            # Reset
            if utt_count == 10000:
                df_char_list.append(df_char)

                df_char = pd.DataFrame([], columns=df_columns)
                utt_count = 0

        # Last dataframe
        df_char_list.append(df_char)

        # Concatenate all dataframes
        df_char = df_char_list[0]

        for i in df_char_list[1:]:
            df_char = pd.concat([df_char, i], axis=0)

        df_char.to_csv(join(csv_save_path, 'word1k.csv'), encoding='utf-8')


def add_element(df, elem_list):
    series = pd.Series(elem_list, index=df.columns)
    df = df.append(series, ignore_index=True)
    return df


def read_text(text_path, vocab_path, data_type, kana2phone_path, lexicon_path):
    """Read transcripts (.sdb) & save files (.npy).
    Args:
        text_path (string): path to a text file of kaldi
        vocab_path (string): path to save vocabulary files
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
    # Read the lexicon file
    word2phone = {}
    phone_set = set([])
    with open(lexicon_path, 'r') as f:
        for line in f:
            word = line.strip().split(' ')[0]
            phones = line.strip().split(' ')[1:]
            phone_set |= set(phones)
            word2phone[word] = ' '.join(phones)
        phone_set.add(SHORT_PAUSE)
        word2phone[SHORT_PAUSE] = SHORT_PAUSE

    # Make vocabulary files
    word1k_vocab_path = mkdir_join(vocab_path, 'word1k.txt')

    trans_dict = {}
    char_set = set([])
    word_dict = {}
    with codecs.open(text_path, 'r', 'utf-8') as f:
        for line in f:
            line = line.strip()
            utt_idx, trans_w_pos = line.split('  ')
            trans_w_pos = trans_w_pos.replace('<sp>', SHORT_PAUSE)
            # NOTE: do not convert trans_w_pos to lowercase here
            trans = SPACE.join([w.split('+')[0]
                                for w in trans_w_pos.split(' ')]).lower()
            trans_pos = SPACE.join([w.split('+')[1].split('/')[0] if '+' in w else SHORT_PAUSE
                                    for w in trans_w_pos.split(' ')])
            # NOTE: word and POS sequence are the same length

            # with filler and disfluency
            trans_left_list, trans_right_list, trans_both_list, trans_remove_list = [], [], [], []
            for w in trans_w_pos.split(' '):
                if '言いよどみ' in w:
                    w_left = SOD + w.split('+')[0]
                    w_right = w.split('+')[0] + EOD
                    w_both = SOD + w.split('+')[0] + EOD
                elif '感動詞' in w:
                    w_left = SOF + w.split('+')[0]
                    w_right = w.split('+')[0] + EOF
                    w_both = SOF + w.split('+')[0] + EOF
                else:
                    w_left = w.split('+')[0]
                    w_right = w.split('+')[0]
                    w_both = w.split('+')[0]
                    trans_remove_list.append(w.split('+')[0])
                trans_left_list.append(w_left)
            trans_left = SPACE.join(trans_left_list).lower()

            # Phoneme
            trans_phone = ''
            if 'eval' not in data_type:
                words = trans_w_pos.split(' ')
                for i, w in enumerate(words):
                    trans_phone += word2phone[w].replace(' ', SPACE)
                    if i != len(words) - 1:
                        trans_phone += SPACE + WORD_BOUNDARY + SPACE

    # Compute OOV rate
    with codecs.open(mkdir_join(vocab_path, 'oov', data_type + '.txt'), 'w', 'utf-8') as f:
        oov_rate = compute_oov_rate(word_dict, word1k_vocab_path)
        f.write('Word (1k):\n')
        f.write('  OOV rate: %f %%\n' % oov_rate)

    # test for vocabulary
    # if 'eval' not in data_type:
    #     print('=====> Convert to index...')
    #     word1k2idx = Word2idx(word1k_vocab_path)
    #
    #     for utt_idx in tqdm(trans_dict.keys()):
    #         word1k2idx(trans_dict[utt_idx]['word'])

    return trans_dict


if __name__ == '__main__':

    main()
