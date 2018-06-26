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

SOF = 'F'
EOF = 'f'
SOD = 'D'
EOD = 'd'


def main():

    for data_type in ['train_' + args.data_size, 'dev', 'eval1', 'eval2', 'eval3']:
        print('=' * 50)
        print(' ' * 20 + data_type)
        print('=' * 50)

        # Convert transcript to index
        print('=> Processing transcripts...')
        trans_dict = read_text(
            text_path=join(args.data_save_path, data_type, 'text'),
            vocab_save_path=mkdir_join(
                args.data_save_path, 'vocab', args.data_size),
            data_type=data_type,
            kana2phone_path='./local/csj_make_trans/kana2phone',
            lexicon_path=None)

        # Make dataset file (.csv)
        print('=> Saving dataset files...')
        csv_save_path = mkdir_join(
            args.data_save_path, 'dataset', args.tool, args.data_size, data_type.split('_')[0])

        df_columns = ['frame_num', 'input_path', 'transcript']
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

        with open(join(args.data_save_path, 'feature', args.tool, args.data_size, data_type.split('_')[0], 'frame_num.pickle'), 'rb') as f:
            frame_num_dict = pickle.load(f)

        utt_count = 0
        df_word_list = []
        df_char_list, df_char_wb_list = [], []
        df_char_wb_left_list, df_char_wb_right_list = [], []
        df_char_wb_both_list, df_char_wb_remove_list = [], []
        # df_phone_list, df_phone_wb_list = [], []
        df_pos_list = []
        for utt_idx, trans in tqdm(trans_dict.items()):
            speaker = utt_idx.split('_')[0]
            feat_utt_save_path = join(
                args.data_save_path, 'feature', args.tool, args.data_size, data_type.split('_')[0], speaker, utt_idx + '.npy')
            frame_num = frame_num_dict[utt_idx]

            if not isfile(feat_utt_save_path):
                raise ValueError('There is no file: %s' % feat_utt_save_path)

            df_word = add_element(
                df_word, [frame_num, feat_utt_save_path, trans['word']])
            df_char = add_element(
                df_char, [frame_num, feat_utt_save_path, trans['char']])
            df_char_wb = add_element(
                df_char_wb, [frame_num, feat_utt_save_path, trans['char_wb']])
            df_char_wb_left = add_element(
                df_char_wb_left, [frame_num, feat_utt_save_path, trans['char_wb_left']])
            df_char_wb_right = add_element(
                df_char_wb_right, [frame_num, feat_utt_save_path, trans['char_wb_right']])
            df_char_wb_both = add_element(
                df_char_wb_both, [frame_num, feat_utt_save_path, trans['char_wb_both']])
            df_char_wb_remove = add_element(
                df_char_wb_remove, [frame_num, feat_utt_save_path, trans['char_wb_remove']])
            # df_phone = add_element(
            #     df_phone, [frame_num, feat_utt_save_path, phone_indices])
            # df_phone_wb = add_element(
            #     df_phone_wb, [frame_num, feat_utt_save_path, phone_wb_indices])
            df_pos = add_element(
                df_pos, [frame_num, feat_utt_save_path,  trans['pos']])
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
    # Make kana set
    kana_set = set([])
    with codecs.open(kana2phone_path, 'r', 'utf-8') as f:
        for line in f:
            line = line.strip()
            kana, phone_seq = line.split('+')
            kana_set.add(kana)

    # Make vocabulary files
    word_vocab_path = mkdir_join(vocab_save_path, 'word.txt')
    char_vocab_path = mkdir_join(vocab_save_path, 'character.txt')
    char_wb_vocab_path = mkdir_join(vocab_save_path, 'character_wb.txt')
    char_wb_left_vocab_path = mkdir_join(
        vocab_save_path, 'character_wb_left.txt')
    char_wb_right_vocab_path = mkdir_join(
        vocab_save_path, 'character_wb_right.txt')
    char_wb_both_vocab_path = mkdir_join(
        vocab_save_path, 'character_wb_both.txt')
    char_wb_remove_vocab_path = mkdir_join(
        vocab_save_path, 'character_wb_remove.txt')
    # phone_vocab_path = mkdir_join(vocab_save_path, 'phone' + '.txt')
    # phone_wb_vocab_path = mkdir_join(vocab_save_path, 'phone_wb' + '.txt')
    pos_vocab_path = mkdir_join(vocab_save_path, 'pos' + '.txt')

    trans_dict = {}
    char_set = set([])
    char_set_remove = set([])
    word_set = set([])
    pos_set = set([])
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

            ###################################
            # with filler and disfluency
            ###################################
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
                trans_right_list.append(w_right)
                trans_both_list.append(w_both)
            trans_left = SPACE.join(trans_left_list)
            trans_right = SPACE.join(trans_right_list)
            trans_both = SPACE.join(trans_both_list)
            trans_remove = SPACE.join(trans_remove_list)

            trans_dict[utt_idx] = [trans, trans_pos,
                                   trans_left, trans_right, trans_both, trans_remove]

            for w in trans.split(SPACE):
                # Count word frequency
                if w not in word_dict.keys():
                    word_dict[w] = 1
                else:
                    word_dict[w] += 1
                word_set.add(w)
                char_set |= set(list(w))

            for w in trans_remove.split(SPACE):
                char_set_remove |= set(list(w))

            for pos in trans_pos.split(SPACE):
                pos_set.add(pos)

    # TODO: load lexicon

    # Save vocabulary files
    if 'train' in data_type:
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
        with codecs.open(char_wb_vocab_path, 'w', 'utf-8') as f:
            for c in char_list + [SPACE, OOV]:
                f.write('%s\n' % c)

        # character-level (char_wb + left, right, both, remove)
        with codecs.open(char_wb_left_vocab_path, 'w', 'utf-8') as f:
            for c in char_list + [SPACE, OOV, SOF, SOD]:
                f.write('%s\n' % c)
        with codecs.open(char_wb_right_vocab_path, 'w', 'utf-8') as f:
            for c in char_list + [SPACE, OOV, SOF, SOD]:
                f.write('%s\n' % c)
        with codecs.open(char_wb_both_vocab_path, 'w', 'utf-8') as f:
            for c in char_list + [SPACE, OOV, SOF, SOD, EOF, EOD]:
                f.write('%s\n' % c)
        with codecs.open(char_wb_remove_vocab_path, 'w', 'utf-8') as f:
            char_list_remove = sorted(list(char_set_remove))
            for c in char_list_remove + [SPACE, OOV]:
                f.write('%s\n' % c)

        # phone-level (phone, phone_wb)
        # with codecs.open(phone_vocab_path, 'w', 'utf-8') as f, codecs.open(phone_wb_vocab_path, 'w', 'utf-8') as f_wb:
        #     phone_list = sorted(list(phone_set))
        #     for phone in phone_list:
        #         f.write('%s\n' % phone)
        #     for phone in phone_list + [SIL]:
        #         f_wb.write('%s\n' % phone)

        # pos-level
        with codecs.open(pos_vocab_path, 'w', 'utf-8') as f:
            pos_list = sorted(list(pos_set))
            for pos in pos_list:
                f.write('%s\n' % pos)

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
    char2idx_wb = Char2idx(char_wb_vocab_path)
    char2idx_wb_left = Char2idx(char_wb_left_vocab_path)
    char2idx_wb_right = Char2idx(char_wb_right_vocab_path)
    char2idx_wb_both = Char2idx(char_wb_both_vocab_path)
    char2idx_wb_remove = Char2idx(char_wb_remove_vocab_path)
    # phone2idx = Phone2idx(phone_vocab_path)
    # phone2idx_wb = Phone2idx(phone_wb_vocab_path)
    pos2idx = Word2idx(pos_vocab_path)

    for utt_idx, [trans, trans_pos, trans_left, trans_right, trans_both, trans_remove] in tqdm(trans_dict.items()):
        if 'eval' in data_type:
            trans_dict[utt_idx] = {
                "word": trans,
                "char": trans.replace(SPACE, ''),
                "char_wb": trans,
                "char_wb_left": trans,
                "char_wb_right": trans,
                "char_wb_both": trans,
                "char_wb_remove": trans_remove,
                "phone": None,
                # "phone": trans_phone,
                "phone_wb": None,
                # "phone_wb": trans_phone.replace(SIL, '').replace('  ', ' '),
                "pos": trans_pos,
            }
            # NOTE: save as it is
        else:
            word_indices = word2idx(trans)
            char_indices = char2idx(trans.replace(SPACE, ''))
            char_wb_indices = char2idx_wb(trans)
            char_wb_left_indices = char2idx_wb_left(trans_left)
            char_wb_right_indices = char2idx_wb_right(trans_right)
            char_wb_both_indices = char2idx_wb_both(trans_both)
            char_wb_remove_indices = char2idx_wb_remove(trans_remove)
            # phone_indices = phone2idx(
            #     trans_phone.replace(SIL, '').replace('  ', ' '))
            # phone_wb_indices = phone2idx_wb(trans_phone)
            pos_indices = pos2idx(trans_pos)

            word_indices = ' '.join(list(map(str, word_indices)))
            char_indices = ' '.join(list(map(str, char_indices)))
            char_wb_indices = ' '.join(list(map(str, char_wb_indices)))
            char_wb_left_indices = ' '.join(
                list(map(str, char_wb_left_indices)))
            char_wb_right_indices = ' '.join(
                list(map(str, char_wb_right_indices)))
            char_wb_both_indices = ' '.join(
                list(map(str, char_wb_both_indices)))
            char_wb_remove_indices = ' '.join(
                list(map(str, char_wb_remove_indices)))
            # phone_indices = ' '.join(
            # list(map(str, phone_indices)))
            # phone_wb_indices = ' '.join(
            # list(map(str, phone_wb_indices)))
            pos_indices = ' '.join(list(map(str, pos_indices)))

            trans_dict[utt_idx] = {
                "word": word_indices,
                "char": char_indices,
                "char_wb": char_wb_indices,
                "char_wb_left": char_wb_left_indices,
                "char_wb_right": char_wb_right_indices,
                "char_wb_both": char_wb_both_indices,
                "char_wb_remove": char_wb_remove_indices,
                # "phone": phone_indices,
                # "phone_wb": phone_wb_indices,
                "pos": pos_indices,
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
