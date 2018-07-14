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
        print('=' * 50)
        print(' ' * 20 + data_type)
        print('=' * 50)

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
        df_char_nowb = pd.DataFrame([], columns=df_columns)
        df_char_left = pd.DataFrame([], columns=df_columns)
        df_char_right = pd.DataFrame([], columns=df_columns)
        df_char_both = pd.DataFrame([], columns=df_columns)
        df_char_remove = pd.DataFrame([], columns=df_columns)
        df_phone = pd.DataFrame([], columns=df_columns)
        df_phone_nowb = pd.DataFrame([], columns=df_columns)
        df_pos = pd.DataFrame([], columns=df_columns)

        with open(join(args.data_save_path, 'feature', args.tool, args.data_size, data_type.split('_')[0], 'frame_num.pickle'), 'rb') as f:
            frame_num_dict = pickle.load(f)

        utt_count = 0
        df_char_list, df_char_nowb_list = [], []
        df_char_left_list, df_char_right_list = [], []
        df_char_both_list, df_char_remove_list = [], []
        df_phone_list, df_phone_nowb_list = [], []
        df_pos_list = []
        for utt_idx, trans in tqdm(trans_dict.items()):
            speaker = utt_idx.split('_')[0]
            feat_utt_save_path = join(
                args.data_save_path, 'feature', args.tool, args.data_size, data_type.split('_')[0], speaker, utt_idx + '.npy')
            x_len = frame_num_dict[utt_idx]

            if not isfile(feat_utt_save_path):
                raise ValueError('There is no file: %s' % feat_utt_save_path)

            df_char = add_element(
                df_char, [x_len, feat_utt_save_path, trans['char']])
            df_char_nowb = add_element(
                df_char_nowb, [x_len, feat_utt_save_path, trans['char_nowb']])
            df_char_left = add_element(
                df_char_left, [x_len, feat_utt_save_path, trans['char_left']])
            df_char_right = add_element(
                df_char_right, [x_len, feat_utt_save_path, trans['char_right']])
            df_char_both = add_element(
                df_char_both, [x_len, feat_utt_save_path, trans['char_both']])
            df_char_remove = add_element(
                df_char_remove, [x_len, feat_utt_save_path, trans['char_remove']])
            df_phone = add_element(
                df_phone, [x_len, feat_utt_save_path, trans['phone']])
            df_phone_nowb = add_element(
                df_phone_nowb, [x_len, feat_utt_save_path, trans['phone_nowb']])
            df_pos = add_element(
                df_pos, [x_len, feat_utt_save_path,  trans['pos']])
            utt_count += 1

            # Reset
            if utt_count == 10000:
                df_char_list.append(df_char)
                df_char_nowb_list.append(df_char_nowb)
                df_char_left_list.append(df_char_left)
                df_char_right_list.append(df_char_right)
                df_char_both_list.append(df_char_both)
                df_char_remove_list.append(df_char_remove)
                df_phone_list.append(df_phone)
                df_phone_nowb_list.append(df_phone_nowb)
                df_pos_list.append(df_pos)

                df_char = pd.DataFrame([], columns=df_columns)
                df_char_nowb = pd.DataFrame([], columns=df_columns)
                df_char_left = pd.DataFrame([], columns=df_columns)
                df_char_right = pd.DataFrame([], columns=df_columns)
                df_char_both = pd.DataFrame([], columns=df_columns)
                df_char_remove = pd.DataFrame([], columns=df_columns)
                df_phone = pd.DataFrame([], columns=df_columns)
                df_phone_nowb = pd.DataFrame([], columns=df_columns)
                df_pos = pd.DataFrame([], columns=df_columns)
                utt_count = 0

        # Last dataframe
        df_char_list.append(df_char)
        df_char_nowb_list.append(df_char_nowb)
        df_char_left_list.append(df_char_left)
        df_char_right_list.append(df_char_right)
        df_char_both_list.append(df_char_both)
        df_char_remove_list.append(df_char_remove)
        df_phone_list.append(df_phone)
        df_phone_nowb_list.append(df_phone_nowb)
        df_pos_list.append(df_pos)

        # Concatenate all dataframes
        df_char = df_char_list[0]
        df_char_nowb = df_char_nowb_list[0]
        df_char_left = df_char_left_list[0]
        df_char_right = df_char_right_list[0]
        df_char_both = df_char_both_list[0]
        df_char_remove = df_char_remove_list[0]
        df_phone = df_phone_list[0]
        df_phone_nowb = df_phone_nowb_list[0]
        df_pos = df_pos_list[0]

        for i in df_char_list[1:]:
            df_char = pd.concat([df_char, i], axis=0)
        for i in df_char_nowb_list[1:]:
            df_char_nowb = pd.concat([df_char_nowb, i], axis=0)
        for i in df_char_left_list[1:]:
            df_char_left = pd.concat([df_char_left, i], axis=0)
        for i in df_char_right_list[1:]:
            df_char_right = pd.concat([df_char_right, i], axis=0)
        for i in df_char_both_list[1:]:
            df_char_both = pd.concat([df_char_both, i], axis=0)
        for i in df_char_remove_list[1:]:
            df_char_remove = pd.concat([df_char_remove, i], axis=0)
        for i in df_phone_list[1:]:
            df_phone = pd.concat([df_phone, i], axis=0)
        for i in df_phone_nowb_list[1:]:
            df_phone_nowb = pd.concat([df_phone_nowb, i], axis=0)
        for i in df_pos_list[1:]:
            df_pos = pd.concat([df_pos, i], axis=0)

        df_char.to_csv(join(csv_save_path, 'word1k.csv'), encoding='utf-8')
        df_char.to_csv(join(csv_save_path, 'word5k.csv'), encoding='utf-8')
        df_char.to_csv(join(csv_save_path, 'word10k.csv'), encoding='utf-8')
        df_char.to_csv(join(csv_save_path, 'word15k.csv'), encoding='utf-8')
        df_char.to_csv(join(csv_save_path, 'word20k.csv'), encoding='utf-8')
        df_char.to_csv(join(csv_save_path, 'word25k.csv'), encoding='utf-8')
        df_char.to_csv(join(csv_save_path, 'character.csv'), encoding='utf-8')
        df_char_nowb.to_csv(
            join(csv_save_path, 'character_nowb.csv'), encoding='utf-8')
        df_char_left.to_csv(
            join(csv_save_path, 'character_left.csv'), encoding='utf-8')
        df_char_right.to_csv(
            join(csv_save_path, 'character_right.csv'), encoding='utf-8')
        df_char_both.to_csv(
            join(csv_save_path, 'character_both.csv'), encoding='utf-8')
        df_char_remove.to_csv(
            join(csv_save_path, 'character_remove.csv'), encoding='utf-8')
        df_phone.to_csv(join(csv_save_path, 'phone.csv'), encoding='utf-8')
        df_phone_nowb.to_csv(
            join(csv_save_path, 'phone_nowb.csv'), encoding='utf-8')
        df_pos.to_csv(join(csv_save_path, 'pos.csv'), encoding='utf-8')


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

    # Make kana set
    kana_set = set([])
    with codecs.open(kana2phone_path, 'r', 'utf-8') as f:
        for line in f:
            line = line.strip()
            kana, phone_seq = line.split('+')
            kana_set.add(kana)

    # Make vocabulary files
    word1k_vocab_path = mkdir_join(vocab_path, 'word1k.txt')
    word5k_vocab_path = mkdir_join(vocab_path, 'word5k.txt')
    word10k_vocab_path = mkdir_join(vocab_path, 'word10k.txt')
    word15k_vocab_path = mkdir_join(vocab_path, 'word15k.txt')
    word20k_vocab_path = mkdir_join(vocab_path, 'word20k.txt')
    word25k_vocab_path = mkdir_join(vocab_path, 'word25k.txt')
    char_vocab_path = mkdir_join(vocab_path, 'character.txt')
    char_nowb_vocab_path = mkdir_join(vocab_path, 'character_nowb.txt')
    char_left_vocab_path = mkdir_join(vocab_path, 'character_left.txt')
    char_right_vocab_path = mkdir_join(vocab_path, 'character_right.txt')
    char_both_vocab_path = mkdir_join(vocab_path, 'character_both.txt')
    char_remove_vocab_path = mkdir_join(vocab_path, 'character_remove.txt')
    phone_vocab_path = mkdir_join(vocab_path, 'phone.txt')
    phone_nowb_vocab_path = mkdir_join(vocab_path, 'phone_nowb.txt')
    pos_vocab_path = mkdir_join(vocab_path, 'pos.txt')

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
                trans_right_list.append(w_right)
                trans_both_list.append(w_both)
            trans_left = SPACE.join(trans_left_list).lower()
            trans_right = SPACE.join(trans_right_list).lower()
            trans_both = SPACE.join(trans_both_list).lower()
            trans_remove = SPACE.join(trans_remove_list).lower()

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

            # Phoneme
            trans_phone = ''
            if 'eval' not in data_type:
                words = trans_w_pos.split(' ')
                for i, w in enumerate(words):
                    trans_phone += word2phone[w].replace(' ', SPACE)
                    if i != len(words) - 1:
                        trans_phone += SPACE + WORD_BOUNDARY + SPACE

            for pos in trans_pos.split(SPACE):
                pos_set.add(pos)

            trans_dict[utt_idx] = {
                "word": trans,
                "char_nowb": trans.replace(SPACE, ''),
                "char": trans,
                "char_left": trans_left,
                "char_right": trans_right,
                "char_both": trans_both,
                "char_remove": trans_remove,
                "phone_nowb": trans_phone.replace('_' + WORD_BOUNDARY + '_', '_'),
                "phone": trans_phone,
                "pos": trans_pos,
            }

    # Save vocabulary files
    if 'train' in data_type:
        # word-level
        with codecs.open(word1k_vocab_path, 'w', 'utf-8') as f:
            word_list = sorted(sorted(list(word_dict.keys()), key=lambda x: word_dict[x], reverse=True)[
                :998]) + [OOV]
            for w in word_list:
                f.write('%s\n' % w)
        with codecs.open(word5k_vocab_path, 'w', 'utf-8') as f:
            word_list = sorted(sorted(list(word_dict.keys()), key=lambda x: word_dict[x], reverse=True)[
                :4998]) + [OOV]
            for w in word_list:
                f.write('%s\n' % w)
        with codecs.open(word10k_vocab_path, 'w', 'utf-8') as f:
            word_list = sorted(sorted(list(word_dict.keys()), key=lambda x: word_dict[x], reverse=True)[
                :9998]) + [OOV]
            for w in word_list:
                f.write('%s\n' % w)
        with codecs.open(word15k_vocab_path, 'w', 'utf-8') as f:
            word_list = sorted(sorted(list(word_dict.keys()), key=lambda x: word_dict[x], reverse=True)[
                :14998]) + [OOV]
            for w in word_list:
                f.write('%s\n' % w)
        with codecs.open(word20k_vocab_path, 'w', 'utf-8') as f:
            word_list = sorted(sorted(list(word_dict.keys()), key=lambda x: word_dict[x], reverse=True)[
                :19998]) + [OOV]
            for w in word_list:
                f.write('%s\n' % w)
        with codecs.open(word25k_vocab_path, 'w', 'utf-8') as f:
            word_list = sorted(sorted(list(word_dict.keys()), key=lambda x: word_dict[x], reverse=True)[
                :24998]) + [OOV]
            for w in word_list:
                f.write('%s\n' % w)

        # character-level (char, char_nowb)
        char_list = sorted(list(char_set))
        with codecs.open(char_nowb_vocab_path, 'w', 'utf-8') as f:
            for c in char_list + [OOV]:
                f.write('%s\n' % c)
        with codecs.open(char_vocab_path, 'w', 'utf-8') as f:
            for c in char_list + [SPACE, OOV]:
                f.write('%s\n' % c)

        # character-level (char + left, right, both, remove)
        with codecs.open(char_left_vocab_path, 'w', 'utf-8') as f:
            for c in char_list + [SPACE, OOV, SOF, SOD]:
                f.write('%s\n' % c)
        with codecs.open(char_right_vocab_path, 'w', 'utf-8') as f:
            for c in char_list + [SPACE, OOV, SOF, SOD]:
                f.write('%s\n' % c)
        with codecs.open(char_both_vocab_path, 'w', 'utf-8') as f:
            for c in char_list + [SPACE, OOV, SOF, SOD, EOF, EOD]:
                f.write('%s\n' % c)
        with codecs.open(char_remove_vocab_path, 'w', 'utf-8') as f:
            char_list_remove = sorted(list(char_set_remove))
            for c in char_list_remove + [SPACE, OOV]:
                f.write('%s\n' % c)

        # phone-level (phone, phone_wb)
        phone_list = sorted(list(phone_set))
        with codecs.open(phone_nowb_vocab_path, 'w', 'utf-8') as f:
            for p in phone_list:
                f.write('%s\n' % p)
        with codecs.open(phone_vocab_path, 'w', 'utf-8') as f:
            for p in phone_list + [WORD_BOUNDARY]:
                f.write('%s\n' % p)

        # pos-level
        with codecs.open(pos_vocab_path, 'w', 'utf-8') as f:
            pos_list = sorted(list(pos_set))
            for pos in pos_list:
                f.write('%s\n' % pos)

    # Compute OOV rate
    with codecs.open(mkdir_join(vocab_path, 'oov', data_type + '.txt'), 'w', 'utf-8') as f:
        oov_rate = compute_oov_rate(word_dict, word1k_vocab_path)
        f.write('Word (1k):\n')
        f.write('  OOV rate: %f %%\n' % oov_rate)

        oov_rate = compute_oov_rate(word_dict, word5k_vocab_path)
        f.write('Word (5k):\n')
        f.write('  OOV rate: %f %%\n' % oov_rate)

        oov_rate = compute_oov_rate(word_dict, word10k_vocab_path)
        f.write('Word (10k):\n')
        f.write('  OOV rate: %f %%\n' % oov_rate)

        oov_rate = compute_oov_rate(word_dict, word15k_vocab_path)
        f.write('Word (15k):\n')
        f.write('  OOV rate: %f %%\n' % oov_rate)

        oov_rate = compute_oov_rate(word_dict, word20k_vocab_path)
        f.write('Word (20k):\n')
        f.write('  OOV rate: %f %%\n' % oov_rate)

        oov_rate = compute_oov_rate(word_dict, word25k_vocab_path)
        f.write('Word (25k):\n')
        f.write('  OOV rate: %f %%\n' % oov_rate)

    # test for vocabulary
    # if 'eval' not in data_type:
    #     print('=====> Convert to index...')
    #     word1k2idx = Word2idx(word1k_vocab_path)
    #     word5k2idx = Word2idx(word5k_vocab_path)
    #     word10k2idx = Word2idx(word10k_vocab_path)
    #     word15k2idx = Word2idx(word15k_vocab_path)
    #     word20k2idx = Word2idx(word20k_vocab_path)
    #     word25k2idx = Word2idx(word25k_vocab_path)
    #     char2idx_nowb = Char2idx(char_nowb_vocab_path)
    #     char2idx = Char2idx(char_vocab_path)
    #     char2idx_left = Char2idx(char_left_vocab_path)
    #     char2idx_right = Char2idx(char_right_vocab_path)
    #     char2idx_both = Char2idx(char_both_vocab_path)
    #     char2idx_remove = Char2idx(char_remove_vocab_path)
    #     phone2idx_nowb = Phone2idx(phone_nowb_vocab_path)
    #     phone2idx = Phone2idx(phone_vocab_path)
    #     pos2idx = Word2idx(pos_vocab_path)
    #
    #     for utt_idx in tqdm(trans_dict.keys()):
    #         word1k2idx(trans_dict[utt_idx]['word'])
    #         word5k2idx(trans_dict[utt_idx]['word'])
    #         word10k2idx(trans_dict[utt_idx]['word'])
    #         word15k2idx(trans_dict[utt_idx]['word'])
    #         word20k2idx(trans_dict[utt_idx]['word'])
    #         word25k2idx(trans_dict[utt_idx]['word'])
    #         char2idx_nowb(trans_dict[utt_idx]['char_nowb'])
    #         char2idx(trans_dict[utt_idx]['char'])
    #         char2idx_left(trans_dict[utt_idx]['char_left'])
    #         char2idx_right(trans_dict[utt_idx]['char_right'])
    #         char2idx_both(trans_dict[utt_idx]['char_both'])
    #         char2idx_remove(trans_dict[utt_idx]['char_remove'])
    #         phone2idx_nowb(trans_dict[utt_idx]['phone_nowb'])
    #         phone2idx(trans_dict[utt_idx]['phone'])
    #         pos2idx(trans_dict[utt_idx]['pos'])

    return trans_dict


def compute_oov_rate(word_dict, vocab_path):

    with codecs.open(vocab_path, 'r', 'utf-8') as f:
        vocab_set = set([])
        for line in f:
            w = line.strip()
            vocab_set.add(w)

    oov_count = 0
    word_num = 0
    for w, freq in word_dict.items():
        word_num += freq
        if w not in vocab_set:
            oov_count += freq

    oov_rate = oov_count * 100 / word_num
    return oov_rate


if __name__ == '__main__':

    main()
