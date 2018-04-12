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
from utils.io.labels.phone import Phone2idx
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

SPACE = '_'
SIL = 'sil'
OOV = 'OOV'
SHORT_PAUSE = '@'


def main():

    # for data_type in ['train', 'dev', 'eval1', 'eval2', 'eval3']:
    for data_type in ['eval1', 'eval2', 'eval3']:
        print('=' * 50)
        print(' ' * 20 + data_type)
        print('=' * 50)

        # Convert transcript to index
        print('=> Processing transcripts...')
        trans_dict = read_text(
            text_path=join(args.data_save_path, data_type, 'text'),
            vocab_save_path=mkdir_join(args.data_save_path, 'vocab'),
            data_type=data_type,
            kana2phone_path='./local/csj_make_trans/kana2phone',
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
        df_word20 = pd.DataFrame([], columns=df_columns)
        df_kanji = pd.DataFrame([], columns=df_columns)
        df_kanji_wb = pd.DataFrame([], columns=df_columns)
        # df_kana = pd.DataFrame([], columns=df_columns)
        # df_kana_wb = pd.DataFrame([], columns=df_columns)
        # df_phone = pd.DataFrame([], columns=df_columns)
        # df_phone_wb = pd.DataFrame([], columns=df_columns)
        df_pos = pd.DataFrame([], columns=df_columns)

        with open(join(args.data_save_path, 'feature', args.tool, data_type,
                       'frame_num.pickle'), 'rb') as f:
            frame_num_dict = pickle.load(f)

        utt_count = 0
        df_word1_list, df_word5_list = [], []
        df_word10_list, df_word15_list, df_word20_list = [], [], []
        df_kanji_list, df_kanji_wb_list = [], []
        # df_kana_list,  df_kana_wb_list = [], []
        # df_phone_list, df_phone_wb_list = [], []
        df_pos_list = []
        for utt_idx, trans_list in tqdm(trans_dict.items()):
            word1_indices, word5_indices = trans_list[:2]
            word10_indices, word15_indices, word20_indices = trans_list[2:5]
            kanji_indices, kanji_wb_indices = trans_list[5:7]
            kana_indices, kana_wb_indices = trans_list[7:9]
            phone_indices, phone_wb_indices = trans_list[9:11]
            pos_indices = trans_list[11]

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
            df_word20 = add_element(
                df_word20, [frame_num, feat_utt_save_path, word20_indices])
            df_kanji = add_element(
                df_kanji, [frame_num, feat_utt_save_path, kanji_indices])
            df_kanji_wb = add_element(
                df_kanji_wb, [frame_num, feat_utt_save_path, kanji_wb_indices])
            # df_kana = add_element(
            #     df_kana, [frame_num, feat_utt_save_path, kana_indices])
            # df_kana_wb = add_element(
            # df_kana_wb, [frame_num, feat_utt_save_path, kana_wb_indices])
            # df_phone = add_element(
            #     df_phone, [frame_num, feat_utt_save_path, phone_indices])
            # df_phone_wb = add_element(
            #     df_phone_wb, [frame_num, feat_utt_save_path, phone_wb_indices])
            df_pos = add_element(
                df_pos, [frame_num, feat_utt_save_path, pos_indices])
            utt_count += 1

            # Reset
            if utt_count == 10000:
                df_word1_list.append(df_word1)
                df_word5_list.append(df_word5)
                df_word10_list.append(df_word10)
                df_word15_list.append(df_word15)
                df_word20_list.append(df_word20)
                df_kanji_list.append(df_kanji)
                df_kanji_wb_list.append(df_kanji_wb)
                # df_kana_list.append(df_kana)
                # df_kana_wb_list.append(df_kana_wb)
                # df_phone_list.append(df_phone)
                # df_phone_wb_list.append(df_phone_wb)
                df_pos_list.append(df_pos)

                df_word1 = pd.DataFrame([], columns=df_columns)
                df_word5 = pd.DataFrame([], columns=df_columns)
                df_word10 = pd.DataFrame([], columns=df_columns)
                df_word15 = pd.DataFrame([], columns=df_columns)
                df_word20 = pd.DataFrame([], columns=df_columns)
                df_kanji = pd.DataFrame([], columns=df_columns)
                df_kanji_wb = pd.DataFrame([], columns=df_columns)
                # df_kana = pd.DataFrame([], columns=df_columns)
                # df_kana_wb = pd.DataFrame([], columns=df_columns)
                # df_phone = pd.DataFrame([], columns=df_columns)
                # df_phone_wb = pd.DataFrame([], columns=df_columns)
                df_pos = pd.DataFrame([], columns=df_columns)
                utt_count = 0

        # Last dataframe
        df_word1_list.append(df_word1)
        df_word5_list.append(df_word5)
        df_word10_list.append(df_word10)
        df_word15_list.append(df_word15)
        df_word20_list.append(df_word20)
        df_kanji_list.append(df_kanji)
        df_kanji_wb_list.append(df_kanji_wb)
        # df_kana_list.append(df_kana)
        # df_kana_wb_list.append(df_kana_wb)
        # df_phone_list.append(df_phone)
        # df_phone_wb_list.append(df_phone_wb)
        df_pos_list.append(df_pos)

        # Concatenate all dataframes
        df_word1 = df_word1_list[0]
        df_word5 = df_word5_list[0]
        df_word10 = df_word10_list[0]
        df_word15 = df_word15_list[0]
        df_word20 = df_word20_list[0]
        df_kanji = df_kanji_list[0]
        df_kanji_wb = df_kanji_wb_list[0]
        # df_kana = df_kana_list[0]
        # df_kana_wb = df_kana_wb_list[0]
        # df_phone = df_phone_list[0]
        # df_phone_wb = df_phone_wb_list[0]
        df_pos = df_pos_list[0]

        for i in df_word1_list[1:]:
            df_word1 = pd.concat([df_word1, i], axis=0)
        for i in df_word5_list[1:]:
            df_word5 = pd.concat([df_word5, i], axis=0)
        for i in df_word10_list[1:]:
            df_word10 = pd.concat([df_word10, i], axis=0)
        for i in df_word15_list[1:]:
            df_word15 = pd.concat([df_word15, i], axis=0)
        for i in df_word20_list[1:]:
            df_word20 = pd.concat([df_word20, i], axis=0)
        for i in df_kanji_list[1:]:
            df_kanji = pd.concat([df_kanji, i], axis=0)
        for i in df_kanji_wb_list[1:]:
            df_kanji_wb = pd.concat([df_kanji_wb, i], axis=0)
        # for i in df_kana_list[1:]:
        #     df_kana = pd.concat([df_kana, i], axis=0)
        # for i in df_kana_wb_list[1:]:
        #     df_kana_wb = pd.concat([df_kana_wb, i], axis=0)
        # for i in df_phone_list[1:]:
        #     df_phone = pd.concat([df_phone, i], axis=0)
        # for i in df_phone_wb_list[1:]:
        #     df_phone_wb = pd.concat([df_phone_wb, i], axis=0)
        for i in df_pos_list[1:]:
            df_pos = pd.concat([df_pos, i], axis=0)

        df_word1.to_csv(join(csv_save_path, 'word1.csv'), encoding='utf-8')
        df_word5.to_csv(join(csv_save_path, 'word5.csv'), encoding='utf-8')
        df_word10.to_csv(join(csv_save_path, 'word10.csv'), encoding='utf-8')
        df_word15.to_csv(join(csv_save_path, 'word15.csv'), encoding='utf-8')
        df_word20.to_csv(join(csv_save_path, 'word20.csv'), encoding='utf-8')
        df_kanji.to_csv(join(csv_save_path, 'kanji.csv'), encoding='utf-8')
        df_kanji_wb.to_csv(
            join(csv_save_path, 'kanji_wb.csv'), encoding='utf-8')
        # df_kana.to_csv(join(csv_save_path, 'kana.csv'), encoding='utf-8')
        # df_kana_wb.to_csv(join(csv_save_path, 'kana_wb.csv'), encoding='utf-8')
        # df_phone.to_csv(join(csv_save_path, 'phone.csv'), encoding='utf-8')
        # df_phone_wb.to_csv(join(csv_save_path, 'phone_wb.csv'), encoding='utf-8')
        df_pos.to_csv(join(csv_save_path, 'pos.csv'), encoding='utf-8')


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
        lexicon_path (string, optional):
    Returns:
        speaker_dict (dict): the dictionary of utterances of each speaker
            key (string) => speaker
            value (dict) => the dictionary of utterance information of each speaker
                key (string) => utterance index
                value (list) => list of
                    [word1_indices, word5_indices,
                     word10_indices, word15_indices, word20_indices,
                     kanji_indices, kanji_wb_indices,
                     phone_indices, phone_wb_indices,
                     pos_indices]
    """
    # Make kana set
    kana_set = set([])
    with codecs.open(kana2phone_path, 'r', 'utf-8') as f:
        for line in f:
            line = line.strip()
            kana, phone_seq = line.split('+')
            kana_set.add(kana)

    # Make vocabulary files
    word1_vocab_path = mkdir_join(vocab_save_path, 'word1.txt')
    word5_vocab_path = mkdir_join(vocab_save_path, 'word5.txt')
    word10_vocab_path = mkdir_join(vocab_save_path, 'word10.txt')
    word15_vocab_path = mkdir_join(vocab_save_path, 'word15.txt')
    word20_vocab_path = mkdir_join(vocab_save_path, 'word20.txt')
    kanji_vocab_path = mkdir_join(vocab_save_path, 'kanji.txt')
    kanji_wb_vocab_path = mkdir_join(vocab_save_path, 'kanji_wb.txt')
    # kana_vocab_path = mkdir_join(vocab_save_path, 'kana' + '.txt')
    # kana_wb_vocab_path = mkdir_join(vocab_save_path, 'kana_wb' + '.txt')
    # phone_vocab_path = mkdir_join(vocab_save_path, 'phone' + '.txt')
    # phone_wb_vocab_path = mkdir_join(vocab_save_path, 'phone_wb' + '.txt')
    pos_vocab_path = mkdir_join(vocab_save_path, 'pos' + '.txt')

    trans_dict = {}
    kanji_set = set([])
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

            trans_dict[utt_idx] = [trans, trans_pos]

            for word in trans.split(SPACE):
                # Count word frequency
                if word not in word_dict.keys():
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1

                word_set.add(word)
                kanji_set |= set(list(word))

            for pos in trans_pos.split(SPACE):
                pos_set.add(pos)

    # TODO: load lexicon

    # Save vocabulary files
    if data_type == 'train':
        # word-level (threshold == 1)
        with codecs.open(word1_vocab_path, 'w', 'utf-8') as f:
            word_list = sorted(list(word_set)) + [OOV]
            for word in word_list:
                f.write('%s\n' % word)
            # NOTE: OOV index is reserved for the dev set

        # word-level (threshold == 5)
        with codecs.open(word5_vocab_path, 'w', 'utf-8') as f:
            word_list = sorted([word for word, freq in list(word_dict.items())
                                if freq >= 5]) + [OOV]
            for word in word_list:
                f.write('%s\n' % word)

        # word-level (threshold == 10)
        with codecs.open(word10_vocab_path, 'w', 'utf-8') as f:
            word_list = sorted([word for word, freq in list(word_dict.items())
                                if freq >= 10]) + [OOV]
            for word in word_list:
                f.write('%s\n' % word)

        # word-level (threshold == 15)
        with codecs.open(word15_vocab_path, 'w', 'utf-8') as f:
            word_list = sorted([word for word, freq in list(word_dict.items())
                                if freq >= 15]) + [OOV]
            for word in word_list:
                f.write('%s\n' % word)

        # word-level (threshold == 20)
        with codecs.open(word20_vocab_path, 'w', 'utf-8') as f:
            word_list = sorted([word for word, freq in list(word_dict.items())
                                if freq >= 20]) + [OOV]
            for word in word_list:
                f.write('%s\n' % word)

        # character-level (kanji, kanji_wb)
        with codecs.open(kanji_vocab_path, 'w', 'utf-8') as f, codecs.open(kanji_wb_vocab_path, 'w', 'utf-8') as f_wb:
            kanji_list = sorted(list(kanji_set))
            for kanji in kanji_list + [OOV]:
                f.write('%s\n' % kanji)
            for kanji in kanji_list + [SPACE, OOV]:
                f_wb.write('%s\n' % kanji)

        # character-level (kana, kana_wb)
        # with codecs.open(kana_vocab_path, 'w', 'utf-8') as f, codecs.open(kana_wb_vocab_path, 'w', 'utf-8') as f_wb:
        #     kana_list = sorted(list(kana_set))
        #     for kana in kana_list:
        #         f.write('%s\n' % kana)
        #     for kana in kana_list + [SPACE]:
        #         f_wb.write('%s\n' % kana)

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
    if data_type != 'train':
        with codecs.open(mkdir_join(vocab_save_path, 'oov', data_type + '.txt'), 'w', 'utf-8') as f:

            # word-level (threshold == 1)
            oov_rate = compute_oov_rate(word_dict, word1_vocab_path)
            f.write('Word (freq1):\n')
            f.write('  OOV rate: %f %%\n' % oov_rate)

            # word-level (threshold == 5)
            oov_rate = compute_oov_rate(word_dict, word5_vocab_path)
            f.write('Word (freq5):\n')
            f.write('  OOV rate: %f %%\n' % oov_rate)

            # word-level (threshold == 10)
            oov_rate = compute_oov_rate(word_dict, word10_vocab_path)
            f.write('Word (freq10):\n')
            f.write('  OOV rate: %f %%\n' % oov_rate)

            # word-level (threshold == 15)
            oov_rate = compute_oov_rate(word_dict, word15_vocab_path)
            f.write('Word (freq15):\n')
            f.write('  OOV rate: %f %%\n' % oov_rate)

            # word-level (threshold == 20)
            oov_rate = compute_oov_rate(word_dict, word20_vocab_path)
            f.write('Word (freq20):\n')
            f.write('  OOV rate: %f %%\n' % oov_rate)

    # Convert to index
    print('=====> Convert to index...')
    word2idx_freq1 = Word2idx(word1_vocab_path)
    word2idx_freq5 = Word2idx(word5_vocab_path)
    word2idx_freq10 = Word2idx(word10_vocab_path)
    word2idx_freq15 = Word2idx(word15_vocab_path)
    word2idx_freq20 = Word2idx(word20_vocab_path)
    kanji2idx = Char2idx(kanji_vocab_path, double_letter=True)
    kanji2idx_wb = Char2idx(kanji_wb_vocab_path, double_letter=True)
    # kana2idx = Char2idx(kana_vocab_path, double_letter=True)
    # kana2idx_wb = Char2idx(kana_wb_vocab_path, double_letter=True)
    # phone2idx = Phone2idx(phone_vocab_path)
    # phone2idx_wb = Phone2idx(phone_wb_vocab_path)
    pos2idx = Word2idx(pos_vocab_path)

    for utt_idx, [trans, trans_pos] in tqdm(trans_dict.items()):
        if 'eval' in data_type:
            trans_dict[utt_idx] = [
                trans, trans, trans,
                trans, trans,
                trans.replace(SPACE, ''), trans,
                # trans_kana.replace(SPACE, ''), trans_kana,
                None, None,
                # trans_phone.replace(SIL, '').replace(
                #     '  ', ' '), trans_phone,
                None, None,
                trans_pos]
            # NOTE: save as it is
        else:
            word1_indices = word2idx_freq1(trans)
            word5_indices = word2idx_freq5(trans)
            word10_indices = word2idx_freq10(trans)
            word15_indices = word2idx_freq15(trans)
            word20_indices = word2idx_freq20(trans)
            kanji_indices = kanji2idx(trans.replace(SPACE, ''))
            kanji_wb_indices = kanji2idx_wb(trans)
            # kana_indices = kana2idx(trans_kana.replace(SPACE, ''))
            # kana_wb_indices = kana2idx_wb(trans_kana)
            # phone_indices = phone2idx(
            #     trans_phone.replace(SIL, '').replace('  ', ' '))
            # phone_wb_indices = phone2idx_wb(trans_phone)
            pos_indices = pos2idx(trans_pos)

            word1_indices = ' '.join(
                list(map(str, word1_indices.tolist())))
            word5_indices = ' '.join(
                list(map(str, word5_indices.tolist())))
            word10_indices = ' '.join(
                list(map(str, word10_indices.tolist())))
            word15_indices = ' '.join(
                list(map(str, word15_indices.tolist())))
            word20_indices = ' '.join(
                list(map(str, word20_indices.tolist())))
            kanji_indices = ' '.join(
                list(map(str, kanji_indices.tolist())))
            kanji_wb_indices = ' '.join(
                list(map(str, kanji_wb_indices.tolist())))
            # kana_indices = ' '.join(
            #     list(map(str, kana_indices.tolist())))
            # kana_wb_indices = ' '.join(
            #     list(map(str, kana_wb_indices.tolist())))
            # phone_indices = ' '.join(
            #     list(map(str, phone_indices.tolist())))
            # phone_wb_indices = ' '.join(
            #     list(map(str, phone_wb_indices.tolist())))
            pos_indices = ' '.join(
                list(map(str, pos_indices.tolist())))

            trans_dict[utt_idx] = [
                word1_indices, word5_indices, word10_indices,
                word15_indices, word20_indices,
                kanji_indices, kanji_wb_indices,
                # kana_indices, kana_wb_indices,
                None, None,
                # phone_indices, phone_wb_indices,
                None, None,
                pos_indices]

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
