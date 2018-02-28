#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, basename, isfile
import sys
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import codecs
from collections import OrderedDict


parser = argparse.ArgumentParser()
parser.add_argument('--text_path', type=str, help='path to transcriptions')

args = parser.parse_args()


def main():
    for data_type in ['train', 'dev', 'eval1', 'eval2', 'eval3']:

        # Read transcriptions
        with codecs.open(args.text_path, 'r', 'utf-8') as f:
            for line in f:
                line = line.strip()

        # Make vocabulary files
        kanji_vocab_file_path = mkdir_join(
            root_path, 'vocab', data_size, 'kanji' + '.txt')
        kanji_div_vocab_file_path = mkdir_join(
            root_path, 'vocab', data_size, 'kanji_divide' + '.txt')
        kana_vocab_file_path = mkdir_join(
            root_path, 'vocab', data_size, 'kana' + '.txt')
        kana_div_vocab_file_path = mkdir_join(
            root_path, 'vocab', data_size, 'kana_divide' + '.txt')
        phone_vocab_file_path = mkdir_join(
            root_path, 'vocab', data_size, 'phone' + '.txt')
        phone_div_vocab_file_path = mkdir_join(
            root_path, 'vocab', data_size, 'phone_divide' + '.txt')
        word_freq1_vocab_file_path = mkdir_join(
            root_path, 'vocab', data_size, 'word_freq1' + '.txt')
        word_freq5_vocab_file_path = mkdir_join(
            root_path, 'vocab', data_size, 'word_freq5' + '.txt')
        word_freq10_vocab_file_path = mkdir_join(
            root_path, 'vocab', data_size, 'word_freq10' + '.txt')
        word_freq15_vocab_file_path = mkdir_join(
            root_path, 'vocab', data_size, 'word_freq15' + '.txt')

        # Reserve some indices
        char_set.discard(SPACE)

    if save_vocab_file:
        # character-level (kanji, kanji_divide)
        kanji_set = set([])
        for char in char_set:
            if (not is_hiragana(char)) and (not is_katakana(char)):
                kanji_set.add(char)
        for kana in kana_list:
            kanji_set.add(kana)
            kanji_set.add(jaconv.kata2hira(kana))
        with open(kanji_vocab_file_path, 'w') as f, open(kanji_div_vocab_file_path, 'w') as f_div:
            kanji_list = sorted(list(kanji_set))
            for kanji in kanji_list:
                f.write('%s\n' % kanji)
            for kanji in kanji_list + [SPACE]:
                f_div.write('%s\n' % kanji)

        # character-level (kana, kana_divide)
        with open(kana_vocab_file_path, 'w') as f, open(kana_div_vocab_file_path, 'w') as f_div:
            kana_list_tmp = sorted(kana_list)
            for kana in kana_list_tmp:
                f.write('%s\n' % kana)
            for kana in kana_list_tmp + [SPACE]:
                f_div.write('%s\n' % kana)

        # phone-level (phone, phone_divide)
        with open(phone_vocab_file_path, 'w') as f, open(phone_div_vocab_file_path, 'w') as f_div:
            phone_list = sorted(list(phone_set))
            for phone in phone_list:
                f.write('%s\n' % phone)
            for phone in phone_list + [SIL]:
                f_div.write('%s\n' % phone)

        # word-level (threshold == 1)
        with open(word_freq1_vocab_file_path, 'w') as f:
            vocab_list = sorted(list(vocab_set)) + [OOV]
            for word in vocab_list:
                f.write('%s\n' % word)

        # word-level (threshold == 5)
        with open(word_freq5_vocab_file_path, 'w') as f:
            vocab_list = sorted([word for word, freq in list(word_count_dict.items())
                                 if freq >= 5]) + [OOV]
            for word in vocab_list:
                f.write('%s\n' % word)

        # word-level (threshold == 10)
        with open(word_freq10_vocab_file_path, 'w') as f:
            vocab_list = sorted([word for word, freq in list(word_count_dict.items())
                                 if freq >= 10]) + [OOV]
            for word in vocab_list:
                f.write('%s\n' % word)

        # word-level (threshold == 15)
        with open(word_freq15_vocab_file_path, 'w') as f:
            vocab_list = sorted([word for word, freq in list(word_count_dict.items())
                                 if freq >= 15]) + [OOV]
            for word in vocab_list:
                f.write('%s\n' % word)

    # Compute OOV rate
    if 'eval' in data_type:
        with open(mkdir_join(root_path, 'oov', data_size, data_type + '.txt'), 'w') as f:

            # word-level (threshold == 1)
            oov_rate = compute_oov_rate(
                speaker_dict, word_freq1_vocab_file_path)
            f.write('Word (freq1):\n')
            f.write('  OOV rate (test): %f %%\n' % oov_rate)

            # word-level (threshold == 5)
            oov_rate = compute_oov_rate(
                speaker_dict, word_freq5_vocab_file_path)
            f.write('Word (freq5):\n')
            f.write('  OOV rate (test): %f %%\n' % oov_rate)

            # word-level (threshold == 10)
            oov_rate = compute_oov_rate(
                speaker_dict, word_freq10_vocab_file_path)
            f.write('Word (freq10):\n')
            f.write('  OOV rate (test): %f %%\n' % oov_rate)

            # word-level (threshold == 15)
            oov_rate = compute_oov_rate(
                speaker_dict, word_freq15_vocab_file_path)
            f.write('Word (freq15):\n')
            f.write('  OOV rate (test): %f %%\n' % oov_rate)

    # Convert to indices
    print('=====> Convert to index...')
    kanji2idx = Char2idx(kanji_vocab_file_path, double_letter=True)
    kanji2idx_div = Char2idx(kanji_div_vocab_file_path, double_letter=True)
    kana2idx = Char2idx(kana_vocab_file_path, double_letter=True)
    kana2idx_div = Char2idx(kana_div_vocab_file_path, double_letter=True)
    phone2idx = Phone2idx(phone_vocab_file_path)
    phone2idx_div = Phone2idx(phone_div_vocab_file_path)
    word2idx_freq1 = Word2idx(word_freq1_vocab_file_path)
    word2idx_freq5 = Word2idx(word_freq5_vocab_file_path)
    word2idx_freq10 = Word2idx(word_freq10_vocab_file_path)
    word2idx_freq15 = Word2idx(word_freq15_vocab_file_path)
    for speaker, utt_dict in tqdm(speaker_dict.items()):
        for utt_index, utt_info in utt_dict.items():
            start_frame, end_frame, trans_kanji, trans_kana, trans_phone = utt_info
            if 'eval' in data_type:
                utt_dict[utt_index] = [
                    start_frame, end_frame,
                    trans_kanji.replace(SPACE, ''), trans_kanji,
                    trans_kana.replace(SPACE, ''), trans_kana,
                    trans_phone.replace(SIL, '').replace(
                        '  ', ' '), trans_phone,
                    trans_kanji, trans_kanji, trans_kanji, trans_kanji]
            else:
                kanji_indices = kanji2idx(trans_kanji.replace(SPACE, ''))
                kanji_div_indices = kanji2idx_div(trans_kanji)
                kana_indices = kana2idx(trans_kana.replace(SPACE, ''))
                kana_div_indices = kana2idx_div(trans_kana)
                phone_indices = phone2idx(
                    trans_phone.replace(SIL, '').replace('  ', ' '))
                phone_div_indices = phone2idx_div(trans_phone)
                word_freq1_indices = word2idx_freq1(trans_kanji)
                word_freq5_indices = word2idx_freq5(trans_kanji)
                word_freq10_indices = word2idx_freq10(trans_kanji)
                word_freq15_indices = word2idx_freq15(trans_kanji)

                kanji_indices = int2str(kanji_indices)
                kanji_div_indices = int2str(kanji_div_indices)
                kana_indices = int2str(kana_indices)
                kana_div_indices = int2str(kana_div_indices)
                phone_indices = int2str(phone_indices)
                phone_div_indices = int2str(phone_div_indices)
                word_freq1_indices = int2str(word_freq1_indices)
                word_freq5_indices = int2str(word_freq5_indices)
                word_freq10_indices = int2str(word_freq10_indices)
                word_freq15_indices = int2str(word_freq15_indices)


if __name__ == '__main__':
    main()
