#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make dataset CSV files (TIMIT corpus)."""

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

sys.path.append('../../../')
from utils.io.labels.phone import Phone2idx
from utils.directory import mkdir_join

parser = argparse.ArgumentParser()
parser.add_argument('--data_save_path', type=str,
                    help='path to save data')
parser.add_argument('--phone_map_file_path', type=str,
                    help='path to phones.60-48-39.map')
parser.add_argument('--tool', type=str,
                    choices=['wav', 'htk', 'python_speech_features', 'librosa'])

args = parser.parse_args()


def main():

    for data_type in ['train', 'dev', 'test']:
        print('=' * 50)
        print(' ' * 20 + data_type)
        print('=' * 50)

        # Convert transcript to index
        print('=> Processing transcripts...')
        trans_dict = read_text(
            text_path=join(args.data_save_path, data_type, 'text'),
            vocab_save_path=mkdir_join(args.data_save_path, 'vocab'),
            data_type=data_type,
            phone_map_file_path=args.phone_map_file_path)

        # Make dataset file (.csv)
        print('=> Saving dataset files...')
        csv_save_path = mkdir_join(
            args.data_save_path, 'dataset', args.tool, data_type)

        df_columns = ['frame_num', 'input_path', 'transcript']
        df_phone61 = pd.DataFrame([], columns=df_columns)
        df_phone48 = pd.DataFrame([], columns=df_columns)
        df_phone39 = pd.DataFrame([], columns=df_columns)

        with open(join(args.data_save_path, 'feature', args.tool,
                       data_type, 'frame_num.pickle'), 'rb') as f:
            frame_num_dict = pickle.load(f)

        for utt_idx, trans_list in tqdm(trans_dict.items()):
            feat_utt_save_path = join(
                args.data_save_path, 'feature', args.tool, data_type, utt_idx + '.npy')
            frame_num = frame_num_dict[utt_idx]

            if not isfile(feat_utt_save_path):
                raise ValueError('There is no file: %s' % feat_utt_save_path)

            phone61_indices, phone48_indices, phone39_indices = trans_list

            df_phone61 = add_element(
                df_phone61, [frame_num, feat_utt_save_path, phone61_indices])
            df_phone48 = add_element(
                df_phone48, [frame_num, feat_utt_save_path, phone48_indices])
            df_phone39 = add_element(
                df_phone39, [frame_num, feat_utt_save_path, phone39_indices])

        df_phone61.to_csv(join(csv_save_path, 'phone61.csv'))
        df_phone48.to_csv(join(csv_save_path, 'phone48.csv'))
        df_phone39.to_csv(join(csv_save_path, 'phone39.csv'))


def add_element(df, elem_list):
    series = pd.Series(elem_list, index=df.columns)
    df = df.append(series, ignore_index=True)
    return df


def read_text(text_path, vocab_save_path, data_type, phone_map_file_path):
    """Read phone transcript.
    Args:
        text_path (string): path to a text file of kaldi
        vocab_save_path (string): path to save vocabulary files
        data_type (string): train or dev or test
        phone_map_file_path (string):
    Returns:
        text_dict (dict):
            key (string) => utterance index
            value (list) => list of
                [phone61_indices, phone48_indices, phone39_indices]
    """
    print('=====> Reading target labels...')

    # Make the phone mapping file (from phone to index)
    phone61_set, phone48_set, phone39_set = set([]), set([]), set([])
    to_phone48, to_phone39, = {}, {}
    with open(phone_map_file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            if len(line) >= 2:
                phone61_set.add(line[0])
                phone48_set.add(line[1])
                phone39_set.add(line[2])
                to_phone48[line[0]] = line[1]
                to_phone39[line[0]] = line[2]
            else:
                # Ignore "q" if phone39 or phone48
                phone61_set.add(line[0])
                to_phone48[line[0]] = ''
                to_phone39[line[0]] = ''

    # Make vocabulary files
    phone61_vocab_path = mkdir_join(vocab_save_path, 'phone61.txt')
    phone48_vocab_path = mkdir_join(vocab_save_path, 'phone48.txt')
    phone39_vocab_path = mkdir_join(vocab_save_path, 'phone39.txt')

    # Save vocabulary files
    if data_type == 'train':
        with open(phone61_vocab_path, 'w') as f:
            for phone in sorted(list(phone61_set)):
                f.write('%s\n' % phone)
        with open(phone48_vocab_path, 'w') as f:
            for phone in sorted(list(phone48_set)):
                f.write('%s\n' % phone)
        with open(phone39_vocab_path, 'w') as f:
            for phone in sorted(list(phone39_set)):
                f.write('%s\n' % phone)

    trans_dict = {}
    with open(text_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Normalize space
            line = re.sub(r'[\s]+', ' ', line)

            utt_idx = line.split(' ')[0]
            phone61_list = line.split(' ')[1:]

            # Map from 61 phones to the corresponding phones
            phone48_list = list(map(lambda x: to_phone48[x], phone61_list))
            phone39_list = list(map(lambda x: to_phone39[x], phone61_list))

            # ignore "q" in phone48 and phone39
            while '' in phone48_list:
                phone48_list.remove('')
            while '' in phone39_list:
                phone39_list.remove('')

            # Convert to string
            trans_phone61 = ' '.join(phone61_list)
            trans_phone48 = ' '.join(phone48_list)
            trans_phone39 = ' '.join(phone39_list)

            # for debug
            # print(trans_phone61)
            # print(trans_phone48)
            # print(trans_phone39)
            # print('=' * 20)

            trans_dict[utt_idx] = [trans_phone61, trans_phone48, trans_phone39]

    # Convert to index
    print('=====> Convert to index...')
    phone2idx_61 = Phone2idx(phone61_vocab_path)
    phone2idx_48 = Phone2idx(phone48_vocab_path)
    phone2idx_39 = Phone2idx(phone39_vocab_path)
    for utt_idx, [trans_phone61, trans_phone48, trans_phone39] in tqdm(trans_dict.items()):
        if data_type == 'test':
            pass
            # trans_dict[utt_idx] = [trans_phone61, trans_phone48, trans_phone39]
            # NOTE: save as it is
        else:
            phone61_indices = phone2idx_61(trans_phone61)
            phone48_indices = phone2idx_48(trans_phone48)
            phone39_indices = phone2idx_39(trans_phone39)

            phone61_indices = ' '.join(list(map(str, phone61_indices)))
            phone48_indices = ' '.join(list(map(str, phone48_indices)))
            phone39_indices = ' '.join(list(map(str, phone39_indices)))

            trans_dict[utt_idx] = [phone61_indices,
                                   phone48_indices,
                                   phone39_indices]
    return trans_dict


if __name__ == '__main__':

    main()
