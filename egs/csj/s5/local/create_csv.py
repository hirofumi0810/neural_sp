#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make dataset for the End-to-End model (CSJ corpus).
   Note that feature extraction depends on transcripts.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile, abspath
import sys
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle

sys.path.append('../')
from csj.path import Path
from csj.feature import read_audio
from csj.transcripts.sdb import read_sdb
from utils.util import mkdir_join
from utils.inputs.wav_split import split_wav
from utils.dataset import add_element

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path to CSJ dataset')
parser.add_argument('--dataset_save_path', type=str,
                    help='path to save dataset')
parser.add_argument('--feature_save_path', type=str,
                    help='path to save input features')
parser.add_argument('--wav_save_path', type=str,
                    help='path to save wav files (per utterance)')
parser.add_argument('--tool', type=str,
                    choices=['htk', 'python_speech_features', 'librosa'])
parser.add_argument('--htk_save_path', type=str, help='path to save features')
parser.add_argument('--normalize', type=str,
                    choices=['global', 'speaker', 'utterance', 'no'])
parser.add_argument('--save_format', type=str, choices=['numpy', 'htk', 'wav'])

parser.add_argument('--feature_type', type=str, choices=['fbank', 'mfcc'])
parser.add_argument('--channels', type=int,
                    help='the number of frequency channels')
parser.add_argument('--window', type=float,
                    help='window width to extract features')
parser.add_argument('--slide', type=float, help='extract features per slide')
parser.add_argument('--energy', type=int, help='if 1, add the energy feature')
parser.add_argument('--delta', type=int, help='if 1, add the energy feature')
parser.add_argument('--deltadelta', type=int,
                    help='if 1, double delta features are also extracted')
parser.add_argument('--subset', type=int,
                    help='If True, create small dataset.')
parser.add_argument('--fullset', type=int,
                    help='If True, create full-size dataset.')

args = parser.parse_args()
path = Path(data_path=args.data_path,
            config_path='./config',
            htk_save_path=args.htk_save_path)

CONFIG = {
    'feature_type': args.feature_type,
    'channels': args.channels,
    'sampling_rate': 16000,
    'window': args.window,
    'slide': args.slide,
    'energy': bool(args.energy),
    'delta': bool(args.delta),
    'deltadelta': bool(args.deltadelta)
}

if args.save_format == 'htk':
    assert args.tool == 'htk'


def main(data_size):

    speaker_dict_dict = {}  # dict of speaker_dict
    for data_type in ['train', 'eval1', 'eval2', 'eval3']:
        print('=' * 50)
        print(' ' * 20 + data_type + ' (' + data_size + ')' + ' ' * 20)
        print('=' * 50)

        ########################################
        # labels
        ########################################
        if data_type == 'train':
            label_paths = path.trans(data_type='train_' + data_size)
        else:
            label_paths = path.trans(data_type=data_type)
        save_vocab_file = True if data_type == 'train' else False

        print('=> Processing transcripts...')
        speaker_dict_dict[data_type] = read_sdb(
            label_paths=label_paths,
            data_size=data_size,
            data_type=data_type,
            root_path=abspath('./'),
            save_vocab_file=save_vocab_file)

        ########################################
        # inputs
        ########################################
        print('\n=> Processing input data...')
        input_save_path = mkdir_join(
            args.feature_save_path, args.save_format, data_size)
        if isfile(join(input_save_path, data_type, 'complete.txt')):
            print('Already exists.')
        else:
            if args.save_format == 'wav':
                ########################################
                # Split WAV files per utterance
                ########################################
                if data_type == 'train':
                    wav_paths = path.wav(data_type='train_' + data_size)
                else:
                    wav_paths = path.wav(data_type=data_type)
                split_wav(wav_paths=wav_paths,
                          speaker_dict=speaker_dict_dict[data_type],
                          save_path=mkdir_join(input_save_path, data_type))
                # NOTE: ex.) save_path:
                # csj/feature/save_format/data_size/data_type/speaker/utt_name.npy

            elif args.save_format in ['numpy', 'htk']:
                if data_type == 'train':
                    if args.tool == 'htk':
                        audio_paths = path.htk(data_type='train_' + data_size)
                    else:
                        audio_paths = path.wav(data_type='train_' + data_size)
                    is_training = True
                    global_mean_male, global_std_male, global_mean_female, global_std_female = None, None, None, None
                else:
                    if args.tool == 'htk':
                        audio_paths = path.htk(data_type=data_type)
                    else:
                        audio_paths = path.wav(data_type=data_type)
                    is_training = False

                    # Load statistics over train dataset
                    global_mean_male = np.load(
                        join(input_save_path, 'train/global_mean_male.npy'))
                    global_std_male = np.load(
                        join(input_save_path, 'train/global_std_male.npy'))
                    global_mean_female = np.load(
                        join(input_save_path, 'train/global_mean_female.npy'))
                    global_std_female = np.load(
                        join(input_save_path, 'train/global_std_female.npy'))
                print(speaker_dict_dict[data_type].keys())
                read_audio(audio_paths=audio_paths,
                           speaker_dict=speaker_dict_dict[data_type],
                           tool=args.tool,
                           config=CONFIG,
                           normalize=args.normalize,
                           is_training=is_training,
                           save_path=mkdir_join(input_save_path, data_type),
                           save_format=args.save_format,
                           global_mean_male=global_mean_male,
                           global_std_male=global_std_male,
                           global_mean_female=global_mean_female,
                           global_std_female=global_std_female)
                # NOTE: ex.) save_path:
                # csj/feature/save_format/data_size/data_type/speaker/*.npy

            # Make a confirmation file to prove that dataset was saved
            # correctly
            with open(join(input_save_path, data_type, 'complete.txt'), 'w') as f:
                f.write('')

        ########################################
        # dataset (csv)
        ########################################
        print('\n=> Saving dataset files...')
        dataset_save_path = mkdir_join(
            args.dataset_save_path, args.save_format, data_size, data_type)

        df_columns = ['frame_num', 'input_path', 'transcript']
        df_kanji = pd.DataFrame([], columns=df_columns)
        df_kanji_divide = pd.DataFrame([], columns=df_columns)
        df_kana = pd.DataFrame([], columns=df_columns)
        df_kana_divide = pd.DataFrame([], columns=df_columns)
        df_phone = pd.DataFrame([], columns=df_columns)
        df_phone_divide = pd.DataFrame([], columns=df_columns)
        df_word_freq1 = pd.DataFrame([], columns=df_columns)
        df_word_freq5 = pd.DataFrame([], columns=df_columns)
        df_word_freq10 = pd.DataFrame([], columns=df_columns)
        df_word_freq15 = pd.DataFrame([], columns=df_columns)

        with open(join(input_save_path, data_type, 'frame_num.pickle'), 'rb') as f:
            frame_num_dict = pickle.load(f)

        utt_count = 0
        df_kanji_list, df_kanji_divide_list = [], []
        df_kana_list,  df_kana_divide_list = [], []
        df_phone_list, df_phone_divide_list = [], []
        df_word_freq1_list, df_word_freq5_list = [], []
        df_word_freq10_list, df_word_freq15_list = [], []
        speaker_dict = speaker_dict_dict[data_type]
        for speaker, utt_dict in tqdm(speaker_dict.items()):
            for utt_index, utt_info in utt_dict.items():
                kanji_indices, kanji_divide_indices = utt_info[2:4]
                kana_indices, kana_divide_indices = utt_info[4:6]
                phone_indices, phone_divide_indices = utt_info[6:8]
                word_freq1_indices, word_freq5_indices = utt_info[8:10]
                word_freq10_indices, word_freq15_indices = utt_info[10:12]

                if args.save_format == 'numpy':
                    input_utt_save_path = join(
                        input_save_path, data_type, speaker, speaker + '_' + utt_index + '.npy')
                elif args.save_format == 'htk':
                    input_utt_save_path = join(
                        input_save_path, data_type, speaker, speaker + '_' + utt_index + '.htk')
                elif args.save_format == 'wav':
                    input_utt_save_path = path.utt2wav(utt_index)
                else:
                    raise ValueError('save_format is numpy or htk or wav.')

                assert isfile(input_utt_save_path)
                frame_num = frame_num_dict[speaker + '_' + utt_index]

                df_kanji = add_element(
                    df_kanji, [frame_num, input_utt_save_path, kanji_indices])
                df_kanji_divide = add_element(
                    df_kanji_divide, [frame_num, input_utt_save_path, kanji_divide_indices])
                df_kana = add_element(
                    df_kana, [frame_num, input_utt_save_path, kana_indices])
                df_kana_divide = add_element(
                    df_kana_divide, [frame_num, input_utt_save_path, kana_divide_indices])
                df_phone = add_element(
                    df_phone, [frame_num, input_utt_save_path, phone_indices])
                df_phone_divide = add_element(
                    df_phone_divide, [frame_num, input_utt_save_path, phone_divide_indices])
                df_word_freq1 = add_element(
                    df_word_freq1, [frame_num, input_utt_save_path, word_freq1_indices])
                df_word_freq5 = add_element(
                    df_word_freq5, [frame_num, input_utt_save_path, word_freq5_indices])
                df_word_freq10 = add_element(
                    df_word_freq10, [frame_num, input_utt_save_path, word_freq10_indices])
                df_word_freq15 = add_element(
                    df_word_freq15, [frame_num, input_utt_save_path, word_freq15_indices])
                utt_count += 1

                # Reset
                if utt_count == 10000:
                    df_kanji_list.append(df_kanji)
                    df_kanji_divide_list.append(df_kanji_divide)
                    df_kana_list.append(df_kana)
                    df_kana_divide_list.append(df_kana_divide)
                    df_phone_list.append(df_phone)
                    df_phone_divide_list.append(df_phone_divide)
                    df_word_freq1_list.append(df_word_freq1)
                    df_word_freq5_list.append(df_word_freq5)
                    df_word_freq10_list.append(df_word_freq10)
                    df_word_freq15_list.append(df_word_freq15)

                    df_kanji = pd.DataFrame([], columns=df_columns)
                    df_kanji_divide = pd.DataFrame([], columns=df_columns)
                    df_kana = pd.DataFrame([], columns=df_columns)
                    df_kana_divide = pd.DataFrame([], columns=df_columns)
                    df_phone = pd.DataFrame([], columns=df_columns)
                    df_phone_divide = pd.DataFrame([], columns=df_columns)
                    df_word_freq1 = pd.DataFrame([], columns=df_columns)
                    df_word_freq5 = pd.DataFrame([], columns=df_columns)
                    df_word_freq10 = pd.DataFrame([], columns=df_columns)
                    df_word_freq15 = pd.DataFrame([], columns=df_columns)
                    utt_count = 0

        # Last dataframe
        df_kanji_list.append(df_kanji)
        df_kanji_divide_list.append(df_kanji_divide)
        df_kana_list.append(df_kana)
        df_kana_divide_list.append(df_kana_divide)
        df_phone_list.append(df_phone)
        df_phone_divide_list.append(df_phone_divide)
        df_word_freq1_list.append(df_word_freq1)
        df_word_freq5_list.append(df_word_freq5)
        df_word_freq10_list.append(df_word_freq10)
        df_word_freq15_list.append(df_word_freq15)

        # Concatenate all dataframes
        df_kanji = df_kanji_list[0]
        df_kanji_divide = df_kanji_divide_list[0]
        df_kana = df_kana_list[0]
        df_kana_divide = df_kana_divide_list[0]
        df_phone = df_phone_list[0]
        df_phone_divide = df_phone_divide_list[0]
        df_word_freq1 = df_word_freq1_list[0]
        df_word_freq5 = df_word_freq5_list[0]
        df_word_freq10 = df_word_freq10_list[0]
        df_word_freq15 = df_word_freq15_list[0]

        for df_i in df_kanji_list[1:]:
            df_kanji = pd.concat([df_kanji, df_i], axis=0)
        for df_i in df_kanji_divide_list[1:]:
            df_kanji_divide = pd.concat([df_kanji_divide, df_i], axis=0)
        for df_i in df_kana_list[1:]:
            df_kana = pd.concat([df_kana, df_i], axis=0)
        for df_i in df_kana_divide_list[1:]:
            df_kana_divide = pd.concat([df_kana_divide, df_i], axis=0)
        for df_i in df_phone_list[1:]:
            df_phone = pd.concat([df_phone, df_i], axis=0)
        for df_i in df_phone_divide_list[1:]:
            df_phone_divide = pd.concat([df_phone_divide, df_i], axis=0)
        for df_i in df_word_freq1_list[1:]:
            df_word_freq1 = pd.concat([df_word_freq1, df_i], axis=0)
        for df_i in df_word_freq5_list[1:]:
            df_word_freq5 = pd.concat([df_word_freq5, df_i], axis=0)
        for df_i in df_word_freq10_list[1:]:
            df_word_freq10 = pd.concat([df_word_freq10, df_i], axis=0)
        for df_i in df_word_freq15_list[1:]:
            df_word_freq15 = pd.concat([df_word_freq15, df_i], axis=0)

        df_kanji.to_csv(join(dataset_save_path, 'kanji.csv'))
        df_kanji_divide.to_csv(join(dataset_save_path, 'kanji_divide.csv'))
        df_kana.to_csv(join(dataset_save_path, 'kana.csv'))
        df_kana_divide.to_csv(join(dataset_save_path, 'kana_divide.csv'))
        df_phone.to_csv(join(dataset_save_path, 'phone.csv'))
        df_phone_divide.to_csv(join(dataset_save_path, 'phone_divide.csv'))
        df_word_freq1.to_csv(join(dataset_save_path, 'word_freq1.csv'))
        df_word_freq5.to_csv(join(dataset_save_path, 'word_freq5.csv'))
        df_word_freq10.to_csv(join(dataset_save_path, 'word_freq10.csv'))
        df_word_freq15.to_csv(join(dataset_save_path, 'word_freq15.csv'))


if __name__ == '__main__':

    # data_sizes = ['aps', 'sps']
    data_sizes = []
    if bool(args.subset):
        data_sizes += ['subset']
    if bool(args.fullset):
        data_sizes += ['fullset']

    for data_size in data_sizes:
        main(data_size)
