#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate input features (LibriSpeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, basename
import sys
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import codecs

sys.path.append('../../../')
from utils.directory import mkdir_join
from utils.feature_extraction.htk import read
from utils.feature_extraction.wav2feature_python_speech_features import wav2feature as w2f_psf
from utils.feature_extraction.wav2feature_librosa import wav2feature as w2f_librosa

parser = argparse.ArgumentParser()
parser.add_argument('--data_save_path', type=str, help='path to save data')
parser.add_argument('--data_size', type=str,
                    choices=['100', '460', '960'])

parser.add_argument('--tool', type=str,
                    choices=['htk', 'python_speech_features', 'librosa', 'wav'])
parser.add_argument('--normalize', type=str,
                    choices=['global', 'speaker', 'utterance', 'no'])

parser.add_argument('--channels', type=int,
                    help='the number of frequency channels')
parser.add_argument('--window', type=float, default=0.025,
                    help='window width to extract features')
parser.add_argument('--slide', type=float, default=0.01,
                    help='extract features per slide')
parser.add_argument('--energy', type=int, help='if 1, add the energy feature')
parser.add_argument('--delta', type=int, help='if 1, add the energy feature')
parser.add_argument('--deltadelta', type=int,
                    help='if 1, double delta features are also extracted')

args = parser.parse_args()


CONFIG = {
    'feature_type': 'fbank',
    'channels': args.channels,
    'sampling_rate': 16000,
    'window': args.window,
    'slide': args.slide,
    'energy': bool(args.energy),
    'delta': bool(args.delta),
    'deltadelta': bool(args.deltadelta)
}


def main():
    print('=> Processing input data...')
    for data_type in ['train_' + args.data_size, 'dev_clean', 'dev_other', 'test_clean', 'test_other']:

        print('===> %s' % data_type)
        feature_save_path = mkdir_join(
            args.data_save_path, 'feature', args.tool, args.data_size, data_type)

        utt_indices = []
        with codecs.open(join(args.data_save_path, data_type, 'text'), 'r', 'utf-8') as f:
            for line in f:
                line = line.strip()
                utt_indices.append(line.split('  ')[0])

        audio_paths = []
        if args.tool == 'htk':
            with open(join(args.data_save_path, data_type, 'htk.scp'), 'r') as f:
                for line in f:
                    htk_path = line.strip()
                    audio_paths.append(htk_path)
        else:
            with open(join(args.data_save_path, data_type, 'wav.scp'), 'r') as f:
                for line in f:
                    line = line.strip()
                    wav_path = line.split(' ')[4]
                    audio_paths.append(wav_path)

        spk2gender = {}
        with open(join(args.data_save_path, data_type, 'spk2gender'), 'r') as f:
            for line in f:
                line = line.strip()
                speaker, gender = line.split(' ')
                spk2gender[speaker] = gender

        if 'train' in data_type:
            global_mean_male, global_std_male = None, None
            global_mean_female, global_std_female = None, None
        else:
            # Load statistics over train dataset
            global_mean_male = np.load(
                join(args.data_save_path, 'feature', args.tool, args.data_size, 'train_' + args.data_size, 'global_mean_male.npy'))
            global_std_male = np.load(
                join(args.data_save_path, 'feature', args.tool, args.data_size, 'train_' + args.data_size, 'global_std_male.npy'))
            global_mean_female = np.load(
                join(args.data_save_path, 'feature', args.tool, args.data_size, 'train_' + args.data_size, 'global_mean_female.npy'))
            global_std_female = np.load(
                join(args.data_save_path, 'feature', args.tool, args.data_size, 'train_' + args.data_size, 'global_std_female.npy'))

        read_audio(data_type=data_type,
                   audio_paths=audio_paths,
                   spk2gender=spk2gender,
                   tool=args.tool,
                   config=CONFIG,
                   normalize=args.normalize,
                   save_path=feature_save_path,
                   global_mean_male=global_mean_male,
                   global_std_male=global_std_male,
                   global_mean_female=global_mean_female,
                   global_std_female=global_std_female)


def read_audio(data_type, audio_paths, spk2gender, tool, config, normalize,
               save_path, global_mean_male=None, global_std_male=None,
               global_mean_female=None, global_std_female=None,
               dtype=np.float32):
    """Read HTK or WAV files.
    Args:
        data_type (string): train_si84 or train_si284 or test_dev93 or test_eval92
        audio_paths (list): paths to audio files
        spk2gender (dict):
            key => speaker
            value => gender
        tool (string): the tool to extract features,
            htk or librosa or python_speech_features
        config (dict): a configuration for feature extraction
        normalize (string):
            no => normalization will be not conducted
            global => normalize input features by global mean & stddev over
                      the training set per gender
            speaker => normalize input features by mean & stddev per speaker
            utterance => normalize input features by mean & stddev per utterancet
                         data by mean & stddev per utterance
        save_path (string): path to save npy files
        global_mean_male (np.ndarray): global mean of male over the training set
        global_std_male (np.ndarray): global standard deviation of male over the training set
        global_mean_female (np.ndarray): global mean of female over the training set
        global_std_female (np.ndarray): global standard deviation of female over the training set
        dtype): the type of data, default is np.float32
    """
    is_training = 'train' in data_type

    if not is_training:
        if global_mean_male is None or global_mean_female is None:
            raise ValueError('Set mean & stddev computed in the training set.')
    if normalize not in ['global', 'speaker', 'utterance', 'no']:
        raise ValueError(
            'normalize must be "utterance" or "speaker" or "global" or "no".')
    if tool not in ['htk', 'python_speech_features', 'librosa']:
        raise TypeError(
            'tool must be "htk" or "python_speech_features" or "librosa".')

    audio_paths_male, audio_paths_female = [], []
    total_frame_num_male, total_frame_num_female = 0, 0
    total_frame_num_dict = {}
    speaker_mean_dict, speaker_std_dict = {}, {}

    # Loop 1: Computing global mean and statistics
    if is_training and normalize != 'no':
        print('=====> Reading audio files...')
        for i, audio_path in enumerate(tqdm(audio_paths)):
            speaker, chapter = audio_path.split('/')[-3:-1]
            utt_idx = basename(audio_path).split('.')[0]
            gender = spk2gender[speaker + '-' + chapter]

            if tool == 'htk':
                feat_utt, sampPeriod, parmKind = read(audio_path)
            elif tool == 'python_speech_features':
                feat_utt = w2f_psf(audio_path,
                                   feature_type=config['feature_type'],
                                   feature_dim=config['channels'],
                                   use_energy=config['energy'],
                                   use_delta1=config['delta'],
                                   use_delta2=config['deltadelta'],
                                   window=config['window'],
                                   slide=config['slide'])
            elif tool == 'librosa':
                feat_utt = w2f_librosa(audio_path,
                                       feature_type=config['feature_type'],
                                       feature_dim=config['channels'],
                                       use_energy=config['energy'],
                                       use_delta1=config['delta'],
                                       use_delta2=config['deltadelta'],
                                       window=config['window'],
                                       slide=config['slide'])

            frame_num, feat_dim = feat_utt.shape
            feat_utt_sum = np.sum(feat_utt, axis=0)

            if i == 0:
                # Initialize global statistics
                global_mean_male = np.zeros((feat_dim,), dtype=dtype)
                global_mean_female = np.zeros((feat_dim,), dtype=dtype)
                global_std_male = np.zeros((feat_dim,), dtype=dtype)
                global_std_female = np.zeros((feat_dim,), dtype=dtype)

            # For computing global mean
            if gender == 'm':
                audio_paths_male.append(audio_path)
                global_mean_male += feat_utt_sum
                total_frame_num_male += frame_num
            elif gender == 'f':
                audio_paths_female.append(audio_path)
                global_mean_female += feat_utt_sum
                total_frame_num_female += frame_num
            else:
                raise ValueError('gender is m or f.')

            # For computing speaker mean & stddev
            if normalize == 'speaker':
                # Initialize speaker statistics
                if speaker not in total_frame_num_dict.keys():
                    total_frame_num_dict[speaker] = 0
                    speaker_mean_dict[speaker] = np.zeros(
                        (feat_dim,), dtype=dtype)
                    speaker_std_dict[speaker] = np.zeros(
                        (feat_dim,), dtype=dtype)
                total_frame_num_dict[speaker] += frame_num
                speaker_mean_dict[speaker] += feat_utt_sum

        print('=====> Computing global mean & stddev...')
        # Compute global mean per gender
        global_mean_male /= total_frame_num_male
        global_mean_female /= total_frame_num_female

        # Compute speaker mean
        if normalize == 'speaker':
            for speaker in speaker_mean_dict.keys():
                speaker_mean_dict[speaker] /= total_frame_num_dict[speaker]

        for audio_path in tqdm(audio_paths):
            speaker, chapter = audio_path.split('/')[-3:-1]
            utt_idx = basename(audio_path).split('.')[0]
            gender = spk2gender[speaker + '-' + chapter]

            if tool == 'htk':
                feat_utt, sampPeriod, parmKind = read(audio_path)
            elif tool == 'python_speech_features':
                feat_utt = w2f_psf(audio_path,
                                   feature_type=config['feature_type'],
                                   feature_dim=config['channels'],
                                   use_energy=config['energy'],
                                   use_delta1=config['delta'],
                                   use_delta2=config['deltadelta'],
                                   window=config['window'],
                                   slide=config['slide'])
            elif tool == 'librosa':
                feat_utt = w2f_librosa(audio_path,
                                       feature_type=config['feature_type'],
                                       feature_dim=config['channels'],
                                       use_energy=config['energy'],
                                       use_delta1=config['delta'],
                                       use_delta2=config['deltadelta'],
                                       window=config['window'],
                                       slide=config['slide'])

            # For computing global stddev
            if gender == 'm':
                global_std_male += np.sum(
                    np.abs(feat_utt - global_mean_male) ** 2, axis=0)
            elif gender == 'f':
                global_std_female += np.sum(
                    np.abs(feat_utt - global_mean_female) ** 2, axis=0)
            else:
                raise ValueError('gender is m or f.')

            # For computing speaker stddev
            if normalize == 'speaker':
                speaker_std_dict[speaker] += np.sum(
                    np.abs(feat_utt - speaker_mean_dict[speaker]) ** 2, axis=0)

        # Compute speaker stddev
        if normalize == 'speaker':
            for speaker in speaker_std_dict.keys():
                speaker_std_dict[speaker] = np.sqrt(
                    speaker_std_dict[speaker] / (total_frame_num_dict[speaker] - 1))

        # Compute global stddev per gender
        global_std_male = np.sqrt(
            global_std_male / (total_frame_num_male - 1))
        global_std_female = np.sqrt(
            global_std_female / (total_frame_num_female - 1))

        # Save global mean & stddev per gender
        np.save(join(save_path, 'global_mean_male.npy'), global_mean_male)
        np.save(join(save_path, 'global_mean_female.npy'),
                global_mean_female)
        np.save(join(save_path, 'global_std_male.npy'), global_std_male)
        np.save(join(save_path, 'global_std_female.npy'), global_std_female)

    # Loop 2: Normalization and saving
    print('=====> Normalization...')
    frame_num_dict = {}
    # sampPeriod, parmKind = None, None
    for audio_path in tqdm(audio_paths):
        speaker, chapter = audio_path.split('/')[-3:-1]
        utt_idx = basename(audio_path).split('.')[0]
        gender = spk2gender[speaker + '-' + chapter]

        if tool == 'htk':
            feat_utt, sampPeriod, parmKind = read(audio_path)
        elif tool == 'python_speech_features':
            feat_utt = w2f_psf(audio_path,
                               feature_type=config['feature_type'],
                               feature_dim=config['channels'],
                               use_energy=config['energy'],
                               use_delta1=config['delta'],
                               use_delta2=config['deltadelta'],
                               window=config['window'],
                               slide=config['slide'])
        elif tool == 'librosa':
            feat_utt = w2f_librosa(audio_path,
                                   feature_type=config['feature_type'],
                                   feature_dim=config['channels'],
                                   use_energy=config['energy'],
                                   use_delta1=config['delta'],
                                   use_delta2=config['deltadelta'],
                                   window=config['window'],
                                   slide=config['slide'])

        if normalize == 'no':
            pass
        elif normalize == 'global' or not is_training:
            # Normalize by mean & stddev over the training set per gender
            if gender == 'm':
                feat_utt -= global_mean_male
                feat_utt /= global_std_male
            elif gender == 'f':
                feat_utt -= global_mean_female
                feat_utt /= global_std_female
            else:
                raise ValueError('gender is m or f.')
        elif normalize == 'speaker':
            # Normalize by mean & stddev per speaker
            feat_utt = (
                feat_utt - speaker_mean_dict[speaker]) / speaker_std_dict[speaker]
        elif normalize == 'utterance':
            # Normalize by mean & stddev per utterance
            utt_mean = np.mean(feat_utt, axis=0, dtype=dtype)
            utt_std = np.std(feat_utt, axis=0, dtype=dtype)
            feat_utt = (feat_utt - utt_mean) / utt_std

        frame_num_dict[utt_idx] = feat_utt.shape[0]

        # Save input features
        np.save(mkdir_join(save_path, speaker,
                           chapter, utt_idx + '.npy'), feat_utt)

    # Save the frame number dictionary
    with open(join(save_path, 'frame_num.pickle'), 'wb') as f:
        pickle.dump(frame_num_dict, f)


if __name__ == '__main__':
    main()
