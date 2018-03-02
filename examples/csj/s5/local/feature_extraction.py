#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate input features (CSJ corpus)."""

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
from collections import OrderedDict

sys.path.append('../../../')
from utils.directory import mkdir_join
from utils.feature_extraction.segmentation import segment
# from utils.feature_extraction.htk import read, write

parser = argparse.ArgumentParser()
parser.add_argument('--data_save_path', type=str, help='path to save data')

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
    for data_type in ['train', 'dev', 'eval1', 'eval2', 'eval3']:

        print('===> %s' % data_type)
        feature_save_path = mkdir_join(
            args.data_save_path, 'feature', args.tool, data_type)

        utt_indices = []
        with codecs.open(join(args.data_save_path, data_type, 'text'), 'r', 'utf-8') as f:
            for line in f:
                line = line.strip()
                utt_indices.append(line.split('  ')[0])

        segment_dict = {}
        utt_num = 0
        with open(join(args.data_save_path, data_type, 'segments'), 'r') as f:
            for line in f:
                line = line.strip()
                utt_idx, speaker, start_time, end_time = line.split(' ')

                if speaker not in segment_dict.keys():
                    segment_dict[speaker] = OrderedDict()
                segment_dict[speaker][utt_idx] = [int(float(start_time) * 100 + 0.5),
                                                  int(float(end_time) * 100 + 0.5)]
                utt_num += 1
        assert len(utt_indices) == utt_num

        spk2audio = {}
        if args.tool == 'htk':
            with open(join(args.data_save_path, data_type, 'wav2htk.scp'), 'r') as f:
                for line in f:
                    line = line.strip()
                    htk_path = line.split('  ')[1]
                    speaker = basename(htk_path).split('.')[0]
                    spk2audio[speaker] = htk_path
        else:
            with open(join(args.data_save_path, data_type, 'wav.scp'), 'r') as f:
                for line in f:
                    line = line.strip()
                    speaker = line.split(' ')[0]
                    wav_path = line.split(' ')[2]
                    spk2audio[speaker] = wav_path

        if args.tool == 'wav':
            # Split WAV files per utterance
            raise ValueError
            # split_wav(wav_paths=wav_paths,
            #           speaker_dict=speaker_dict_dict[data_type],
            #           save_path=mkdir_join(feature_save_path, data_type))

        else:
            if data_type == 'train':
                global_mean_male, global_std_male = None, None
                global_mean_female, global_std_female = None, None
            else:
                # Load statistics over train dataset
                global_mean_male = np.load(
                    join(args.data_save_path, 'feature', args.tool, 'train/global_mean_male.npy'))
                global_std_male = np.load(
                    join(args.data_save_path, 'feature', args.tool, 'train/global_std_male.npy'))
                global_mean_female = np.load(
                    join(args.data_save_path, 'feature', args.tool, 'train/global_mean_female.npy'))
                global_std_female = np.load(
                    join(args.data_save_path, 'feature', args.tool, 'train/global_std_female.npy'))

            read_audio(data_type=data_type,
                       spk2audio=spk2audio,
                       segment_dict=segment_dict,
                       tool=args.tool,
                       config=CONFIG,
                       normalize=args.normalize,
                       save_path=feature_save_path,
                       global_mean_male=global_mean_male,
                       global_std_male=global_std_male,
                       global_mean_female=global_mean_female,
                       global_std_female=global_std_female)


def read_audio(data_type, spk2audio, segment_dict, tool, config, normalize,
               save_path, global_mean_male=None, global_std_male=None,
               global_mean_female=None, global_std_female=None,
               dtype=np.float32):
    """Read HTK or WAV files.
    Args:
        data_type (string):
        spk2audio (dict):
            key (string) =>
            value () =>
        segment_dict (dict):
            key (string) =>
            value (dict) =>
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
        global_mean_male (np.ndarray, optional): global mean of male over the
            training set
        global_std_male (np.ndarray, optional): global standard deviation of
            male over the training set
        global_mean_female (np.ndarray, optional): global mean of female over
            the training set
        global_std_female (np.ndarray, optional): global standard deviation of
            female over the training set
        dtype (optional): the type of data, default is np.float32
    """
    if data_type != 'train':
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
    speaker_mean_dict = {}

    # NOTE: assume that speakers are different between sessions

    # Loop 1: Computing global mean and statistics
    if data_type == 'train' and normalize != 'no':
        print('=====> Reading audio files...')
        for i, speaker in enumerate(tqdm(segment_dict.keys())):
            audio_path = spk2audio[speaker]

            # Divide each audio file into utterances
            _, feat_utt_sum, speaker_mean, _, total_frame_num_speaker = segment(
                audio_path,
                speaker,
                segment_dict[speaker],  # dict of utterances
                is_training=True,
                sil_duration=0,
                tool=tool,
                config=config)

            if i == 0:
                # Initialize global statistics
                feat_dim = feat_utt_sum.shape[0]
                global_mean_male = np.zeros((feat_dim,), dtype=dtype)
                global_mean_female = np.zeros((feat_dim,), dtype=dtype)
                global_std_male = np.zeros((feat_dim,), dtype=dtype)
                global_std_female = np.zeros((feat_dim,), dtype=dtype)

            # For computing global mean
            if speaker[3] == 'M':
                audio_paths_male.append(audio_path)
                global_mean_male += feat_utt_sum
                total_frame_num_male += total_frame_num_speaker
            elif speaker[3] == 'F':
                audio_paths_female.append(audio_path)
                global_mean_female += feat_utt_sum
                total_frame_num_female += total_frame_num_speaker
            else:
                raise ValueError('gender is M or F.')

            # For computing speaker mean & stddev
            if normalize == 'speaker':
                speaker_mean_dict[speaker] = speaker_mean
                total_frame_num_dict[speaker] = total_frame_num_speaker
                # NOTE: speaker mean is already computed

        print('=====> Computing global mean & stddev...')
        # Compute global mean per gender
        global_mean_male /= total_frame_num_male
        global_mean_female /= total_frame_num_female

        for speaker in tqdm(segment_dict.keys()):
            audio_path = spk2audio[speaker]

            # Divide each audio into utterances
            feat_dict_speaker, _, _, _, _ = segment(
                audio_path,
                speaker,
                segment_dict[speaker],
                is_training=True,
                sil_duration=0,
                tool=tool,
                config=config)

            # For computing global stddev
            if speaker[3] == 'M':
                for feat_utt in feat_dict_speaker.values():
                    global_std_male += np.sum(
                        np.abs(feat_utt - global_mean_male) ** 2, axis=0)
            elif speaker[3] == 'F':
                for feat_utt in feat_dict_speaker.values():
                    global_std_female += np.sum(
                        np.abs(feat_utt - global_mean_female) ** 2, axis=0)
            else:
                raise ValueError('gender is M or F.')

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

    # Loop 2: Normalization and Saving
    print('=====> Normalization...')
    frame_num_dict = {}
    # sampPeriod, parmKind = None, None
    for speaker in tqdm(segment_dict.keys()):
        audio_path = spk2audio[speaker]

        if normalize == 'speaker' and data_type == 'train':
            speaker_mean = speaker_mean_dict[speaker]
        else:
            speaker_mean = None

        # Divide each audio into utterances
        feat_dict_speaker, _, speaker_mean, speaker_std, _ = segment(
            audio_path,
            speaker,
            segment_dict[speaker],
            is_training=data_type == 'train',
            sil_duration=0,
            tool=tool,
            config=config,
            mean=speaker_mean)  # for compute speaker stddev
        # NOTE: feat_dict_speaker have been not normalized yet

        for utt_idx, feat_utt in feat_dict_speaker.items():
            if normalize == 'no':
                pass
            elif normalize == 'global' or not data_type == 'train':
                # Normalize by mean & stddev over the training set per gender
                if speaker[3] == 'M':
                    feat_utt -= global_mean_male
                    feat_utt /= global_std_male
                elif speaker[3] == 'F':
                    feat_utt -= global_mean_female
                    feat_utt /= global_std_female
                else:
                    raise ValueError('gender is M or F.')
            elif normalize == 'speaker':
                # Normalize by mean & stddev per speaker
                feat_utt = (feat_utt - speaker_mean) / speaker_std
            elif normalize == 'utterance':
                # Normalize by mean & stddev per utterance
                utt_mean = np.mean(feat_utt, axis=0, dtype=dtype)
                utt_std = np.std(feat_utt, axis=0, dtype=dtype)
                feat_utt = (feat_utt - utt_mean) / utt_std

            frame_num_dict[utt_idx] = feat_utt.shape[0]

            # Save input features
            np.save(mkdir_join(save_path, speaker,
                               utt_idx + '.npy'), feat_utt)

            # if sampPeriod is None:
            #     _, sampPeriod, parmKind = read(audio_path)
            # write(feat_utt,
            #       htk_path=mkdir_join(save_path, speaker, utt_idx + '.htk'),
            #       sampPeriod=sampPeriod,
            #       parmKind=parmKind)

    # Save the frame number dictionary
    with open(join(save_path, 'frame_num.pickle'), 'wb') as f:
        pickle.dump(frame_num_dict, f)


if __name__ == '__main__':
    main()
