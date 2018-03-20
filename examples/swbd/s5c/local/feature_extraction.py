#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make input features (Switchboard corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, basename
import sys
import numpy as np
import pickle
from tqdm import tqdm
import argparse
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
    'sampling_rate': 8000,
    'window': args.window,
    'slide': args.slide,
    'energy': bool(args.energy),
    'delta': bool(args.delta),
    'deltadelta': bool(args.deltadelta)
}


def main():
    print('=> Processing input data...')
    for data_type in ['train', 'dev', 'eval2000_swbd', 'eval2000_ch']:

        if 'eval' in data_type:
            data_type_tmp = 'eval2000'
        else:
            data_type_tmp = data_type

        print('===> %s' % data_type)
        feature_save_path = mkdir_join(
            args.data_save_path, 'feature', args.tool, data_type)

        utt_indices = []
        with open(join(args.data_save_path, data_type_tmp, 'text'), 'r') as f:
            for line in f:
                line = line.strip()

                utt_idx = line.split('  ')[0]

                if data_type == 'eval2000_swbd' and utt_idx[:2] == 'en':
                    continue
                if data_type == 'eval2000_ch' and utt_idx[:2] == 'sw':
                    continue

                utt_indices.append(utt_idx)

        segment_dict = {}
        utt_num = 0
        with open(join(args.data_save_path, data_type_tmp, 'segments'), 'r') as f:
            for line in f:
                line = line.strip()
                utt_idx, speaker, start_time, end_time = line.split(' ')

                if data_type == 'eval2000_swbd' and utt_idx[:2] == 'en':
                    continue
                if data_type == 'eval2000_ch' and utt_idx[:2] == 'sw':
                    continue

                if speaker not in segment_dict.keys():
                    segment_dict[speaker] = OrderedDict()
                segment_dict[speaker][utt_idx] = [int(float(start_time) * 100 + 0.5),
                                                  int(float(end_time) * 100 + 0.5)]
                utt_num += 1
        assert len(utt_indices) == utt_num

        spk2audio = {}
        if args.tool == 'htk':
            with open(join(args.data_save_path, data_type_tmp, 'wav2htk.scp'), 'r') as f:
                for line in f:
                    line = line.strip()
                    htk_path = line.split('  ')[1]
                    speaker = basename(htk_path).split('.')[0]

                    if data_type == 'eval2000_swbd' and speaker[:2] == 'en':
                        continue
                    if data_type == 'eval2000_ch' and speaker[:2] == 'sw':
                        continue

                    spk2audio[speaker] = htk_path
        else:
            with open(join(args.data_save_path, data_type_tmp, 'wav.scp'), 'r') as f:
                for line in f:
                    line = line.strip()
                    speaker = line.split(' ')[0]
                    if data_type == 'dev':
                        wav_path = join(args.data_save_path,
                                        'wav_1ch', 'train', speaker + '.wav')
                    else:
                        wav_path = join(args.data_save_path,
                                        'wav_1ch', data_type, speaker + '.wav')
                    spk2audio[speaker] = wav_path

        if args.tool == 'wav':
            # Split WAV files per utterance
            raise ValueError
            # split_wav(wav_paths=wav_paths,
            #           speaker_dict=speaker_dict_dict[data_type],
            #           save_path=mkdir_join(feature_save_path, data_type))

        else:
            if data_type == 'train':
                global_mean, global_std = None, None
            else:
                # Load statistics over train dataset
                global_mean = np.load(
                    join(args.data_save_path, 'feature', args.tool, 'train/global_mean.npy'))
                global_std = np.load(
                    join(args.data_save_path, 'feature', args.tool, 'train/global_std.npy'))

            read_audio(data_type=data_type,
                       spk2audio=spk2audio,
                       segment_dict=segment_dict,
                       tool=args.tool,
                       config=CONFIG,
                       normalize=args.normalize,
                       save_path=feature_save_path,
                       global_mean=global_mean,
                       global_std=global_std)


def read_audio(data_type, spk2audio, segment_dict, tool, config, normalize,
               save_path, global_mean=None, global_std=None, dtype=np.float32):
    """Read HTK or WAV files.
    Args:
        data_type (string):
        spk2audio (dict):
        segment_dict (dict):
            key (string) =>
            value (dict) =>
        tool (string): the tool to extract features,
            htk or librosa or python_speech_features
        config (dict): a configuration for feature extraction
        normalize (string):
            no => normalization will be not conducted
            global => normalize input features by global mean & stddev over
                      the training set
            speaker => normalize input features by mean & stddev per speaker
            utterance => normalize input features by mean & stddev per utterancet
                         data by mean & stddev per utterance
        save_path (string): path to save npy files
        global_mean (np.ndarray, optional): global mean over the training set
        global_std (np.ndarray, optional): global standard deviation over
            the training set
        dtype (optional): the type of data, default is np.float32
    """
    if data_type != 'train':
        if global_mean is None or global_std is None:
            raise ValueError('Set mean & stddev computed in the training set.')
    if normalize not in ['global', 'speaker', 'utterance', 'no']:
        raise ValueError(
            'normalize must be "utterance" or "speaker" or "global" or "no".')
    if tool not in ['htk', 'python_speech_features', 'librosa']:
        raise TypeError(
            'tool must be "htk" or "python_speech_features" or "librosa".')

    total_frame_num = 0
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
                global_mean = np.zeros((feat_dim,), dtype=dtype)
                global_std = np.zeros((feat_dim,), dtype=dtype)

            global_mean += feat_utt_sum
            total_frame_num += total_frame_num_speaker

            # For computing speaker stddev
            if normalize == 'speaker':
                speaker_mean_dict[speaker] = speaker_mean
                total_frame_num_dict[speaker] = total_frame_num_speaker
                # NOTE: speaker mean is already computed

        print('=====> Computing global mean & stddev...')
        # Compute global mean
        global_mean /= total_frame_num

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
            for feat_utt in feat_dict_speaker.values():
                global_std += np.sum(
                    np.abs(feat_utt - global_mean) ** 2, axis=0)

        # Compute global stddev
        global_std = np.sqrt(global_std / (total_frame_num - 1))

        # Save global mean & std
        np.save(join(save_path, 'global_mean.npy'), global_mean)
        np.save(join(save_path, 'global_std.npy'), global_std)

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
                # Normalize by mean & stddev over the training set
                feat_utt -= global_mean
                feat_utt /= global_std
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
            np.save(mkdir_join(save_path, speaker, utt_idx + '.npy'), feat_utt)

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
