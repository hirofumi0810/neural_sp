#! /usr/bin/env python
# -*- coding: utf-8 -*

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io.wavfile
from python_speech_features import mfcc, fbank


def wav2feature(wav_paths, feature_type='logfbank', feature_dim=40,
                energy=True, delta1=True, delta2=True):
    """Read wav file & convert to MFCC or log mel filterbank features.
    Args:
        wav_paths: list of the path to a wav file
        batch_size: int, the batch size
        feature_type: logfbank or fbank or mfcc
        feature_dim: int, the demension of each feature
        energy: if True, add energy
        delta1: if True, add delta features
        delta2: if True, add delta delta features
    Returns:
        inputs: A tensor of size `[B, T, input_size]`
        inputs_seq_len: A tensor of size `[B]`
    """
    if feature_type not in ['logmelfbank', 'logfbank', 'fbank', 'mfcc']:
        raise ValueError(
            'feature_type is "logmelfbank" or "logfbank" or "fbank" or "mfcc".')
    if not isinstance(wav_paths, list):
        raise ValueError('wav_paths must be a list.')
    if delta2 and not delta1:
        delta1 = True

    batch_size = len(wav_paths)
    max_time = 0
    for wav_path in wav_paths:
        # Read wav file
        fs, audio = scipy.io.wavfile.read(wav_path)
        if len(audio) > max_time:
            max_time = len(audio)
    input_size = feature_dim
    if energy:
        input_size + 1
    if delta2:
        input_size *= 3
    elif delta1:
        input_size *= 2

    inputs = None
    inputs_seq_len = np.zeros((batch_size,))
    for i, wav_path in enumerate(wav_paths):
        if feature_type == 'mfcc':
            feat = mfcc(audio, samplerate=fs, numcep=feature_dim)
            if energy:
                energy_feat = fbank(audio, samplerate=fs, nfilt=feature_dim)[1]
                feat = np.c_[feat, energy_feat]
        else:
            fbank_feat, energy_feat = fbank(
                audio, samplerate=fs, nfilt=feature_dim)
            if feature_type == 'logfbank':
                fbank_feat = np.log(fbank_feat)
            feat = fbank_feat
            if energy:
                # logenergy = np.log(energy_feat)
                feat = np.c_[feat, energy_feat]

        if delta2:
            delta1_feat = _delta(feat, N=2)
            delta2_feat = _delta(delta1_feat, N=2)
            feat = np.c_[feat, delta1_feat, delta2_feat]
        elif delta1:
            delta1_feat = _delta(feat, N=2)
            feat = np.c_[feat, delta1_feat]

        # Normalize per wav
        feat = (feat - np.mean(feat)) / np.std(feat)

        if inputs is None:
            max_time = feat.shape[0]
            input_size = feat.shape[-1]
            inputs = np.zeros((batch_size, max_time, input_size))

        inputs[i] = feat
        inputs_seq_len[i] = len(feat)

    return inputs, inputs_seq_len


def _delta(feat, N):
    """Compute delta features from a feature vector sequence.
    Args:
        feat: A numpy array of size (NUMFRAMES by number of features)
            containing features. Each row holds 1 feature vector.
        N: For each frame, calculate delta features based on preceding and
            following N frames
    Returns:
        A numpy array of size (NUMFRAMES by number of features) containing
            delta features. Each row holds 1 delta feature vector.
    """
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N + 1)])
    delta_feat = np.empty_like(feat)
    # padded version of feat
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')
    for t in range(NUMFRAMES):
        # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
        delta_feat[t] = np.dot(np.arange(-N, N + 1),
                               padded[t: t + 2 * N + 1]) / denominator
    return delta_feat
