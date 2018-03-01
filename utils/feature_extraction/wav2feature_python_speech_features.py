#! /usr/bin/env python
# -*- coding: utf-8 -*

"""python_speech_features based feature extraction.
        See details in https://github.com/jameslyons/python_speech_features
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess
import numpy as np
import scipy.io.wavfile
from python_speech_features import mfcc, fbank


def wav2feature(wav_path, feature_type='fbank', feature_dim=40,
                use_energy=True, use_delta1=True, use_delta2=True,
                window=0.025, slide=0.01, dtype=np.float32):
    """Read wav file & convert to MFCC or log mel filterbank features.
    Args:
        wav_path (string): the path to a wav file
        feature_type (string, optional): fbank or mfcc
        feature_dim (int, optional): the demension of each feature
        use_energy (bool, optional): if True, add energy
        use_delta1 (bool, optional): if True, add delta features
        use_delta2 (bool, optional): if True, add delta delta features
        window (float, optional): window width to extract features
        slide (float, optional): extract features per 'slide'
        dtype (optional): default is np.float32
    Returns:
        feat (np.ndarray): A tensor of size `[T, feature_dim]`
    """
    if feature_type not in ['fbank', 'mfcc']:
        raise ValueError('feature_type is or "fbank" or "mfcc".')
    if use_delta2:
        delta1 = True

    # Read wav file
    try:
        fs, audio = scipy.io.wavfile.read(wav_path)
    except ValueError:
        # Read NIST file
        wav_path_tmp = '/tmp//tmp.wav'
        # result = subprocess.call(['sph2pipe', '-f', 'wav', wav_path, wav_path_tmp])
        result = subprocess.call(['sox', wav_path, '-t', 'wav', wav_path_tmp])

        if result != 0:
            raise ValueError

        # Try again
        fs, audio = scipy.io.wavfile.read(wav_path_tmp)
        subprocess.call(['rm', wav_path_tmp])

    if feature_type == 'mfcc':
        feat = mfcc(audio,
                    samplerate=fs,
                    numcep=feature_dim)
        if use_energy:
            energy_feat = fbank(audio,
                                samplerate=fs,
                                nfilt=feature_dim)[1]
            energy_feat = energy_feat.reshape(-1, 1)
            feat = np.concatenate((feat, energy_feat), axis=1)
            # NOTE: only fbank function retures energy
    else:
        fbank_feat, energy_feat = fbank(audio,
                                        samplerate=fs,
                                        winlen=window,
                                        winstep=slide,
                                        nfilt=feature_dim,
                                        nfft=512,
                                        lowfreq=0,
                                        highfreq=None,
                                        preemph=0.97,
                                        winfunc=np.hamming)
        if feature_type == 'fbank':
            feat = np.log(fbank_feat)
        if use_energy:
            energy_feat = energy_feat.reshape(-1, 1)
            # logenergy = np.log(energy_feat)

            feat = np.concatenate((feat, energy_feat), axis=1)
            # NOTE: energy_feat may be not log-scale.

    if use_delta2:
        delta1_feat = _delta(feat, N=2)
        delta2_feat = _delta(delta1_feat, N=2)
        feat = np.concatenate((feat, delta1_feat, delta2_feat), axis=1)
    elif delta1:
        delta1_feat = _delta(feat, N=2)
        feat = np.concatenate((feat, delta1_feat), axis=1)

    return feat


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
