#! /usr/bin/env python
# -*- coding: utf-8 -*

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""librosa based feature extraction.
        https://github.com/librosa/librosa
"""

import librosa
import subprocess
import numpy as np


def wav2feature(wav_path, feature_type='logfbank', feature_dim=40,
                use_energy=True, use_delta1=True, use_delta2=True,
                window=0.025, slide=0.01, dtype=np.float64):
    """Read wav file & convert to MFCC or log mel filterbank features.
    Args:
        wav_path (string): the path to a wav file
        feature_type (string, optional): logfbank or fbank or mfcc
        feature_dim (int, optional): the demension of each feature
        use_energy (bool, optional): if True, add energy
        use_delta1 (bool, optional): if True, add delta features
        use_delta2 (bool, optional): if True, add delta delta features
        window (float, optional): window width to extract features
        slide (float, optional): extract features per 'slide'
        dtype (optional): default is np.float64
    Returns:
        feat (np.ndarray): A tensor of size `[T, feature_dim]`
    """
    if feature_type == 'logmelfbank':
        feature_type = 'logfbank'
    if feature_type not in ['logfbank', 'fbank', 'mfcc']:
        raise ValueError('feature_type is or "logfbank" or "fbank" or "mfcc".')
    if use_delta2:
        delta1 = True

    # Read wav file
    try:
        y, sr = librosa.load(wav_path)
    except ValueError:
        # Read NIST file
        wav_path_tmp = './tmp.wav'
        # result = subprocess.call(['sph2pipe', '-f', 'wav', wav_path, wav_path_tmp])
        result = subprocess.call(['sox', wav_path, '-t', 'wav', wav_path_tmp])

        if result != 0:
            raise ValueError

        # Try again
        y, sr = librosa.load(wav_path_tmp)
        subprocess.call(['rm', wav_path_tmp])

    if feature_type == 'mfcc':
        feat = librosa.feature.mfcc(y=y,
                                    sr=sr,
                                    # S=None,
                                    n_mfcc=feature_dim)
        if use_energy:
            rmse = librosa.feature.rmse(y=y,
                                        frame_length=2048,
                                        hop_length=512)[0]
            feat = np.concatenate((feat, rmse), axis=0)
    else:
        feat = librosa.feature.melspectrogram(y=y,
                                              sr=sr,
                                              S=None,
                                              n_fft=2048,
                                              hop_length=512,
                                              #   power=2.0,  # default is 2
                                              n_mels=feature_dim,
                                              fmin=0,
                                              fmax=None)
        # NOTE: feat: `[feature_dim, T]`

        if feature_type == 'logfbank':
            # feat = librosa.core.logamplitude(feat)
            feat = librosa.core.spectrum.power_to_db(feat)
        if use_energy:
            rmse = librosa.feature.rmse(y=y,
                                        frame_length=2048,
                                        hop_length=512)
            # NOTE: `[1, T]`
            feat = np.concatenate((feat, rmse), axis=0)

    # Convert to time-major
    feat = feat.transpose((1, 0))

    if use_delta2:
        delta1_feat = librosa.feature.delta(feat, width=9)
        delta2_feat = librosa.feature.delta(delta1_feat, width=9)
        feat = np.concatenate((feat, delta1_feat, delta2_feat), axis=1)
    elif delta1:
        delta1_feat = librosa.feature.delta(feat, width=9)
        feat = np.concatenate((feat, delta1_feat), axis=1)

    return feat
