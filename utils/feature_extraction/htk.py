#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
from struct import unpack, pack
import numpy as np


def read(htk_path):
    """Read each HTK file.
    Args:
        htk_path (string): path to a HTK file
    Returns:
        input_data (np.ndarray): A tensor of size (frame_num, feature_dim)
        sampPeriod (int):
        parmKind (int):
    """
    # print('...Reading: %s' % htk_path)
    with open(htk_path, "rb") as f:
        # Read header
        spam = f.read(12)
        frame_num, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)

        # for debug
        # print(frame_num)  # frame num
        # print(sampPeriod)  # 10ms
        # print(sampSize)  # feature dim * 4 (byte)
        # print(parmKind)

        # Read data
        feature_dim = int(sampSize / 4)
        f.seek(12, 0)
        input_data = np.fromfile(f, 'f')
        try:
            input_data = input_data.reshape(-1, feature_dim)
        except:
            print(input_data.shape)
            raise ValueError

        input_data.byteswap(True)

    return input_data, sampPeriod, parmKind


def write(input_data, htk_path, sampPeriod, parmKind):
    """Save numpy array as a HTK file.
    Args:
        input_data (np.ndarray): A tensor of size (frame_num, feature_dim)
        htk_path (string): path to a HTK file
        sampPeriod (int):
        parmKind (int):
    """
    # print('...Saving: %s' % htk_path)
    with open(htk_path, "wb") as f:
        # Write header
        frame_num, feature_dim = input_data.shape
        sampSize = feature_dim * 4
        f.write(pack(">iihh", frame_num, sampPeriod, sampSize, parmKind))

        # Write data
        f.write(pack(">%df" % (frame_num * feature_dim), *input_data.ravel()))


def save_config(audio_file_type, feature_type, channels, config_save_path,
                sampling_rate=16000, window=0.025, slide=0.01,
                energy=True, delta=True, deltadelta=True):
    """Save a configuration file for HTK.
    Args:
        audio_file_type (string): nist or wav
        feature_type (string): the type of features, logmelfbank or mfcc
        channels (int): the number of frequency channels
        config_save_path (string): path to save the config file
        sampling_rate (float, optional):
        window (float, optional): window width to extract features
        slide (float, optional): extract features per 'slide'
        energy (bool, optional): if True, add the energy feature
        delta (bool, optional): if True, delta features are also extracted
        deltadelta (bool, optional): if True, double delta features are also extracted
"""
    with open(join(config_save_path, feature_type + '.conf'), 'w') as f:
        if audio_file_type not in ['nist', 'wav']:
            raise ValueError('audio_file_type must be nist or wav.')
        f.write('SOURCEFORMAT = %s\n' % audio_file_type.upper())

        # Sampling rate
        if sampling_rate == 16000:
            f.write('SOURCERATE = 625\n')
        elif sampling_rate == 8000:
            f.write('SOURCERATE = 1250\n')

        # Target features
        if feature_type == 'fbank':
            feature_type = 'FBANK'  # log mel-filter bank channel outputs
        elif feature_type == 'mfcc':
            feature_type = 'MFCC'  # mel-frequency cepstral coefficients
        # elif feature_type == 'linearmelfbank':
        # feature_type = 'MELSPEC'  # linear mel-filter bank channel outputs
        else:
            raise ValueError('feature_type must be fbank or mfcc.')

        if energy:
            feature_type += '_E'
        if delta:
            feature_type += '_D'
        if deltadelta:
            feature_type += '_A'
        f.write('TARGETKIND = %s\n' % feature_type)

        # f.write('DELTAWINDOW = 2')
        # f.write('ACCWINDOW = 2')

        # Extract features per slide
        f.write('TARGETRATE = %.1f\n' % (slide * 10000000))

        f.write('SAVECOMPRESSED = F\n')
        f.write('SAVEWITHCRC = F\n')

        # Window size
        f.write('WINDOWSIZE = %.1f\n' % (window * 10000000))

        f.write('USEHAMMING = T\n')
        f.write('PREEMCOEF = 0.97\n')
        f.write('NUMCHANS = %d\n' % channels)
        f.write('ENORMALISE = F\n')
        f.write('ZMEANSOURCE = T\n')
