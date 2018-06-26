#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make configuration file for HTK toolkit (WSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse

sys.path.append('../../../')
from src.utils.feature_extraction.htk import save_config

parser = argparse.ArgumentParser()
parser.add_argument('--data_save_path', type=str,
                    help='path to save data')
parser.add_argument('--config_save_path', type=str,
                    help='path to save the configuration file')

parser.add_argument('--audio_file_type', type=str,
                    help='wav or nist')
parser.add_argument('--channels', type=int,
                    help='the number of frequency channels')
parser.add_argument('--sampling_rate', type=int,
                    help='sampling rate')
parser.add_argument('--window', type=float, default=0.025,
                    help='window width to extract features')
parser.add_argument('--slide', type=float, default=0.01,
                    help='extract features per slide')
parser.add_argument('--energy', type=int,
                    help='if 1, add the energy feature')
parser.add_argument('--delta', type=int,
                    help='if 1, add the energy feature')
parser.add_argument('--deltadelta', type=int,
                    help='if 1, double delta features are also extracted')


def main():

    args = parser.parse_args()

    # HTK settings
    save_config(audio_file_type=args.audio_file_type,
                feature_type='fbank',
                channels=args.channels,
                config_save_path=args.config_save_path,
                sampling_rate=args.sampling_rate,
                window=args.window,
                slide=args.slide,
                energy=bool(args.energy),
                delta=bool(args.delta),
                deltadelta=bool(args.deltadelta))


if __name__ == '__main__':
    main()
