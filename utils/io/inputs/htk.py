#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from struct import unpack
import numpy as np


def read_htk(htk_path):
    """Read each HTK file.
    Args:
        htk_path (string): path to a HTK file
    Returns:
        input_data (np.ndarray): A tensor of size (frame_num, feature_dim)
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
        input_data = input_data.reshape(-1, feature_dim)
        input_data.byteswap(True)

    return input_data
