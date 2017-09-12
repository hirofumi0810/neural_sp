#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import chainer
import numpy as np

# from utils.data.inputs.splicing import do_splice
from preprocessing.feature_extraction_python_speech_features import wav2feature


def np2var(inputs, is_chainer=False, return_list=False):
    """Convert form np.ndarray to Variable.
    Args:
        inputs (np.ndarray):
        is_chainer (bool, optional): if True, return chainer.Variable
        return_list(bool, optional): if True, return list of chainer.Variable
    Returns:
        
    """
    if is_chainer:
        if not return_list:
            return chainer.Variable(inputs, requires_grad=False)

        var_list = []
        for i in range(inputs.shape[0]):
            if len(inputs.shape) != 1:
                var_list.append(
                    chainer.Variable(inputs[i], requires_grad=False))
            else:
                var_list.append(
                    chainer.Variable(np.array(inputs[i]), requires_grad=False))
        # volatile??
        return var_list
    else:
        return Variable(torch.from_numpy(inputs).float(), requires_grad=False)
        # NOTE: which are better, 32-bit or 64-bit?


def _read_text(trans_path):
    """Read char-level transcripts.
    Args:
        trans_path (string): path to a transcript text file
    Returns:
        transcript (strig): a text of transcript
    """
    # Read ground truth labels
    with open(trans_path, 'r') as f:
        line = f.readlines()[-1]
        transcript = ' '.join(line.strip().lower().split(' ')[2:])
    return transcript


def generate_data(model, batch_size=1, splice=1):
    """
    Args:
        model (string): ctc or attention
        batch_size (int): the size of mini-batch
        splice (int): frames to splice. Default is 1 frame.
    Returns:
        inputs: `[B, T, input_size]`
        labels: `[B]`
        inputs_seq_len: `[B, frame_num]`
        labels_seq_len: `[B]` (if model is attention)
    """
    # Make input data
    inputs, inputs_seq_len = wav2feature(
        ['./sample/LDC93S1.wav'] * batch_size,
        feature_type='logfbank', feature_dim=40,
        energy=True, delta1=True, delta2=True, dtype=np.float32)

    # Splicing
    # inputs = do_splice(inputs, splice=splice)

    # Make transcripts
    if model == 'ctc':
        transcript = _read_text('./sample/LDC93S1.txt').replace('.', '')
        labels = np.array([alpha2idx(transcript)] * batch_size, np.int32)
        labels_seq_len = np.array([len(labels[0])] * batch_size)
        return inputs, labels, inputs_seq_len
        # return inputs, labels, inputs_seq_len, labels_seq_len

    elif model == 'attention':
        transcript = _read_text('./sample/LDC93S1.txt').replace('.', '')
        transcript = '<' + transcript + '>'
        labels = np.array([alpha2idx(transcript)] * batch_size, np.int32)
        labels_seq_len = np.array([len(labels[0])] * batch_size)
        return inputs, labels, inputs_seq_len, labels_seq_len


def alpha2idx(transcript):
    """Convert from alphabet to index.
    Args:
        transcript (string): sequence of characters
    Returns:
        index_list (list): list of indices
    """
    char_list = list(transcript)

    # 0 is reserved for space
    space_index = 0
    first_index = ord('a') - 1
    index_list = []
    for char in char_list:
        if char == ' ':
            index_list.append(space_index)
        elif char == '<':
            index_list.append(26)
        elif char == '>':
            index_list.append(27)
        else:
            index_list.append(ord(char) - first_index)
    return index_list


def idx2alpha(index_list):
    """Convert from index to alphabet.
    Args:
        index_list (list): list of indices
    Returns:
        transcript (string): sequence of character
    """
    # 0 is reserved to space
    first_index = ord('a') - 1
    char_list = []
    for num in index_list:
        if num == 0:
            char_list.append(' ')
        elif num == 26:
            char_list.append('<')
        elif num == 27:
            char_list.append('>')
        else:
            char_list.append(chr(num + first_index))
    transcript = ''.join(char_list)
    return transcript
