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

SPACE_INDEX = 0
BLANK_INDEX = 27
SOS_INDEX = 27
EOS_INDEX = 28


def np2var_chainer(inputs):
    """
    Args:
        inputs (np.ndarray): A tensor of size `[B, T, input_size]`
    Returns:
        chainer.Variable of size `[T, B, input_size]`
    """
    return chainer.Variable(inputs, requires_grad=False)


def np2varlist_chainer(inputs):
    """
    Args:
        inputs (np.ndarray): A tensor of size `[B, T, input_size]`
    Returns:
        var_list (list): list of character.Variable of size `[T, input_size]`
            Note that len(var_list) == B.
    """
    assert len(inputs.shape) == 3

    var_list = []
    for i_batch in range(inputs.shape[0]):
        var_list.append(chainer.Variable(inputs[i_batch], requires_grad=False))
    # volatile??

    return var_list


def np2var_pytorch(inputs, is_chainer=False, return_list=False):
    """Convert form np.ndarray to Variable.
    Args:
        inputs (np.ndarray): A tensor of size `[B, T, input_size]`
        is_chainer (bool, optional): if True, return chainer.Variable
        return_list (bool, optional): if True, return list of chainer.Variable
    Returns:

    """
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
        inputs: A tensor of size `[B, T, input_size]`
        labels: `[B, max_label_seq_len]`
        inputs_seq_len: A tensor of size `[B]`
        labels_seq_len: A tensor of size `[B]`
    """
    # Make input data
    inputs, inputs_seq_len = wav2feature(
        ['./sample/LDC93S1.wav'] * batch_size,
        feature_type='logfbank', feature_dim=40,
        energy=True, delta1=True, delta2=True, dtype=np.float32)

    # Splicing
    # inputs = do_splice(inputs, splice=splice)

    # Make transcripts
    transcript = _read_text('./sample/LDC93S1.txt').replace('.', '')
    if model == 'attention':
        transcript = '<' + transcript + '>'
    labels = np.array([alpha2idx(transcript)] * batch_size, np.int32)
    labels_seq_len = np.array([len(labels[0])] * batch_size)

    return inputs, labels, inputs_seq_len, labels_seq_len


def alpha2idx(transcript):
    """Convert from alphabet to index.
    Args:
        transcript (string): a sequence of characters
    Returns:
        index_list (list): indices of alphabets
    """
    char_list = list(transcript)

    # 0 is reserved for space
    first_index = ord('a') - 1
    index_list = []
    for char in char_list:
        if char == ' ':
            index_list.append(SPACE_INDEX)
        elif char == '<':
            index_list.append(SOS_INDEX)
        elif char == '>':
            index_list.append(EOS_INDEX)
        else:
            index_list.append(ord(char) - first_index)
    return index_list


def idx2alpha(indices):
    """Convert from index to alphabet.
    Args:
        indices (Variable): Variable of indices
        blank_index (int, optional): the index of the blank class
    Returns:
        transcript (string): a sequence of character
    """
    # 0 is reserved to space
    first_index = ord('a') - 1
    char_list = []
    for var in indices:
        if var.data == 0:
            char_list.append(' ')
        elif var.data == BLANK_INDEX:
            continue
            # TODO: fix this
        elif var.data == SOS_INDEX:
            char_list.append('<')
        elif var.data == EOS_INDEX:
            char_list.append('>')
        else:
            char_list.append(chr(var.data + first_index))
    transcript = ''.join(char_list)
    return transcript
