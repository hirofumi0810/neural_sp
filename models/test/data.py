#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import chainer
import numpy as np

from utils.io.inputs.splicing import do_splice
from utils.io.inputs.frame_stacking import stack_frame
from utils.io.inputs.feature_extraction import wav2feature

SPACE = '_'
SOS = '<'
EOS = '>'

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


def np2var_pytorch(inputs, dtype='float'):
    """Convert form np.ndarray to Variable.
    Args:
        inputs (np.ndarray): A tensor of size `[B, T, input_size]`
        type (string, optional): float or long or int
    Returns:
        inputs (torch.Variable): A tensor of size `[B, T, input_size]`
    """
    inputs = torch.from_numpy(inputs)
    if dtype == 'float':
        inputs = inputs.float()
    elif dtype == 'long':
        inputs = inputs.long()
    elif dtype == 'int':
        inputs = inputs.int()

    inputs = Variable(inputs, requires_grad=False)
    # NOTE: which is better, 32-bit or 64-bit?

    return inputs


def _read_text(trans_path):
    """Read char-level transcripts.
    Args:
        trans_path (string): path to a transcript text file
    Returns:
        transcript (string): a text of transcript
    """
    # Read ground truth labels
    with open(trans_path, 'r') as f:
        line = f.readlines()[-1]
        transcript = ' '.join(line.strip().lower().split(' ')[2:])
    return transcript


def generate_data(model, batch_size=1, num_stack=1, splice=1):
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
        ['../sample/LDC93S1.wav'] * batch_size,
        feature_type='logfbank', feature_dim=40,
        energy=False, delta1=True, delta2=True, dtype=np.float32)

    # Frame stacking
    inputs = stack_frame(inputs,
                         num_stack=num_stack,
                         num_skip=num_stack,
                         progressbar=False)
    if num_stack != 1:
        for i in range(len(inputs_seq_len)):
            inputs_seq_len[i] = len(inputs[i])

    # Splice
    inputs = do_splice(inputs,
                       splice=splice,
                       batch_size=batch_size,
                       num_stack=num_stack)

    # Make transcripts
    transcript = _read_text('../sample/LDC93S1.txt').replace('.', '')
    if model == 'attention':
        transcript = SOS + transcript + EOS
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
        elif char == SOS:
            index_list.append(SOS_INDEX)
        elif char == EOS:
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
    # NOTE: 0 is reserved to space

    first_index = ord('a') - 1
    char_list = []
    for index in indices:
        if index == 0:
            char_list.append(SPACE)
        elif index == BLANK_INDEX:
            continue
            # TODO: fix this
        elif index == SOS_INDEX:
            char_list.append(SOS)
        elif index == EOS_INDEX:
            char_list.append(EOS)
        else:
            char_list.append(chr(index + first_index))
    transcript = ''.join(char_list)
    return transcript
