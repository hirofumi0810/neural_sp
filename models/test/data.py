#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

from utils.io.inputs.splicing import do_splice
from utils.io.inputs.frame_stacking import stack_frame
from utils.io.inputs.feature_extraction import wav2feature

SPACE = '_'
SOS = '<'
EOS = '>'


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


def generate_data(model_type, label_type='char', batch_size=1,
                  num_stack=1, splice=1, backend='pytorch'):
    """
    Args:
        model_type (string): ctc or attention
        label_type (string, optional): char or word or word_char
        batch_size (int): the size of mini-batch
        splice (int): frames to splice. Default is 1 frame.
    Returns:
        inputs (np.ndarray or list): A tensor of size `[B, T, input_size]`
        labels: `[B, max_label_seq_len]`
        inputs_seq_len (np.ndarray or list): A tensor of size `[B]`
        labels_seq_len: A tensor of size `[B]`
    """
    # Make input data
    inputs, inputs_seq_len = wav2feature(
        ['../../sample/LDC93S1.wav'] * batch_size,
        feature_type='logfbank', feature_dim=40,
        energy=False, delta1=True, delta2=True, dtype=np.float32)

    max_frame_num = math.ceil(inputs_seq_len[0] / num_stack)
    if backend == 'pytorch':
        inputs_new = np.zeros((batch_size, max_frame_num, inputs.shape[-1] * num_stack * splice),
                              dtype=np.float32)
    elif backend == 'chainer':
        inputs_new = [None] * batch_size

    for i, i_batch in enumerate(range(batch_size)):
        # Frame stacking
        data_i = stack_frame(
            inputs[i_batch], num_stack=num_stack, num_skip=num_stack,
            dtype=np.float32)

        # Splice
        data_i = do_splice(data_i, splice=splice, num_stack=num_stack,
                           dtype=np.float32)

        inputs_new[i_batch] = data_i
        inputs_seq_len[i_batch] = len(data_i)

    # Make transcripts
    trans = _read_text('../../sample/LDC93S1.txt')
    trans = trans.replace('.', '').replace(' ', SPACE)
    if label_type == 'char':
        if model_type == 'attention':
            trans = SOS + trans + EOS
        labels = np.array([char2idx(trans)] * batch_size, dtype=np.int32)
        labels_seq_len = np.array(
            [len(char2idx(trans))] * batch_size, dtype=np.int32)
        return inputs_new, labels, inputs_seq_len, labels_seq_len

    elif label_type == 'word':
        if model_type == 'attention':
            trans = SOS + SPACE + trans + SPACE + EOS
        labels = np.array([word2idx(trans)] * batch_size, dtype=np.int32)
        labels_seq_len = np.array(
            [len(word2idx(trans))] * batch_size, dtype=np.int32)
        return inputs_new, labels, inputs_seq_len, labels_seq_len

    elif label_type == 'word_char':
        if model_type == 'attention':
            trans_word = SOS + SPACE + trans + SPACE + EOS
            trans_char = SOS + trans + EOS
        elif model_type == 'ctc':
            trans_word = trans
            trans_char = trans
        labels = np.array([word2idx(trans_word)] * batch_size, dtype=np.int32)
        labels_sub = np.array(
            [char2idx(trans_char)] * batch_size, dtype=np.int32)
        labels_seq_len = np.array(
            [len(word2idx(trans_word))] * batch_size, dtype=np.int32)
        labels_seq_len_sub = np.array(
            [len(char2idx(trans_char))] * batch_size, dtype=np.int32)
        return inputs_new, labels, labels_sub, inputs_seq_len, labels_seq_len, labels_seq_len_sub

    else:
        raise NotImplementedError


def char2idx(transcript):
    """Convert from character to index.
    Args:
        transcript (string): a sequence of string
    Returns:
        index_list (list): indices of characters
    """
    char_list = list(transcript)

    first_idx = ord('a') - 1
    last_idx = ord('z') - first_idx
    # NOTE: 0 is reserved for space

    index_list = []
    for char in char_list:
        if char == SPACE:
            index_list.append(0)
        elif char == SOS:
            index_list.append(last_idx + 2)
        elif char == EOS:
            index_list.append(last_idx + 1)
        else:
            index_list.append(ord(char) - first_idx)
    return index_list


def idx2char(indices):
    """Convert from index to character.
    Args:
        indices (Variable): Variable of indices
        blank_index (int, optional): the index of the blank class
    Returns:
        transcript (string): a sequence of string
    """
    if isinstance(indices, np.ndarray):
        indices = indices.tolist()

    first_idx = ord('a') - 1
    last_idx = ord('z') - first_idx
    # NOTE: 0 is reserved for space

    char_list = []
    for idx in indices:
        if idx == 0:
            char_list.append(SPACE)
        elif idx == last_idx + 2:
            char_list.append(SOS)
        elif idx == last_idx + 1:
            char_list.append(EOS)
        else:
            char_list.append(chr(idx + first_idx))
    transcript = ''.join(char_list)
    return transcript


def word2idx(transcript):
    """Convert from word to index.
    Args:
        transcript (string): a sequence of space-separated string
    Returns:
        index_list (list): indices of words
    """
    word_list = transcript.split(SPACE)

    # Register word dict
    vocab = set([])
    for word in word_list:
        if word in [SOS, EOS]:
            continue
        vocab.add(word)

    word_dict = {}
    with open('../../word.txt', 'w') as f:
        for idx, word in enumerate(sorted(list(vocab))):
            word_dict[word] = idx
            f.write('%s\n' % word)
        word_dict[SOS] = len(vocab) + 1
        word_dict[EOS] = len(vocab)
        f.write('%s\n' % EOS)
        f.write('%s\n' % SOS)

    index_list = []
    for word in word_list:
        index_list.append(word_dict[word])
    return index_list


def idx2word(indices):
    """Convert from index to word.
    Args:
        indices (Variable): Variable of indices
        blank_index (int, optional): the index of the blank class
    Returns:
        transcript (string): a sequence of string
    """
    if isinstance(indices, np.ndarray):
        indices = indices.tolist()

    word_dict = {}
    with open('../../word.txt', 'r') as f:
        for idx, line in enumerate(f):
            word = line.strip()
            word_dict[idx] = word

    word_list = []
    for idx in indices:
        word_list.append(word_dict[idx])
    transcript = SPACE.join(word_list)
    return transcript
