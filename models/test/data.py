#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


def generate_data(model, label_type='char', batch_size=1, num_stack=1, splice=1):
    """
    Args:
        model (string): ctc or attention
        label_type (string, optional): char or word
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
    transcript = _read_text('../sample/LDC93S1.txt')
    transcript = transcript.replace('.', '').replace(' ', SPACE)
    if label_type == 'char':
        if model == 'attention':
            transcript = SOS + transcript + EOS
            labels = np.array([char2idx(transcript)] * batch_size, np.int32)
        elif model == 'ctc':
            labels = np.array([char2idx(transcript)] * batch_size, np.int32)
            labels = labels.reshape((-1,))
        labels_seq_len = np.array([len(char2idx(transcript))] * batch_size)
    elif label_type == 'word':
        if model == 'attention':
            transcript = SOS + SPACE + transcript + SPACE + EOS
            labels = np.array([word2idx(transcript)] * batch_size, np.int32)
        elif model == 'ctc':
            labels = np.array([word2idx(transcript)] * batch_size, np.int32)
            labels = labels.reshape((-1,))
        labels_seq_len = np.array([len(word2idx(transcript))] * batch_size)
    elif label_type == 'word_char':
        pass
    else:
        raise NotImplementedError

    return inputs, labels, inputs_seq_len, labels_seq_len


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
            index_list.append(last_idx + 1)
        elif char == EOS:
            index_list.append(last_idx + 2)
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
        indices = list(indices)

    first_idx = ord('a') - 1
    last_idx = ord('z') - first_idx
    # NOTE: 0 is reserved for space

    char_list = []
    for idx in indices:
        if idx == 0:
            char_list.append(SPACE)
        elif idx == last_idx + 1:
            char_list.append(SOS)
        elif idx == last_idx + 2:
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
    with open('./word.txt', 'w') as f:
        for idx, word in enumerate(sorted(list(vocab))):
            word_dict[word] = idx
            f.write('%s\n' % word)
        word_dict[SOS] = len(vocab)
        word_dict[EOS] = len(vocab) + 1
        f.write('%s\n' % SOS)
        f.write('%s\n' % EOS)

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
        indices = list(indices)

    word_dict = {}
    with open('./word.txt', 'r') as f:
        for idx, line in enumerate(f):
            word = line.strip()
            word_dict[idx] = word

    word_list = []
    for idx in indices:
        word_list.append(word_dict[idx])
    transcript = SPACE.join(word_list)
    return transcript
