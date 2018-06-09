#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from utils.io.inputs.feature_extraction import wav2feature

SPACE = '_'
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


def generate_data(label_type='char', batch_size=1):
    """Generate dataset for unit test.
    Args:
        label_type (string, optional): char or word or word_char
        batch_size (int): the size of mini-batch
    Returns:
        xs (list): A list of length `[B]`, which contains arrays of size `[T, input_size]`
        ys (list): A list of length `[B]`
        ys_sub (list): A list of length `[B]`
    """
    # Make input data
    _xs, _ = wav2feature(
        ['../../sample/LDC93S1.wav'] * batch_size,
        feature_type='logfbank', feature_dim=40,
        energy=False, delta1=True, delta2=True, dtype=np.float32)

    xs = []
    for i, b in enumerate(range(batch_size)):
        xs += [_xs[b][: len(_xs[b]) - i]]

    # Make transcripts
    trans = _read_text('../../sample/LDC93S1.txt')
    trans = trans.replace('.', '').replace(' ', SPACE)
    if label_type == 'char':
        ys = [char2idx(trans)] * batch_size
        return xs, ys
    elif label_type == 'word':
        ys = [word2idx(trans)] * batch_size
        return xs, ys
    elif label_type == 'word_char':
        ys = [word2idx(trans)] * batch_size
        ys_sub = [char2idx(trans)] * batch_size
        return xs, ys, ys_sub
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
        if word in [EOS]:
            continue
        vocab.add(word)

    word_dict = {}
    with open('../../word.txt', 'w') as f:
        for idx, word in enumerate(sorted(list(vocab))):
            word_dict[word] = idx
            f.write('%s\n' % word)
        word_dict[EOS] = len(vocab)
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
