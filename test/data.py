#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from src.utils.io.inputs.feature_extraction import wav2feature

SPACE = '_'
EOS = '>'


def _read_text(trans_path):
    """Read char-level transcripts.
    Args:
        trans_path (string): path to a transcript text file
    Returns:
        trans (string): a text of transcript
    """
    # Read ground truth labels
    with open(trans_path, 'r') as f:
        line = f.readlines()[-1]
        trans = SPACE.join(line.strip().lower().split(' ')[2:])
    return trans


def _read_phone(trans_path):
    """Read phoneme-level transcripts.
    Args:
        trans_path (string): path to a transcript text file
    Returns:
        transcript (string): a text of transcript
    """
    # Read ground truth labels
    trans = ''
    with open(trans_path, 'r') as f:
        for line in f:
            phone = line.strip().split(' ')[-1]
            trans += SPACE + phone
    return trans[1:]


def generate_data(label_type='char', batch_size=1):
    """Generate dataset for unit test.
    Args:
        label_type (string, optional): char or word or phone or word_char or word_phone
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
    if label_type in ['phone', 'p2w']:
        trans = _read_phone('../../sample/LDC93S1.phn')
    else:
        trans = _read_text('../../sample/LDC93S1.txt')
        trans = trans.replace('.', '')
    if label_type == 'char':
        ys = [char2idx(trans)] * batch_size
        return xs, ys
    elif label_type == 'word':
        ys = [word2idx(trans)] * batch_size
        return xs, ys
    elif label_type == 'phone':
        ys = [word2idx(trans, vocab_path='../../phone.txt')] * batch_size
        return xs, ys
    elif label_type == 'word_char':
        ys = [word2idx(trans)] * batch_size
        ys_sub = [char2idx(trans)] * batch_size
        return xs, ys, ys_sub
    elif label_type == 'word_phone':
        trans_phone = _read_phone('../../sample/LDC93S1.phn')
        ys = [word2idx(trans)] * batch_size
        ys_sub = [
            word2idx(trans_phone, vocab_path='../../phone.txt')] * batch_size
        return xs, ys, ys_sub
    else:
        raise NotImplementedError(label_type)


def generate_data_p2w(label_type_in='phone', label_type_out='word', batch_size=1):
    """Generate dataset for unit test.
    Args:
        label_type_in (string, optional): phone or char
        label_type_out (string, optional): char or word
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
    if label_type_in == 'phone':
        trans_in = _read_phone('../../sample/LDC93S1.phn')
    else:
        trans_in = _read_text('../../sample/LDC93S1.txt')
        trans_in = trans_in.replace('.', '')
    trans_out = _read_text('../../sample/LDC93S1.txt')
    trans_out = trans_out.replace('.', '')

    if label_type_in == 'char':
        ys_in = [char2idx(trans_in)] * batch_size
    if label_type_in == 'phone':
        ys_in = [word2idx(trans_in, vocab_path='../../phone.txt')] * batch_size

    if label_type_out == 'word':
        ys_out = [word2idx(trans_out)] * batch_size
    elif label_type_out == 'char':
        ys_out = [char2idx(trans_out)] * batch_size

    return ys_in, ys_out


def char2idx(trans):
    """Convert from character to index.
    Args:
        trans (string): a sequence of string
    Returns:
        index_list (list): indices of characters
    """
    chars = list(trans)

    first_idx = ord('a') - 1
    last_idx = ord('z') - first_idx
    # NOTE: 0 is reserved for space

    index_list = []
    for char in chars:
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
        trans (string): a sequence of string
    """
    if isinstance(indices, np.ndarray):
        indices = indices.tolist()

    first_idx = ord('a') - 1
    last_idx = ord('z') - first_idx
    # NOTE: 0 is reserved for space

    chars = []
    for idx in indices:
        if idx == 0:
            chars.append(SPACE)
        elif idx == last_idx + 1:
            chars.append(EOS)
        else:
            chars.append(chr(idx + first_idx))
    trans = ''.join(chars)
    return trans


def word2idx(trans, vocab_path='../../word.txt'):
    """Convert from word to index.
    Args:
        trans (string): a sequence of space-separated string
    Returns:
        index_list (list): indices of words
    """
    words = trans.split(SPACE)

    # Register word dict
    vocab = set([])
    for word in words:
        if word in [EOS]:
            continue
        vocab.add(word)

    word_dict = {}
    with open(vocab_path, 'w') as f:
        for idx, word in enumerate(sorted(list(vocab))):
            word_dict[word] = idx
            f.write('%s\n' % word)
        word_dict[EOS] = len(vocab)
        f.write('%s\n' % EOS)

    index_list = []
    for word in words:
        index_list.append(word_dict[word])
    return index_list


def idx2word(indices, vocab_path='../../word.txt'):
    """Convert from index to word.
    Args:
        indices (Variable): Variable of indices
        blank_index (int, optional): the index of the blank class
    Returns:
        trans (string): a sequence of string
    """
    if isinstance(indices, np.ndarray):
        indices = indices.tolist()

    word_dict = {}
    with open(vocab_path, 'r') as f:
        for idx, line in enumerate(f):
            word = line.strip()
            word_dict[idx] = word

    words = []
    for idx in indices:
        words.append(word_dict[idx])
    trans = SPACE.join(words)
    return trans
