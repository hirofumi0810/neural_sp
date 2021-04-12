# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Utility functions for data loader."""

import codecs
import random


def count_vocab_size(dict_path):
    vocab_count = 1  # for <blank>
    with codecs.open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() != '':
                vocab_count += 1
    return vocab_count


def _set_batch_size_seq(batch_size, min_xlen, min_ylen, dynamic_batching, num_replicas):
    if not dynamic_batching:
        return batch_size

    if min_xlen <= 800:
        pass
    elif min_xlen <= 1600 or 80 < min_ylen <= 100:
        batch_size //= 2
    else:
        batch_size //= 8

    batch_size = batch_size // num_replicas * num_replicas
    batch_size = max(num_replicas, batch_size)
    # NOTE: ensure batch size>=1 for all replicas
    return batch_size


def _set_batch_size_bin(max_n_bins, lengths, num_replicas):
    total_bin = 0
    batch_size = 0
    for length in lengths:
        if length > max_n_bins:
            raise ValueError(f"max_n_bins is too small: {max_n_bins}")
        if total_bin + length <= max_n_bins:
            total_bin += length
            batch_size += 1
        else:
            break

    batch_size = batch_size // num_replicas * num_replicas
    batch_size = max(num_replicas, batch_size)
    # NOTE: ensure batch size>=1 for all replicas
    return batch_size


def set_batch_size(batch_size, batch_size_type, dynamic_batching, num_replicas,
                   df, offset):
    if batch_size_type == 'seq':
        min_xlen = df[offset:offset + 1]['xlen'].values[0]
        min_ylen = df[offset:offset + 1]['ylen'].values[0]
        _batch_size = _set_batch_size_seq(batch_size, min_xlen, min_ylen,
                                          dynamic_batching, num_replicas)
    elif batch_size_type == 'frame':
        xlens = df[offset:]['xlen'].values
        _batch_size = _set_batch_size_bin(batch_size, xlens, num_replicas)
    elif batch_size_type == 'token':
        ylens = df[offset:]['ylen'].values
        _batch_size = _set_batch_size_bin(batch_size, ylens, num_replicas)
    else:
        raise NotImplementedError(batch_size_type)
    return _batch_size


def sort_bucketing(df, batch_size, batch_size_type, dynamic_batching,
                   num_replicas=1):
    """Bucket utterances in a sorted dataframe. This is also used for evaluation.

    Args:
        batch_size (int): size of mini-batch
        batch_size_type (str): type of batch size counting
        dynamic_batching (bool): change batch size dynamically in training
        num_replicas (int): number of replicas for distributed training
    Returns:
        indices_buckets (List[List]): bucketted utterances

    """
    indices_buckets = []  # list of list
    offset = 0
    indices_rest = list(df.index)
    while True:
        _batch_size = set_batch_size(batch_size, batch_size_type, dynamic_batching,
                                     num_replicas, df, offset)

        indices = list(df[offset:offset + _batch_size].index)
        if len(indices) >= num_replicas:
            indices_buckets.append(indices)
        offset += len(indices)
        if offset >= len(df):
            break

    return indices_buckets


def shuffle_bucketing(df, batch_size, batch_size_type, dynamic_batching,
                      seed=None, num_replicas=1):
    """Bucket utterances having a similar length and shuffle them for Transformer training.

    Args:
        batch_size (int): size of mini-batch
        batch_size_type (str): type of batch size counting
        dynamic_batching (bool): change batch size dynamically in training
        seed (int): seed for randomization
        num_replicas (int): number of replicas for distributed training
    Returns:
        indices_buckets (List[List]): bucketted utterances

    """
    indices_buckets = []  # list of list
    offset = 0
    while True:
        _batch_size = set_batch_size(batch_size, batch_size_type, dynamic_batching,
                                     num_replicas, df, offset)

        indices = list(df[offset:offset + _batch_size].index)
        if len(indices) >= num_replicas:
            indices_buckets.append(indices)
        offset += len(indices)
        if offset >= len(df):
            break

    # shuffle buckets globally
    if seed is not None:
        random.seed(seed)
    random.shuffle(indices_buckets)
    return indices_buckets


def longform_bucketing(df, batch_size, max_n_frames):
    """Bucket utterances for long-form evaluation."""
    assert batch_size == 1
    indices_buckets = []  # list of list
    offset = 0
    n_frames_total = 0
    _batch_size = 0
    speaker_prev = df.loc[offset]['speaker']
    while True:
        xlen = df.loc[offset + _batch_size]['xlen']
        speaker = df.loc[offset + _batch_size]['speaker']
        if (n_frames_total + xlen > max_n_frames) or (offset + _batch_size >= len(df) - 1):
            indices = list(df[offset:offset + _batch_size + 1].index)
            indices_buckets.append(indices)
            offset += len(indices)
            n_frames_total = 0
            _batch_size = 0
        else:
            n_frames_total += xlen
            _batch_size += 1
        speaker_prev = speaker
        if offset >= len(df):
            break

    return indices_buckets


def discourse_bucketing(df, batch_size):
    """Bucket utterances by timestamp for discourse-aware training."""
    indices_buckets = []  # list of list
    session_groups = [(k, v) for k, v in df.groupby('n_utt_in_session').groups.items()]
    for n_utt, ids in session_groups:
        first_utt_ids = [i for i in ids if df['n_prev_utt'][i] == 0]
        for i in range(0, len(first_utt_ids), batch_size):
            first_utt_ids_mb = first_utt_ids[i:i + batch_size]
            for j in range(n_utt):
                indices = [k + j for k in first_utt_ids_mb]
                indices_buckets.append(indices)

    return indices_buckets
