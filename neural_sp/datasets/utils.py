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


def set_batch_size(batch_size, min_xlen, min_ylen, dynamic_batching):
    if not dynamic_batching:
        return batch_size

    if min_xlen <= 800:
        pass
    elif min_xlen <= 1600 or 80 < min_ylen <= 100:
        batch_size //= 2
    else:
        batch_size //= 8

    return max(1, batch_size)


def sort_bucketing(df, batch_size, dynamic_batching, num_replicas=1):
    """Bucket utterances in a sorted dataframe. This is also used for evaluation."""
    indices_buckets = []  # list of list
    offset = 0
    indices_rest = list(df.index)
    while True:
        min_xlen = df[offset:offset + 1]['xlen'].values[0]
        min_ylen = df[offset:offset + 1]['ylen'].values[0]
        _batch_size = set_batch_size(batch_size, min_xlen, min_ylen, dynamic_batching)
        _batch_size = max(num_replicas, _batch_size)
        # NOTE: ensure batch size>=1 for all replicas
        indices = list(df[offset:offset + _batch_size].index)
        indices_buckets.append(indices)
        offset += len(indices)
        if offset >= len(df):
            break

    return indices_buckets


def shuffle_bucketing(df, batch_size, dynamic_batching, seed=None, num_replicas=1):
    """Bucket utterances having a similar length and shuffle them for Transformer training."""
    indices_buckets = []  # list of list
    offset = 0
    while True:
        min_xlen = df[offset:offset + 1]['xlen'].values[0]
        min_ylen = df[offset:offset + 1]['ylen'].values[0]
        _batch_size = set_batch_size(batch_size, min_xlen, min_ylen, dynamic_batching)
        _batch_size = max(num_replicas, _batch_size)
        # NOTE: ensure batch size>=1 for all replicas
        indices = list(df[offset:offset + _batch_size].index)
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
