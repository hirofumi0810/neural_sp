# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Utility functions for data loader."""

import codecs
import random

random.seed(1)


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
        batch_size //= 4

    return max(1, batch_size)


def shuffle_bucketing(df, batch_size, dynamic_batching):
    indices_buckets = []  # list of list
    offset = 0
    while True:
        min_xlen = df[offset:offset + 1]['xlen'].values[0]
        min_ylen = df[offset:offset + 1]['ylen'].values[0]
        _batch_size = set_batch_size(batch_size, min_xlen, min_ylen,
                                     dynamic_batching)
        indices = list(df[offset:offset + _batch_size].index)
        indices_buckets.append(indices)
        offset += len(indices)
        if offset >= len(df):
            break

    # shuffle buckets
    random.shuffle(indices_buckets)
    return indices_buckets


def discourse_bucketing(df, batch_size):
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
