#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate a model by accuracy."""

import logging
from tqdm import tqdm


logger = logging.getLogger(__name__)


def eval_accuracy(models, dataset, batch_size=1, progressbar=False):
    """Evaluate a Seq2seq by teacher-forcing accuracy.

    Args:
        models (list): models to evaluate
        dataset (Dataset): evaluation dataset
        batch_size (int): batch size
        progressbar (bool): if True, visualize the progressbar
    Returns:
        accuracy (float): Average accuracy

    """
    total_acc = 0
    n_tokens = 0

    # Reset data counter
    dataset.reset()

    if progressbar:
        pbar = tqdm(total=len(dataset))

    while True:
        batch, is_new_epoch = dataset.next(batch_size)
        bs = len(batch['ys'])
        _, observation = models[0](batch, task='all', is_eval=True)
        n_tokens_b = sum([len(y) for y in batch['ys']])
        total_acc += observation['acc.att'] * n_tokens_b
        n_tokens += n_tokens_b

        if progressbar:
            pbar.update(bs)

        if is_new_epoch:
            break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset.reset()

    accuracy = total_acc / n_tokens

    logger.debug('Accuracy (%s): %.2f %%' % (dataset.set, accuracy))

    return accuracy
