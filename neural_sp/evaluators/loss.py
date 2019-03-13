#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate a model by loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm


def eval_loss(models, dataset, recog_params, progressbar=False):
    """Evaluate a model by loss.

    Args:
        models (list): the models to evaluate
        dataset: An instance of a `Dataset' class
        recog_params (dict):
        progressbar (bool): if True, visualize the progressbar
    Returns:
        loss_avg (float): average loss

    """
    # Reset data counter
    dataset.reset()

    model = models[0]
    # TODO(hirofumi): ensemble decoding

    total_loss = 0
    if progressbar:
        pbar = tqdm(total=len(dataset))
    while True:
        batch, is_new_ep = dataset.next(recog_params['recog_batch_size'])
        bs = len(batch['utt_ids'])

        assert not dataset.is_test
        loss, loss_acc_fwd, loss_acc_bwd, loss_acc_sub = model(
            batch['xs'], batch['ys'], batch['ys_sub'], is_eval=True)

        total_loss += loss.item() * bs

        if progressbar:
            pbar.update(bs)

        if is_new_ep:
            break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset.reset()

    loss_avg = total_loss / len(dataset)

    return loss_avg
