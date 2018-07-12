#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method of RNNLMs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from tqdm import tqdm
import numpy as np


def eval_ppl(models, dataset, bptt, progressbar=False):
    """Evaluate a RNNLM by perprexity.
    Args:
        models (list): the models to evaluate
        dataset: An instance of a `Dataset' class
        progressbar (bool): if True, visualize the progressbar
    Returns:
        ppl (float): Perplexity
    """
    # Reset data counter
    dataset.reset()

    model = models[0]
    # TODO: fix this

    # Change to the evaluation mode
    model.eval()

    loss = 0
    num_utt = 0
    if progressbar:
        pbar = tqdm(total=len(dataset))  # TODO: fix this
    while True:
        batch, is_new_epoch = dataset.next(batch_size=1)

        ys = np.array(batch['ys'])
        batch_size = len(batch['input_names'])

        # Truncate
        ys = ys.reshape((batch_size, -1))
        # ys: `[B, T]`

        num_step = ys.shape[1] // bptt
        offset = 0
        for i in range(num_step):
            ys_bptt = ys[:, offset: offset + bptt]
            offset += bptt
            loss += model(ys_bptt, is_eval=True)[0].data[0]
        num_utt += 1

        if progressbar:
            pbar.update(len(batch['ys']))

        if is_new_epoch:
            break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset.reset()

    ppl = math.exp(loss / num_utt)

    return ppl
