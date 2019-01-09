#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate a RNNLM by perplexity."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
from tqdm import tqdm


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

    # Change to the evaluation mode
    model.eval()

    total_loss = 0
    num_tokens = 0
    if progressbar:
        pbar = tqdm(total=len(dataset))
    while True:
        ys, is_new_epoch = dataset.next()
        batch_size = len(ys)

        for t in range(len(ys[0]) - 1):
            total_loss += model(ys[:][t:t + 2], is_eval=True)[0].item() * batch_size
            num_tokens += batch_size

        if progressbar:
            pbar.update(np.sum([len(y) for y in ys]))

        if is_new_epoch:
            break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset.reset()

    ppl = math.exp(total_loss / num_tokens)

    return ppl
