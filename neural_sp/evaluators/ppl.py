#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate a RNNLM by perplexity."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
from tqdm import tqdm

from neural_sp.models.lm.gated_convlm import GatedConvLM
from neural_sp.models.lm.rnnlm import RNNLM

logger = logging.getLogger("decoding").getChild('ppl')


def eval_ppl(models, dataset, batch_size=1, bptt=-1,
             recog_params=None, n_caches=0, progressbar=False):
    """Evaluate a Seq2seq or RNNLM by perprexity and loss.

    Args:
        models (list): the models to evaluate
        dataset: An instance of a `Dataset' class
        batch_size (int):
        bptt (int):
        recog_params (dict):
        n_caches (int):
        progressbar (bool): if True, visualize the progressbar
    Returns:
        ppl (float): Average perplexity
        loss (float): Average loss

    """
    # Reset data counter
    dataset.reset()

    model = models[0]
    is_lm = False
    skip_thought = False
    if isinstance(model, RNNLM) or isinstance(model, GatedConvLM):
        is_lm = True
    elif 'skip' in model.enc_type:
        skip_thought = True

    # Change to the evaluation mode
    model.eval()

    total_loss = 0
    n_tokens = 0
    hidden = None  # for RNNLM
    if progressbar:
        pbar = tqdm(total=len(dataset))
    while True:
        if is_lm:
            ys, is_new_epoch = dataset.next(batch_size)
            bs, time = ys.shape[:2]
            if n_caches > 0:
                assert isinstance(model, RNNLM)
                # NOTE: cache is not supported for GatedConvLM now
                for t in range(time - 1):
                    loss, hidden = model(ys[:, t:t + 2], hidden, is_eval=True, n_caches=n_caches)[:2]
                    total_loss += loss.item() * bs
                    n_tokens += bs

                    if progressbar:
                        pbar.update(bs)
            else:
                loss, hidden = model(ys, hidden, is_eval=True)[:2]
                total_loss += loss.item() * bs * (time - 1)
                n_tokens += bs * (time - 1)

                if progressbar:
                    pbar.update(bs * (time - 1))
        else:
            batch, is_new_epoch = dataset.next(recog_params['recog_batch_size'])
            if skip_thought:
                loss, _ = model(batch['ys'],
                                ys_prev=batch['ys_prev'],
                                ys_next=batch['ys_next'],
                                is_eval=True)
            else:
                loss, _ = model(batch, task='all', is_eval=True)
            total_loss += loss.item()
            del loss
            n_tokens += sum([len(y) for y in batch['ys']])

            if progressbar:
                pbar.update(1)

        if is_new_epoch:
            break

    if progressbar:
        pbar.close()

    # Reset data counters
    dataset.reset()

    avg_loss = total_loss / n_tokens
    ppl = np.exp(avg_loss)

    logger.info('PPL (%s): %.2f %%' % (dataset.set, ppl))
    logger.info('Loss (%s): %.2f %%' % (dataset.set, avg_loss))

    return ppl, avg_loss
