# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Utility functions for evaluation."""

import argparse
import logging
import os
import torch

from neural_sp.bin.train_utils import (
    load_checkpoint,
    load_config
)
from neural_sp.models.lm.build import build_lm


logger = logging.getLogger(__name__)


def load_lm(lm_path, mem_len=0):
    conf_lm = load_config(os.path.join(os.path.dirname(lm_path), 'conf.yml'))
    args_lm = argparse.Namespace()
    for k, v in conf_lm.items():
        setattr(args_lm, k, v)
    args_lm.recog_mem_len = mem_len
    lm = build_lm(args_lm)
    load_checkpoint(lm_path, lm)
    lm.backward = args_lm.backward
    return lm


def average_checkpoints(model, best_model_path, n_average, topk_list=[]):
    if n_average == 1:
        return model

    if 'avg' in best_model_path:
        checkpoint_avg = torch.load(best_model_path, map_location='cpu')
        model.load_state_dict(checkpoint_avg['model_state_dict'])
        return model

    n_models = 0
    checkpoint_avg = {'model_state_dict': None}
    if len(topk_list) == 0:
        epoch = int(float(best_model_path.split('model.epoch-')[1]) * 10) / 10
        score = None
        if epoch >= 1:
            epoch = int(epoch)
            topk_list = [(i, score) for i in range(epoch, max(0, epoch - n_average - 1), -1)]
        else:
            topk_list = [(epoch, score)]
    for ep, _ in topk_list:
        if n_models == n_average:
            break
        checkpoint_path = best_model_path.split('model.epoch-')[0] + 'model.epoch-' + str(ep)
        if os.path.isfile(checkpoint_path):
            logger.info("=> Loading checkpoint (epoch:%d): %s" % (ep, checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if checkpoint_avg['model_state_dict'] is None:
                # first checkpoint
                checkpoint_avg['model_state_dict'] = checkpoint['model_state_dict']
                n_models += 1
                continue
            for k, v in checkpoint['model_state_dict'].items():
                checkpoint_avg['model_state_dict'][k] += v
            n_models += 1

    # take an average
    logger.info('Take average for %d models' % n_models)
    for k, v in checkpoint_avg['model_state_dict'].items():
        checkpoint_avg['model_state_dict'][k] /= n_models
    model.load_state_dict(checkpoint_avg['model_state_dict'])

    # save as a new checkpoint
    checkpoint_avg_path = best_model_path.split('model.epoch-')[0] + 'model-avg' + str(n_average)
    if os.path.isfile(checkpoint_avg_path):
        os.remove(checkpoint_avg_path)
    torch.save(checkpoint_avg, checkpoint_avg_path)
