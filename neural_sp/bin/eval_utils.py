#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Utility functions for evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import torch

logger = logging.getLogger(__name__)


def average_checkpoints(model, best_model_path, epoch, n_average):
    if n_average == 1:
        return model

    n_models = 1
    state_dict_ave = model.state_dict()
    for i in range(epoch - 1, 0, -1):
        if n_models == n_average:
            break
        checkpoint_path = best_model_path.replace('-' + str(epoch), '-' + str(i))
        if os.path.isfile(checkpoint_path):
            logger.info("=> Loading checkpoint (epoch:%d): %s" % (i, checkpoint_path))
            params = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)['state_dict']
            for k, v in params.items():
                state_dict_ave[k] += v
            n_models += 1

    # take an average
    logger.info('Take average for %d models' % n_models)
    for k, v in state_dict_ave.items():
        state_dict_ave[k] /= n_models
    model.load_state_dict(state_dict_ave)

    avrage_checkpoint_path = best_model_path.replace('-' + str(epoch), '-avg' + str(n_average))
    torch.save(model, avrage_checkpoint_path)

    return model
