#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Utility functions for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from glob import glob
import logging
import os
import time
import torch
import yaml

logger = logging.getLogger(__name__)


def measure_time(func):
    @functools.wraps(func)
    def _measure_time(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        elapse = time.time() - start
        print("Takes {} seconds.".format(elapse))
    return _measure_time


def load_config(config_path):
    """Load a configration yaml file.

    Args:
        config_path (str):
    Returns:
        params (dict):

    """
    with open(config_path, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    params = conf['param']
    return params


def save_config(conf, save_path):
    """Save a configuration file as a yaml file.

    Args:
        conf (dict):

    """
    with open(os.path.join(save_path), "w") as f:
        f.write(yaml.dump({'param': conf}, default_flow_style=False))


def set_logger(save_path, stdout=False):
    """Set logger.

    Args:
        save_path (str): path to save a log file
        stdout (bool):

    """
    format = '%(asctime)s %(name)s line:%(lineno)d %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG if stdout else logging.INFO,
                        format=format,
                        filename=save_path if not stdout else None)


def set_save_path(save_path):
    """Change directory name to avoid name ovarlapping.

    Args:
        save_path (str):
    Returns:
        save_path_new (str):

    """
    # Reset model directory
    model_idx = 0
    save_path_new = save_path
    while True:
        if os.path.isfile(os.path.join(save_path_new, 'conf.yml')):
            # Training of the first model have not been finished yet
            model_idx += 1
            save_path_new = save_path + '_' + str(model_idx)
        else:
            break
    if not os.path.isdir(save_path_new):
        os.mkdir(save_path_new)
    return save_path_new


def load_checkpoint(model, checkpoint_path, optimizer=None):
    """Load checkpoint.

    Args:
        model (torch.nn.Module):
        checkpoint_path (str): path to the saved model (model..epoch-*)
        optimizer (LRScheduler): optimizer wrapped by LRScheduler class

    """
    if not os.path.isfile(checkpoint_path):
        raise ValueError('There is no checkpoint')

    epoch = int(os.path.basename(checkpoint_path).split('-')[-1]) - 1

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        raise ValueError("No checkpoint found at %s" % checkpoint_path)

    # Restore parameters
    logger.info("=> Loading checkpoint (epoch:%d): %s" % (epoch + 1, checkpoint_path))
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except KeyError:
        model.load_state_dict(checkpoint['state_dict'])
        checkpoint['model_state_dict'] = checkpoint['state_dict']
        checkpoint['optimizer_state_dict'] = checkpoint['optimizer']
        del checkpoint['state_dict']
        del checkpoint['optimizer']
        torch.save(checkpoint, checkpoint_path + '.tmp')

    # Restore optimizer
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # NOTE: fix this later
        optimizer.optimizer.param_groups[0]['params'] = []
        for param_group in list(model.parameters()):
            optimizer.optimizer.param_groups[0]['params'].append(param_group)
    else:
        logger.warning('Optimizer is not loaded.')
