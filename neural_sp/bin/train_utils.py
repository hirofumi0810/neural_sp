#! /usr/bin/env python
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

logger = logging.getLogger('training')


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


def set_logger(save_path, key):
    """Set logger.

    Args:
        save_path (str):
        key (str):
    Returns:
        logger ():

    """
    logger = logging.getLogger(key)
    sh = logging.StreamHandler()
    fh = logging.FileHandler(save_path)

    logger.setLevel(logging.DEBUG)
    sh.setLevel(logging.WARNING)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)s line:%(lineno)d %(levelname)s: %(message)s')
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger


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


def load_checkpoint(model, checkpoint_path, resume=False):
    """Load checkpoint.

    Args:
        model (torch.nn.Module):
        checkpoint_path (str): path to the saved model (model..epoch-*)
        epoch (int): negative values mean the offset from the last saved model
        resume (bool): if True, restore the save optimizer
    Returns:
        model (torch.nn.Module):
        checkpoints (dict):
            epoch (int): the currnet epoch
            step (int): the current step
            metric_dev_best (float): the current best performance

    """
    if not os.path.isfile(checkpoint_path):
        raise ValueError('There is no checkpoint')

    epoch = int(os.path.basename(checkpoint_path).split('-')[-1])

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        raise ValueError("No checkpoint found at %s" % checkpoint_path)

    # Restore parameters
    model.load_state_dict(checkpoint['state_dict'])

    # Restore optimizer
    if resume:
        logger.info("=> Loading checkpoint (epoch:%d): %s" % (epoch, checkpoint_path))

        if hasattr(model, 'optimizer'):
            model.optimizer.load_state_dict(checkpoint['optimizer'])

            for state in model.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda(0)
                        # state[k] = v.cuda(self.device_id)
                        # TODO (hirofumi): Fix for multi-GPU
            # NOTE: from https://github.com/pytorch/pytorch/issues/2830
        else:
            raise ValueError('Set optimizer.')
    else:
        logger.info("=> Loading checkpoint (epoch:%d): %s" % (epoch, checkpoint_path))

    return_values = {
        'optimizer': checkpoint['optimizer'],
        'epoch': epoch + 1,
        'step': checkpoint['step'] + 1,
        'metric_dev_best': checkpoint['metric_dev_best']
    }
    return model, return_values


def save_checkpoint(model, save_path, optimizer, epoch, step, metric_dev_best,
                    remove_old_checkpoints=True):
    """Save checkpoint.

    Args:
        model (torch.nn.Module):
        save_path (str): path to the directory to save a model
        optimizer (LRScheduler): optimizer
        epoch (int): currnet epoch
        step (int): current step
        metric_dev_best (float):
        remove_old_checkpoints (bool): if True, all checkpoints
            other than the best one will be deleted

    """
    model_path = os.path.join(save_path, 'model.epoch-' + str(epoch))

    # Remove old checkpoints
    if remove_old_checkpoints:
        for path in glob(os.path.join(save_path, 'model.epoch-*')):
            os.remove(path)

    # Save parameters, optimizer, step index etc.
    checkpoint = {
        "state_dict": model.module.state_dict(),
        "optimizer": optimizer.optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "metric_dev_best": metric_dev_best
    }
    torch.save(checkpoint, model_path)

    logger.info("=> Saved checkpoint (epoch:%d): %s" % (epoch, model_path))
