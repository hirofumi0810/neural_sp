# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Utility functions for training."""

import codecs
import functools
import logging
import numpy as np
import os
import time
import torch
import yaml

logger = logging.getLogger(__name__)


def compute_subsampling_factor(args):
    """Register subsample factors to args.

        Args:
            args (Namespace):
        Returns:
            args (Namespace):

    """
    if args.resume:
        return args

    args.subsample_factor = 1
    args.subsample_factor_sub1 = 1
    args.subsample_factor_sub2 = 1
    if 'conv' in args.enc_type and args.conv_poolings:
        for p in args.conv_poolings.split('_'):
            args.subsample_factor *= int(p.split(',')[0].replace('(', ''))
    subsample = [int(s) for s in args.subsample.split('_')]
    if args.train_set_sub1:
        args.subsample_factor_sub1 = args.subsample_factor * \
            int(np.prod(subsample[:args.enc_n_layers_sub1]))
    if args.train_set_sub2:
        args.subsample_factor_sub2 = args.subsample_factor * \
            int(np.prod(subsample[:args.enc_n_layers_sub2]))
    args.subsample_factor *= int(np.prod(subsample))

    return args


def measure_time(func):
    @functools.wraps(func)
    def _measure_time(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        elapse = time.time() - start
        print("Takes {} seconds.".format(elapse))
    return _measure_time


def load_config(config_path):
    """Load a configuration yaml file.

    Args:
        config_path (str):
    Returns:
        params (dict):

    """
    if not os.path.isfile(config_path):
        raise ValueError("No configuration found at %s" % config_path)

    with codecs.open(config_path, "r", encoding='utf-8') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    params = conf['param']
    return params


def save_config(conf, save_path):
    """Save a configuration file as a yaml file.

    Args:
        conf (dict):

    """
    with codecs.open(os.path.join(save_path), "w", encoding='utf-8') as f:
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
    """Change directory name to avoid name overlapping.

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


def load_checkpoint(checkpoint_path, model=None, scheduler=None, amp=None):
    """Load checkpoint.

    Args:
        checkpoint_path (str): path to the saved model (model.epoch-*)
        model (torch.nn.Module):
        scheduler (LRScheduler): optimizer wrapped by LRScheduler class
        amp ():
    Returns:
        topk_list (List): (epoch, metric)

    """
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        raise ValueError("No checkpoint found at %s" % checkpoint_path)

    # Restore parameters
    if 'avg' not in checkpoint_path:
        epoch = int(os.path.basename(checkpoint_path).split('-')[-1]) - 1
        logger.info("=> Loading checkpoint (epoch:%d): %s" % (epoch + 1, checkpoint_path))
    else:
        logger.info("=> Loading checkpoint: %s" % checkpoint_path)
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])

    # Restore scheduler/optimizer
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['optimizer_state_dict'], model.use_cuda)
        # NOTE: fix this later
        scheduler.optimizer.param_groups[0]['params'] = []
        for param_group in list(model.parameters()):
            scheduler.optimizer.param_groups[0]['params'].append(param_group)
    else:
        logger.warning('Scheduler/Optimizer is not loaded.')

    # Restore apex
    if amp is not None:
        amp.load_state_dict(checkpoint['amp_state_dict'])
    else:
        logger.warning('amp is not loaded.')

    if 'optimizer_state_dict' in checkpoint.keys() and 'topk_list' in checkpoint['optimizer_state_dict'].keys():
        topk_list = checkpoint['optimizer_state_dict']['topk_list']
    else:
        topk_list = []
    return topk_list
