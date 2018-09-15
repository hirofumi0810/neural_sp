#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"General utility functions."

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import logging
import os
import six
import time


def mkdir_join(path, *dir_name):
    """Concatenate root path and 1 or more paths, and make a new direcory if the direcory does not exist.

    Args:
        path (str): path to a diretcory
        dir_name (str): a direcory name
    Returns:
        path to the new directory

    """
    if not os.path.isdir(path):
        os.mkdir(path)
    for i in six.moves.range(len(dir_name)):
        if '.' not in dir_name[i]:
            path = os.path.join(path, dir_name[i])
            if not os.path.isdir(path):
                os.mkdir(path)
        else:
            path = os.path.join(path, dir_name[i])
    return path


def measure_time(func):
    @functools.wraps(func)
    def _measure_time(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        elapse = time.time() - start
        print("Takes {} seconds.".format(elapse))
    return _measure_time


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
    sh.setLevel(logging.WARNING)
    fh = logging.FileHandler(save_path)
    fh.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s line:%(lineno)d %(levelname)s:  %(message)s')
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger
