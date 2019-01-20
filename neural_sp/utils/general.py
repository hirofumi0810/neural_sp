#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"General utility functions."

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
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
        # dir
        if i < len(dir_name) - 1:
            path = os.path.join(path, dir_name[i])
            if not os.path.isdir(path):
                os.mkdir(path)
        elif '.' not in dir_name[i]:
            path = os.path.join(path, dir_name[i])
            if not os.path.isdir(path):
                os.mkdir(path)
        # file
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
