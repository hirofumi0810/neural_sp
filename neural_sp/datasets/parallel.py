#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Parallel processing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing as mp


def multiprocess(func, args, core=4):
    """Wrapper for parallel processing.

    Args:
        func (function):
        args (tuple or dict): arguments for func
        core (int):
    Returns:
        result_tuple (tuple): tuple of returns

    """
    try:
        p = mp.Pool(core)

        # result_tuple = p.map(func, args)
        result_tuple = p.map_async(func, args).get(9999999)
        # NOTE: for KeyboardInterrupt

        # Clean up
        p.close()
        p.terminate()
        p.join()
    except KeyboardInterrupt:
        p.close()
        p.terminate()
        p.join()
        raise KeyboardInterrupt

    return result_tuple
