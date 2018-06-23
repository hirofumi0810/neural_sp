#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Parallel computing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import multiprocessing as mp


def make_parallel(func, args, core=mp.cpu_count() - 1):
    """
    Args:
        func (function):
        args (tuple or dict): arguments for func
    Returns:
        result_tuple (tuple): tuple of returns
    """
    try:
        p = mp.Pool(core)

        result_tuple = p.map(func, args)
        # result_tuple = p.map_async(func, args).get(9999999)
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
