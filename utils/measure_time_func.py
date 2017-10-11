#! /usr/bin/env python
# -*- coding: utf-8 -*

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import time


def measure_time(func):
    @functools.wraps(func)
    def _measure_time(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        elapse = time.time() - start
        print("Takes {} seconds.".format(elapse))
    return _measure_time
