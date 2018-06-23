#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging


def set_logger(save_path, key):
    """Set logger.
    Args:
        save_path (string):
    Returns:
        logger ():
    """
    logger = logging.getLogger(key)
    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    fh = logging.FileHandler(save_path)
    fh.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s line:%(lineno)d %(levelname)s:  %(message)s')
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger
