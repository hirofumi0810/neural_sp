#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging


def set_logger(save_path):

    logger = logging.getLogger('training')
    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    fh = logging.FileHandler(os.path.join(save_path, 'train.log'))
    fh.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s line:%(lineno)d %(levelname)s:  %(message)s')
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger
