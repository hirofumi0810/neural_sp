#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import join, isdir


def mkdir(path_to_dir):
    """Make a new directory if the directory does not exist.
    Args:
        path_to_dir (string): path to a directory
    Returns:
        path (string): path to the new directory
    """
    if path_to_dir is not None and (not isdir(path_to_dir)):
        os.mkdir(path_to_dir)
    return path_to_dir


def mkdir_join(path_to_dir, *dir_name):
    """Concatenate root path and 1 or more paths, and make a new direcory if
    the direcory does not exist.
    Args:
        path_to_dir (string): path to a diretcory
        dir_name (string): a direcory name
    Returns:
        path to the new directory
    """
    if path_to_dir is None:
        return path_to_dir
    for i in range(len(dir_name)):
        if '.' not in dir_name[i]:
            path_to_dir = mkdir(join(path_to_dir, dir_name[i]))
        else:
            path_to_dir = join(path_to_dir, dir_name[i])
    return path_to_dir
