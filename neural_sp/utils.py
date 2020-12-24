# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Unility functions for general purposes."""

import os


def mkdir_join(path, *dir_name):
    """Concatenate root path and 1 or more paths, and make a new directory if the directory does not exist.
    Args:
        path (str): path to a directory
        dir_name (str): a directory name
    Returns:
        path to the new directory
    """
    if not os.path.isdir(path):
        os.mkdir(path)
    for i in range(len(dir_name)):
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
