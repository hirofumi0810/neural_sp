# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Unility functions for general purposes."""

from pathlib import Path


def mkdir_join(path, *dir_name):
    """Concatenate root path and 1 or more paths, and make a new directory if the directory does not exist.
    Args:
        path (str): path to a directory
        dir_name (str): a directory name
    Returns:
        path to the new directory
    """
    p = Path(path)
    if not p.is_dir():
        p.mkdir()
    for i in range(len(dir_name)):
        # dir
        if i < len(dir_name) - 1:
            p = p.joinpath(dir_name[i])
            if not p.is_dir():
                p.mkdir()
        elif '.' not in dir_name[i]:
            p = p.joinpath(dir_name[i])
            if not p.is_dir():
                p.mkdir()
        # file
        else:
            p = p.joinpath(dir_name[i])
    return str(p.absolute())
