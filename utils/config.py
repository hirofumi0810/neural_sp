#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load the config file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import yaml
import shutil


def load_config(config_path, is_eval=False):
    """Load configure file.
    Args:
        config_path (string):
        is_eval (bool, option):
    Returns:
        params (dict):
    """
    with open(config_path, "r") as f:
        config = yaml.load(f)

        # Load the parent config file
        if 'parent' in config.keys():
            if not is_eval:
                with open(config['parent'], "r") as fp:
                    config_parent = yaml.load(fp)

            else:
                with open(config_path.replace('config.yml', 'config_parent.yml'), "r") as fp:
                    config_parent = yaml.load(fp)

            params = config_parent['param']

            # Override
            for key in config['param'].keys():
                params[key] = config['param'][key]

        else:
            params = config['param']

    return params


def save_config(config_path, save_path):
    """Save configure file.
    Args:
        config_path (string):
    Returns:
        save_path (string):
    """

    shutil.copyfile(config_path, join(save_path, 'config.yml'))

    # Save the parent config file
    with open(config_path, "r") as f:
        config = yaml.load(f)

        # Load the parent config file
        if 'parent' in config.keys():
            config_parent_path = config['parent']
            shutil.copyfile(config_parent_path, join(
                save_path, 'config_parent.yml'))
