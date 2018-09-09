#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Load the config file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml


def load_config(config_path):
    """Load a configration yaml file.

    Args:
        config_path (str):
    Returns:
        params (dict):

    """
    with open(config_path, "r") as f:
        config = yaml.load(f)

        # Load the parent config file
        if 'parent' in config.keys():
            with open(config['parent'], "r") as fp:
                config_parent = yaml.load(fp)
            params = config_parent['param']

            # Override
            for key in config['param'].keys():
                params[key] = config['param'][key]
        else:
            params = config['param']
    return params


def save_config(config, save_path):
    """Save a configuration file as a yaml file.

    Args:
        config (dict):

    """
    with open(os.path.join(save_path, 'config.yml'), "w") as f:
        f.write(yaml.dump({'param': config}, default_flow_style=False))
