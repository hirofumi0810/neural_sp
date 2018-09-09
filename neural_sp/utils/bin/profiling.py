#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Show the results of profiling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pstats

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
args = parser.parse_args()


def main():

    stats = pstats.Stats(os.path.join(args.model_path, 'train.profile'))
    stats.sort_stats('time').print_stats(30)


if __name__ == '__main__':
    main()
