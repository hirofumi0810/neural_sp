#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Utility funcitons for beam search decoding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import logging
# import math
# import numpy as np
# import os
# import random
# import shutil
# import torch
# import torch.nn as nn


def remove_complete_hyp(hyps_sorted, end_hyps, beam_width, eos, prune=True):
    new_hyps = []
    is_finish = False
    for hyp in hyps_sorted:
        if len(hyp['hyp']) > 1 and hyp['hyp'][-1] == eos:
            end_hyps += [hyp]
        else:
            new_hyps += [hyp]
    if len(end_hyps) >= beam_width:
        if prune:
            end_hyps = end_hyps[:beam_width]
        is_finish = True
    return new_hyps, end_hyps, is_finish


def add_ctc_score():
    raise NotImplementedError


def add_lm_score():
    raise NotImplementedError
