#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Add Gaussian noise to input features."""

import torch


def add_gaussian_noise(xs):
    noise = torch.normal(torch.zeros(xs.shape[-1]), 0.075)
    if xs.is_cuda:
        noise = noise.cuda()
    xs.data += noise
    return xs
