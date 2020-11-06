# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Add Gaussian noise to input features."""

import torch


def add_input_noise(xs, std):
    noise = torch.normal(xs.new_zeros(xs.shape[-1]), std)
    xs.data += noise
    return xs
