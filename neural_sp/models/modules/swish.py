# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Swish activation.
   See details in https://arxiv.org/abs/1710.05941."""

import torch


class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
