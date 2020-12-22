# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Softplus function."""

import torch


def softplus(x):
    if hasattr(torch.nn.functional, 'softplus'):
        return torch.nn.functional.softplus(x.float()).type_as(x)
    else:
        raise NotImplementedError
