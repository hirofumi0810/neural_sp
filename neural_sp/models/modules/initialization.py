#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Parameter initialization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import torch.nn as nn

logger = logging.getLogger(__name__)


def init_with_normal_dist(n, p, std):
    if 'norm' in n and 'weight' in n:
        assert p.dim() == 1
        nn.init.normal_(p, 1.0, std)  # layer normalization
        logger.info('Initialize %s with %s / (1.0, %.3f)' % (n, 'normal', std))
    elif p.dim() == 1:
        nn.init.constant_(p, 0.)  # bias
        logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.))
    elif p.dim() == 2:
        nn.init.normal_(p, mean=0, std=std)
        logger.info('Initialize %s with %s / (0.0, %.3f)' % (n, 'normal', std))
    else:
        raise ValueError(n)


def init_with_xavier_dist(n, p):
    # https://github.com/kimiyoung/transformer-xl/blob/44781ed21dbaec88b280f74d9ae2877f52b492a5/pytorch/train.py
    if p.dim() == 1:
        nn.init.constant_(p, 0.)  # bias
        logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.))
    elif p.dim() in [2, 3]:
        nn.init.xavier_uniform_(p)  # linear layer
        logger.info('Initialize %s with %s' % (n, 'xavier_uniform'))
    else:
        raise ValueError(n)


def init_with_lecun(n, p, param_init):
    if p.dim() == 1:
        nn.init.constant_(p, 0.)  # bias
        logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.))
    elif p.dim() == 2:
        fan_in = p.size(1)
        nn.init.normal_(p, mean=0., std=1. / math.sqrt(fan_in))  # linear weight
        logger.info('Initialize %s with %s / %.3f' % (n, 'lecun', param_init))
    elif p.dim() == 3:
        fan_in = p.size(1) * p[0][0].numel()
        nn.init.normal_(p, mean=0., std=1. / math.sqrt(fan_in))  # 1d conv weight
        logger.info('Initialize %s with %s / %.3f' % (n, 'lecun', param_init))
    elif p.dim() == 4:
        fan_in = p.size(1) * p[0][0].numel()
        nn.init.normal_(p, mean=0., std=1. / math.sqrt(fan_in))  # 2d conv weight
        logger.info('Initialize %s with %s / %.3f' % (n, 'lecun', param_init))
    else:
        raise ValueError(n)


# def init_embedding(embed):
#     nn.init.normal_(embed.weight, mean=0., std=d_model**-0.5)
#     nn.init.constant_(embed.weight[pad], 0.)
