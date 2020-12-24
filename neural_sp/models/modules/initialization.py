# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Parameter initialization."""

import logging
import math
import torch.nn as nn

logger = logging.getLogger(__name__)


def init_like_transformer_xl(n, p, std):
    """Initialize like TransformerXL.
        See https://github.com/kimiyoung/transformer-xl/blob/44781ed21dbaec88b280f74d9ae2877f52b492a5/pytorch/train.py

    Args:
        n (str): parameter name
        p (Tensor): parameter
        str (float): standard deviation

    """
    if 'norm' in n and 'weight' in n:
        assert p.dim() == 1
        nn.init.normal_(p, mean=1.0, std=std)  # layer normalization
        logger.info('Initialize %s with %s / (1.0, %.3f)' % (n, 'normal', std))
    elif p.dim() == 1:
        nn.init.constant_(p, 0.)  # bias
        logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.))
    elif p.dim() == 2:
        nn.init.normal_(p, mean=0, std=std)
        logger.info('Initialize %s with %s / (0.0, %.3f)' % (n, 'normal', std))
    else:
        raise ValueError(n)


def init_with_xavier_uniform(n, p):
    """Initialize with Xavier uniform distribution.

    Args:
        n (str): parameter name
        p (Tensor): parameter

    """
    if p.dim() == 1:
        nn.init.constant_(p, 0.)  # bias
        logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.))
    elif p.dim() in [2, 3, 4]:
        nn.init.xavier_uniform_(p)  # linear layer
        logger.info('Initialize %s with %s' % (n, 'xavier_uniform'))
    else:
        raise ValueError(n)


def init_with_lecun_normal(n, p, param_init):
    """Initialize with Lecun style.

    Args:
        n (str): parameter name
        p (Tensor): parameter
        param_init (float):

    """
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


def init_with_uniform(n, p, param_init):
    """Initialize with uniform distribution.

    Args:
        n (str): parameter name
        p (Tensor): parameter
        param_init (float):

    """
    if p.dim() == 1:
        nn.init.constant_(p, 0.)  # bias
        logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.))
    elif p.dim() in [2, 3, 4]:
        nn.init.uniform_(p, a=-param_init, b=param_init)
        logger.info('Initialize %s with %s / %.3f' % (n, 'uniform', param_init))
    else:
        raise ValueError(n)
