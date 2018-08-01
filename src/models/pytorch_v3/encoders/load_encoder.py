#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load each encoder (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.models.pytorch_v3.encoders.cnn import CNNEncoder
from src.models.pytorch_v3.encoders.rnn import RNNEncoder
# from models.pytorch_v3.encoders.resnet import ResNetEncoder

ENCODERS = {
    "lstm": RNNEncoder,
    "gru": RNNEncoder,
    "rnn": RNNEncoder,
    "cnn": CNNEncoder,
    # "resnet": ResNetEncoder,
}


def load(enc_type):
    """Load an encoder.

    Args:
        enc_type (string): name of the encoder in the key of ENCODERS
    Returns:
        model (nn.Module): An encoder class

    """
    if enc_type not in ENCODERS:
        raise TypeError(
            "enc_type should be one of [%s], you provided %s." %
            (", ".join(ENCODERS), enc_type))
    return ENCODERS[enc_type]
