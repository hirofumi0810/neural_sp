#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load each encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.layers.encoders.rnn import RNN_Encoder
# from models.layers.encoders import vgg
# from models.layers.encoders import resnet
# from models.layers.encoders import hierarchical


ENCODERS = {
    "lstm": RNN_Encoder,
    "gru": RNN_Encoder,
    "rnn": RNN_Encoder
}


def load(encoder_type):
    """Load an encoder.
    Args:
        encoder_type (string): name of the encoder in the key of ENCODERS
    Returns:
        model (nn.Module): An encoder class
    """
    if encoder_type not in ENCODERS:
        raise ValueError(
            "encoder_type should be one of [%s], you provided %s." %
            (", ".join(ENCODERS), encoder_type))
    return ENCODERS[encoder_type]
