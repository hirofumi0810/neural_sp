#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from models.base import ModelBase
from models.layers.encoders.load_encoder import load


class CTC(ModelBase):
    """The Connectionist Temporal Classification model.
    Args:
        input_size (int): the dimensions of input vectors
        num_units (int): the number of units in each layer
        num_layers (int): the number of layers
        num_classes (int): the number of classes of target labels
            (except for a blank label)
        encoder_type (string): lstm or gru or rnn
        bidirectional (bool): if True create a bidirectional model
        use_peephole (bool, optional): if True, use peephole
        splice (int, optional): frames to splice. Default is 1 frame.
        parameter_init (float, optional): Range of uniform distribution to
            initialize weight parameters
        clip_grad (float, optional): Range of gradient clipping (> 0)
        clip_activation (float, optional): Range of activation clipping (> 0)
        num_proj (int, optional): the number of nodes in recurrent projection layer
        weight_decay (float, optional): Regularization parameter for weight decay
        bottleneck_dim (int, optional): the dimensions of the bottleneck layer
    """

    def __init__(self,
                 input_size,
                 num_units,
                 num_layers,
                 num_classes,
                 encoder_type,
                 bidirectional,
                 use_peephole=True,
                 splice=1,
                 parameter_init=0.1,
                 clip_grad=None,
                 clip_activation=None,
                 num_proj=None,
                 weight_decay=0.0,
                 bottleneck_dim=None):

        super(ModelBase, self).__init__()
        # TODO: weight_decay とかはここで入力した方がいい？

        self.input_size = input_size
        self.num_units = num_units
        self.num_layers = num_layers
        self.num_classes = num_classes + 1  # add blank class
        self.encoder_type = encoder_type
        self.bidirectional = bidirectional
        # self.use_peephole = use_peephole
        self.splice = splice
        self.parameter_init = parameter_init
        self.clip_grad = clip_grad
        self.clip_activation = clip_activation
        # self.num_proj = num_proj
        self.weight_decay = weight_decay
        # self.bottleneck_dim = bottleneck_dim

        # Load encoder
        encoder = load(encoder_type=encoder_type)
        if encoder_type in ['lstm', 'gru', 'rnn']:
            self.encoder = encoder(input_size=input_size,
                                   num_units=num_units,
                                   num_layers=num_layers,
                                   num_classes=self.num_classes,
                                   rnn_type=encoder_type,
                                   bidirectional=bidirectional,
                                   parameter_init=parameter_init)

        elif encoder_type in ['vgg_lstm', 'conv_lstm']:
            raise ValueError

        else:
            # resnet or vgg, and so on.
            raise NotImplementedError

    def forward(self, inputs):
        """
        Args:
            inputs ():
        Returns:
            logits ():
        """
        logits, final_state = self.encoder(inputs)
        return logits
