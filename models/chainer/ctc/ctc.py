#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from chainer import functions as F
# from chainer.functions import connectionist_temporal_classification as ctc_loss

from models.chainer.ctc.ctc_loss_from_chainer import connectionist_temporal_classification as ctc_loss
from models.chainer.base import ModelBase
from models.chainer.encoders.load_encoder import load


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
        parameter_init (float, optional): the range of uniform distribution to
            initialize weight parameters
        clip_grad (float, optional): the range of gradient clipping (> 0)
        clip_activation (float, optional): range of activation clipping (> 0)
        num_proj (int, optional): the number of nodes in recurrent projection layer
        weight_decay (float, optional): regularization parameter for weight decay
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

        #  error echeck
        assert clip_grad is None or clip_grad > 0
        assert clip_activation is None or clip_activation > 0
        assert weight_decay >= 0
        # TODO: Add None

        # Load encoder
        encoder = load(encoder_type=encoder_type)
        if encoder_type in ['lstm', 'gru', 'rnn_tanh', 'rnn_relu']:
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

    def __call__(self, inputs):
        """
        Args:
            inputs (list): list of tensors of size `[T, input_size]`
        Returns:
            logits (list): list of tensors of size `[T, num_classes + 1]`
        """
        logits, final_state = self.encoder(inputs)
        return logits

    def compute_loss(self, logits, labels, blank_index,
                     inputs_seq_len, labels_seq_len):
        """Compute loss. However, do not do back-propagation yet.
        Args:
            logits (list): list of tensors of size `[T, num_classes + 1]`
            labels (chainer.Variable): A tensor of size `[B, num_classes]`
            blank_index (int): the index of tha blank class
            inputs_seq_len (chianer.Variable): A tensor of size `[B,]`
            labels_seq_len (chainer.Variable): A tensor of size `[B,]`
        Returns:
            loss (chainer.Variable):
        """
        # Pad variable length logits
        logits = F.pad_sequence(logits, padding=-1)

        # Convert to time-major
        # logits = F.swapaxes(logits, (1, 0, 2))  # for ver 3.
        logits = F.swapaxes(logits, axis1=0, axis2=1)

        logits = [t[0, :, :] for t in F.split_axis(logits, len(logits), axis=0)]

        loss = ctc_loss(logits, labels, blank_symbol=blank_index,
                        input_length=inputs_seq_len,
                        label_length=labels_seq_len)

        return loss
