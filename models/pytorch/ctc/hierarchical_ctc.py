#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Hierarchical CTC model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from warpctc_pytorch import CTCLoss
except:
    raise ImportError('Install warpctc_pytorch.')

import torch.nn as nn

from models.pytorch.ctc.ctc import CTC, _concatenate_labels
from models.pytorch.encoders.load_encoder import load
from utils.io.variable import np2var

NEG_INF = -float("inf")
LOG_0 = NEG_INF
LOG_1 = 0


class HierarchicalCTC(CTC):
    """Hierarchical CTC model.
    Args:
        input_size (int): the dimension of input features
        encoder_type (string): the type of the encoder. Set lstm or gru or rnn.
        bidirectional (bool): if True create a bidirectional encoder
        num_units (int): the number of units in each layer
        num_proj (int): the number of nodes in recurrent projection layer
        num_layers (int): the number of layers of the encoder of the main task
        num_layers_sub (int): the number of layers of the encoder of the sub task
        fc_list (list):
        dropout (float): the probability to drop nodes
        main_loss_weight (float): A weight parameter for the main CTC loss
        num_classes (int): the number of classes of target labels of the main task
            (excluding a blank class)
        num_classes_sub (int): the number of classes of target labels of the sub task
            (excluding a blank class)
        parameter_init (float, optional): Range of uniform distribution to
            initialize weight parameters
        logits_temperature (float):
        num_stack (int, optional): the number of frames to stack
        splice (int, optional): frames to splice. Default is 1 frame.
        conv_channels (list, optional):
        conv_kernel_sizes (list, optional):
        conv_strides (list, optional):
        poolings (list, optional):
        batch_norm (bool, optional):
    """

    def __init__(self,
                 input_size,
                 encoder_type,
                 bidirectional,
                 num_units,
                 num_proj,
                 num_layers,
                 num_layers_sub,  # ***
                 fc_list,
                 dropout,
                 main_loss_weight,  # ***
                 num_classes,
                 num_classes_sub,  # ***
                 parameter_init=0.1,
                 logits_temperature=1,
                 num_stack=1,
                 splice=1,
                 conv_channels=[],
                 conv_kernel_sizes=[],
                 conv_strides=[],
                 poolings=[],
                 batch_norm=False):

        super(HierarchicalCTC, self).__init__(
            input_size=input_size,  # 120 or 123
            encoder_type=encoder_type,
            bidirectional=bidirectional,
            num_units=num_units,
            num_proj=num_proj,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
            parameter_init=parameter_init,
            fc_list=fc_list,
            logits_temperature=logits_temperature,
            batch_norm=batch_norm)

        self.num_layers_sub = num_layers_sub

        # Setting for CTC
        self.num_classes_sub = num_classes_sub + 1
        # NOTE: Add blank class

        # Setting for MTL
        self.main_loss_weight = main_loss_weight

        # Load an instance
        encoder = load(encoder_type=encoder_type + '_hierarchical')

        # Call the encoder function
        # NOTE: overide encoder
        if encoder_type in ['lstm', 'gru', 'rnn']:
            self.encoder = encoder(
                input_size=input_size,  # 120 or 123
                rnn_type=encoder_type,
                bidirectional=bidirectional,
                num_units=num_units,
                num_proj=num_proj,
                num_layers=num_layers,
                num_layers_sub=num_layers_sub,
                dropout=dropout,
                parameter_init=parameter_init,
                use_cuda=self.use_cuda,
                batch_first=False,
                num_stack=num_stack,
                splice=splice,
                conv_channels=conv_channels,
                conv_kernel_sizes=conv_kernel_sizes,
                conv_strides=conv_strides,
                poolings=poolings,
                batch_norm=batch_norm)
        else:
            raise NotImplementedError

        self.fc_sub = nn.Linear(
            num_units * self.num_directions, self.num_classes_sub)

    def forward(self, inputs, labels, labels_sub, inputs_seq_len,
                labels_seq_len, labels_seq_len_sub, volatile=False):
        """Forward computation.
        Args:
            inputs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            labels (np.ndarray): A tensor of size `[B, T_out]`
            labels_sub (np.ndarray): A tensor of size `[B, T_out_sub]`
            inputs_seq_len (np.ndarray): A tensor of size `[B]`
            labels_seq_len (np.ndarray): A tensor of size `[B]`
            labels_seq_len_sub (np.ndarray): A tensor of size `[B]`
            volatile (bool, optional): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (FloatTensor): A tensor of size `[1]`
            loss_main (FloatTensor): A tensor of size `[1]`
            loss_sub (FloatTensor): A tensor of size `[1]`
        """
        # Wrap by Variable
        inputs = np2var(inputs, use_cuda=self.use_cuda)
        labels = np2var(labels, dtype='int', use_cuda=False)
        labels_sub = np2var(labels_sub, dtype='int', use_cuda=False)
        inputs_seq_len = np2var(
            inputs_seq_len, dtype='int', use_cuda=self.use_cuda)
        labels_seq_len = np2var(labels_seq_len, dtype='int', use_cuda=False)
        labels_seq_len_sub = np2var(
            labels_seq_len_sub, dtype='int', use_cuda=False)

        _labels = labels + 1
        _labels_sub = labels_sub + 1
        # NOTE: index 0 is reserved for blank

        # Encode acoustic features
        logits, logits_sub, perm_indices = self._encode(
            inputs, inputs_seq_len, volatile=volatile, is_multi_task=True)

        # Permutate indices
        _labels = _labels[perm_indices.cpu()]
        _labels_sub = _labels_sub[perm_indices.cpu()]
        inputs_seq_len = inputs_seq_len[perm_indices]
        labels_seq_len = labels_seq_len[perm_indices.cpu()]
        labels_seq_len_sub = labels_seq_len_sub[perm_indices.cpu()]

        max_time, batch_size = logits.size()[:2]

        # Concatenate all labels for warpctc_pytorch
        # `[B, T_out]` -> `[1,]`
        concatenated_labels = _concatenate_labels(
            _labels, labels_seq_len)
        concatenated_labels_sub = _concatenate_labels(
            _labels_sub, labels_seq_len_sub)

        # Output smoothing
        if self.logits_temperature != 1:
            logits /= self.logits_temperature
            logits_sub /= self.logits_temperature

        # Compute CTC loss
        ctc_loss_fn = CTCLoss()
        loss_main = ctc_loss_fn(logits, concatenated_labels,
                                inputs_seq_len.cpu(), labels_seq_len)
        loss_sub = ctc_loss_fn(logits_sub, concatenated_labels_sub,
                               inputs_seq_len.clone().cpu(), labels_seq_len_sub)
        loss = loss_main * self.main_loss_weight + \
            loss_sub * (1 - self.main_loss_weight)

        # Average the loss by mini-batch
        loss /= batch_size

        return loss, loss_main * self.main_loss_weight / batch_size, loss_sub * (1 - self.main_loss_weight) / batch_size
