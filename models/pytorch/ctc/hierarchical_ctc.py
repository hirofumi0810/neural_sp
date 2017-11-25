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
# import torch.nn.functional as F

from models.pytorch.ctc.ctc import CTC
from models.pytorch.encoders.load_encoder import load
from utils.io.variable import var2np

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
        num_layers (int): the number of layers of the encoder
        dropout (float): the probability to drop nodes
        main_loss_weight (float): A weight parameter for the main CTC loss
        num_classes (int): the number of classes of target labels
            (excluding a blank class)
        num_stack (int, optional): the number of frames to stack
        splice (int, optional): frames to splice. Default is 1 frame.
        parameter_init (float, optional): Range of uniform distribution to
            initialize weight parameters
        bottleneck_dim (int, optional):
        logits_temperature (float):
    """

    def __init__(self,
                 input_size,
                 encoder_type,
                 bidirectional,
                 num_units,
                 num_proj,
                 num_layers,
                 num_layers_sub,  # ***
                 dropout,
                 main_loss_weight,  # ***
                 num_classes,
                 num_classes_sub,  # ***
                 num_stack=1,
                 splice=1,
                 parameter_init=0.1,
                 bottleneck_dim=None,
                 logits_temperature=1):

        super(HierarchicalCTC, self).__init__(
            input_size=input_size,
            encoder_type=encoder_type,
            bidirectional=bidirectional,
            num_units=num_units,
            num_proj=num_proj,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
            num_stack=num_stack,
            splice=splice,
            parameter_init=parameter_init,
            bottleneck_dim=bottleneck_dim,
            logits_temperature=logits_temperature)

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
                input_size=self.input_size,
                rnn_type=encoder_type,
                bidirectional=bidirectional,
                num_units=num_units,
                num_proj=num_proj,
                num_layers=num_layers,
                num_layers_sub=num_layers_sub,
                dropout=dropout,
                parameter_init=parameter_init,
                use_cuda=self.use_cuda,
                batch_first=False)
        else:
            raise NotImplementedError

        self.fc_sub = nn.Linear(
            num_units * self.num_directions, self.num_classes_sub)

    def forward(self, inputs, labels, labels_sub, inputs_seq_len,
                labels_seq_len, labels_seq_len_sub, volatile=False):
        """Forward computation.
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            labels (LongTensor): A tensor of size `[B, T_out]`
            labels_sub (LongTensor): A tensor of size `[B, T_out_sub]`
            inputs_seq_len (IntTensor): A tensor of size `[B]`
            labels_seq_len (IntTensor): A tensor of size `[B]`
            labels_seq_len_sub (IntTensor): A tensor of size `[B]`
            volatile (bool, optional): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (FloatTensor): A tensor of size `[1]`
            loss_main (FloatTensor): A tensor of size `[1]`
            loss_sub (FloatTensor): A tensor of size `[1]`
        """
        labels_tmp = labels + 1
        labels_sub_tmp = labels_sub + 1
        # NOTE: index 0 is reserved for blank

        # Encode acoustic features
        logits, logits_sub, perm_indices = self._encode(
            inputs, inputs_seq_len, volatile=volatile)

        # Permutate indices
        labels_tmp = labels_tmp[perm_indices]
        labels_sub_tmp = labels_sub_tmp[perm_indices]
        inputs_seq_len = inputs_seq_len[perm_indices]
        labels_seq_len = labels_seq_len[perm_indices]
        labels_seq_len_sub = labels_seq_len_sub[perm_indices]

        max_time, batch_size = logits.size()[:2]

        # Concatenate all labels for warpctc_pytorch
        # `[B, T_out]` -> `[1,]`
        concatenated_labels = self._concatenate_labels(
            labels_tmp, labels_seq_len)
        concatenated_labels_sub = self._concatenate_labels(
            labels_sub_tmp, labels_seq_len_sub)

        # Output smoothing
        if self.logits_temperature != 1:
            logits /= self.logits_temperature
            logits_sub /= self.logits_temperature

        # Compute CTC loss
        ctc_loss_fn = CTCLoss()
        loss_main = ctc_loss_fn(
            logits, concatenated_labels.cpu(),
            inputs_seq_len.cpu(), labels_seq_len.cpu())
        loss_sub = ctc_loss_fn(
            logits_sub, concatenated_labels_sub.cpu(),
            inputs_seq_len.cpu(), labels_seq_len_sub.cpu())
        loss = loss_main * self.main_loss_weight + \
            loss_sub * (1 - self.main_loss_weight)

        # Average the loss by mini-batch
        loss /= batch_size

        return loss, loss_main * self.main_loss_weight / batch_size, loss_sub * (1 - self.main_loss_weight) / batch_size

    def _encode(self, inputs, inputs_seq_len, volatile):
        """Encode acoustic features.
        Args:
            inputs (FloatTensor): A tensor of size `[B, T, input_size]`
            inputs_seq_len (IntTensor): A tensor of size `[B]`
            volatile (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            logits (FloatTensor): A tensor of size
                `[T, B, num_classes (including blank)]`
            logits_sub (FloatTensor): A tensor of size
                `[T, B, num_classes_sub (including blank)]`
        """
        encoder_outputs, _, encoder_outputs_sub, _, perm_indices = self.encoder(
            inputs, inputs_seq_len, volatile, mask_sequence=True)
        max_time, batch_size = encoder_outputs.size()[:2]

        # Convert to 2D tensor
        # encoder_outputs = encoder_outputs.contiguous()
        encoder_outputs = encoder_outputs.view(max_time, batch_size, -1)
        encoder_outputs_sub = encoder_outputs_sub.view(
            max_time, batch_size, -1)

        if self.bottleneck_dim is not None:
            logits = self.bottleneck(encoder_outputs)
            logits = self.fc(logits)
        else:
            logits = self.fc(encoder_outputs)
        logits_sub = self.fc_sub(encoder_outputs_sub)

        # Reshape back to 3D tensor
        logits = logits.view(max_time, batch_size, -1)
        logits_sub = logits_sub.view(max_time, batch_size, -1)

        return logits, logits_sub, perm_indices

    def decode_sub(self, inputs, inputs_seq_len, beam_width=1,
                   max_decode_length=None):
        """
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            inputs_seq_len (IntTensor): A tensor of size `[B]`
            beam_width (int, optional): the size of beam
            max_decode_length: not used
        Returns:
            best_hyp ():
            perm_indices ():
        """
        # Encode acoustic features
        _, logits_sub, perm_indices = self._encode(
            inputs, inputs_seq_len, volatile=True)

        # Convert to batch-major
        logits_sub = logits_sub.transpose(0, 1)

        # log_probs = F.log_softmax(logits_sub, dim=logits_sub.dim() - 1)
        log_probs = self.log_softmax(logits_sub)
        # TODO: update pytorch version

        if beam_width == 1:
            # torch-based decoder
            best_hyp = self._decode_greedy(log_probs, inputs_seq_len)
        else:
            # torch-based decoder
            # best_hyp = self._decode_beam(log_probs, inputs_seq_len, beam_width)

            # numpy-based decoder
            log_probs = var2np(log_probs)
            inputs_seq_len = var2np(inputs_seq_len)
            best_hyp = self.decode_beam(log_probs, inputs_seq_len, beam_width)

        best_hyp -= 1
        # NOTE: index 0 is reserved for blank

        return best_hyp, perm_indices
