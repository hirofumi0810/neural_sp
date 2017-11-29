#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""The Connectionist Temporal Classification model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from warpctc_pytorch import CTCLoss
except ImportError:
    raise ImportError('Install warpctc_pytorch.')
try:
    import pytorch_ctc
except ImportError:
    raise ImportError('Install pytorch_ctc.')

import torch
import torch.nn as nn
from torch.autograd import Variable
# import torch.nn.functional as F

from models.pytorch.base import ModelBase
from models.pytorch.encoders.load_encoder import load
from models.pytorch.ctc.decoders.greedy_decoder import GreedyDecoder
from models.pytorch.ctc.decoders.beam_search_decoder import BeamSearchDecoder
from utils.io.variable import var2np

NEG_INF = -float("inf")
LOG_0 = NEG_INF
LOG_1 = 0


class CTC(ModelBase):
    """The Connectionist Temporal Classification model.
    Args:
        input_size (int): the dimension of input features
        encoder_type (string): the type of the encoder. Set lstm or gru or rnn.
        bidirectional (bool): if True create a bidirectional encoder
        num_units (int): the number of units in each layer
        num_proj (int): the number of nodes in recurrent projection layer
        num_layers (int): the number of layers of the encoder
        dropout (float): the probability to drop nodes
        num_classes (int): the number of classes of target labels
            (excluding a blank class)
        parameter_init (float, optional): Range of uniform distribution to
            initialize weight parameters
        bottleneck_dim_list (list, optional):
        logits_temperature (float):
        num_stack (int, optional): the number of frames to stack
        splice (int, optional): frames to splice. Default is 1 frame.
        channels (list, optional):
        kernel_sizes (list, optional):
        strides (list, optional):
        batch_norm (bool, optional):
    """

    def __init__(self,
                 input_size,
                 encoder_type,
                 bidirectional,
                 num_units,
                 num_proj,
                 num_layers,
                 dropout,
                 num_classes,
                 parameter_init=0.1,
                 bottleneck_dim_list=[],
                 logits_temperature=1,
                 num_stack=1,
                 splice=1,
                 channels=[],
                 kernel_sizes=[],
                 strides=[],
                 batch_norm=False):

        super(ModelBase, self).__init__()

        self.num_directions = 2 if bidirectional else 1
        self.bottleneck_dim_list = bottleneck_dim_list

        # Setting for CTC
        self.num_classes = num_classes + 1
        # NOTE: Add blank class
        self.blank_index = 0
        # NOTE: index 0 is reserved for blank in warpctc_pytorch
        self.logits_temperature = logits_temperature

        self.parameter_init = parameter_init

        # Load an instance
        encoder = load(encoder_type=encoder_type)

        # Call the encoder function
        if encoder_type in ['lstm', 'gru', 'rnn']:
            self.encoder = encoder(
                input_size=input_size,  # 120 or 123
                rnn_type=encoder_type,
                bidirectional=bidirectional,
                num_units=num_units,
                num_proj=num_proj,
                num_layers=num_layers,
                dropout=dropout,
                parameter_init=parameter_init,
                use_cuda=self.use_cuda,
                batch_first=False,
                num_stack=num_stack,
                splice=splice,
                channels=channels,
                kernel_sizes=kernel_sizes,
                strides=strides,
                batch_norm=batch_norm)
        else:
            raise NotImplementedError

        if len(bottleneck_dim_list) > 0:
            bottleneck_layers = []
            for i in range(len(bottleneck_dim_list)):
                if i == 0:
                    bottleneck_layers.append(nn.Linear(
                        num_units *
                        self.num_directions, bottleneck_dim_list[i],
                        bias=not batch_norm))
                else:
                    if batch_norm:
                        bottleneck_layers.append(nn.BatchNorm1d(
                            bottleneck_dim_list[i - 1]))
                    bottleneck_layers.append(nn.Linear(
                        bottleneck_dim_list[i - 1], bottleneck_dim_list[i],
                        bias=not batch_norm))
                bottleneck_layers.append(nn.Dropout(p=dropout))
            # TODO: try batch_norm
            self.bottleneck_layers = nn.Sequential(*bottleneck_layers)
            self.fc = nn.Linear(bottleneck_dim_list[-1], self.num_classes,
                                bias=not batch_norm)
        else:
            self.fc = nn.Linear(
                num_units * self.num_directions, self.num_classes)

        # Set CTC decoders
        self._decode_greedy_np = GreedyDecoder(blank_index=self.blank_index)
        self._decode_beam_np = BeamSearchDecoder(blank_index=self.blank_index)
        # TODO: set space index

    def forward(self, inputs, labels, inputs_seq_len, labels_seq_len,
                volatile=False):
        """Forward computation.
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            labels (LongTensor): A tensor of size `[B, T_out]`
            inputs_seq_len (IntTensor): A tensor of size `[B]`
            labels_seq_len (IntTensor): A tensor of size `[B]`
            volatile (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (FloatTensor): A tensor of size `[1]`
        """
        _labels = labels + 1
        # NOTE: index 0 is reserved for blank

        # Encode acoustic features
        logits, perm_indices = self._encode(
            inputs, inputs_seq_len, volatile=volatile)

        # Permutate indices
        _labels = _labels[perm_indices]
        inputs_seq_len = inputs_seq_len[perm_indices]
        labels_seq_len = labels_seq_len[perm_indices]

        max_time, batch_size = logits.size()[:2]

        # Concatenate all labels for warpctc_pytorch
        # `[B, T_out]` -> `[1,]`
        concatenated_labels = _concatenate_labels(
            _labels, labels_seq_len)

        # Output smoothing
        if self.logits_temperature != 1:
            logits /= self.logits_temperature

        # Compute CTC loss
        ctc_loss_fn = CTCLoss()
        ctc_loss = ctc_loss_fn(logits, concatenated_labels.cpu(),
                               inputs_seq_len.cpu(), labels_seq_len.cpu())

        # Average the loss by mini-batch
        ctc_loss /= batch_size

        return ctc_loss

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
            perm_indices (FloatTensor):
        """
        encoder_outputs, _, perm_indices = self.encoder(
            inputs, inputs_seq_len, volatile, mask_sequence=True)
        max_time, batch_size = encoder_outputs.size()[:2]

        # Convert to 2D tensor
        encoder_outputs = encoder_outputs.view(max_time * batch_size, -1)
        # contiguous()

        if len(self.bottleneck_dim_list) > 0:
            encoder_outputs = self.bottleneck_layers(encoder_outputs)
        logits = self.fc(encoder_outputs)

        # Reshape back to 3D tensor
        logits = logits.view(max_time, batch_size, -1)

        return logits, perm_indices

    def posteriors(self, inputs, inputs_seq_len, temperature=1, blank_prior=None):
        """
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            inputs_seq_len (IntTensor): A tensor of size `[B]`
            temperature (float, optional): the temperature parameter for the
                softmax layer in the inference stage
            blank_prior (float, optional):
        Returns:
            probs (FloatTensor): A tensor of size `[]`
        """
        # Encode acoustic features
        logits, perm_indices = self._encode(
            inputs, inputs_seq_len, volatile=True)

        # Convert to batch-major
        logits = logits.transpose(0, 1)

        probs = self.softmax(logits / temperature)

        # Divide by blank prior
        if blank_prior is not None:
            raise NotImplementedError

        return probs

    def decode(self, inputs, inputs_seq_len, beam_width=1,
               max_decode_length=None, decode_by_pytorch_ctc=False):
        """
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            inputs_seq_len (IntTensor): A tensor of size `[B]`
            beam_width (int, optional): the size of beam
            max_decode_length: not used
            decode_by_pytorch_ctc (bool, optional): if True, decoding will be
                conducted by using pytorch_ctc package.
        Returns:
            best_hyps ():
            perm_indices ():
        """
        # Encode acoustic features
        if hasattr(self, 'main_loss_weight'):
            logits, _, perm_indices = self._encode(
                inputs, inputs_seq_len, volatile=True)
        else:
            logits, perm_indices = self._encode(
                inputs, inputs_seq_len, volatile=True)

        # Convert to batch-major
        logits = logits.transpose(0, 1)

        if decode_by_pytorch_ctc:
            raise NotImplementedError
            # probs = self.softmax(logits)
            # scorer = pytorch_ctc.Scorer()
            # decoder = pytorch_ctc.CTCBeamDecoder(
            #     scorer, labels, top_paths=1, beam_width=beam_width,
            #     blank_index=0, space_index=28, merge_repeated=True)
            # output, score, out_seq_len = decoder.decode(probs, seq_len=None)
        else:
            # log_probs = F.log_softmax(logits, dim=logits.dim() - 1)
            log_probs = self.log_softmax(logits)
            # TODO: update pytorch version

            if beam_width == 1:
                best_hyps = self._decode_greedy_np(
                    var2np(log_probs), var2np(inputs_seq_len))
            else:
                best_hyps = self._decode_beam_np(
                    var2np(log_probs), var2np(inputs_seq_len), beam_width)

            best_hyps -= 1
            # NOTE: index 0 is reserved for blank

        return best_hyps, perm_indices


def _concatenate_labels(labels, labels_seq_len):
    """Concatenate all labels in mini-batch and convert to a 1D tensor.
    Args:
        labels (LongTensor): A tensor of size `[B, T_out]`
        labels_seq_len (IntTensor): A tensor of size `[B]`
    Returns:
        concatenated_labels (): A tensor of size `[all_label_num]`
    """
    batch_size = labels.size(0)
    total_lables_seq_len = labels_seq_len.data.sum()
    concatenated_labels = Variable(
        torch.zeros(total_lables_seq_len)).int()
    label_counter = 0
    for i_batch in range(batch_size):
        concatenated_labels[label_counter:label_counter +
                            labels_seq_len.data[i_batch]] = labels[i_batch][:labels_seq_len.data[i_batch]]
        label_counter += labels_seq_len.data[i_batch]

    return concatenated_labels
