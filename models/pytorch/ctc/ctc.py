#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from warpctc_pytorch import CTCLoss
except:
    raise ImportError('Install warpctc_pytorch.')

import numpy as np
from itertools import groupby

import torch
import torch.nn.functional as F

from models.pytorch.base import ModelBase
from models.pytorch.encoders.load_encoder import load
# from models.pytorch.ctc.decoders.greedy_decoder import GreedyDecoder
from models.pytorch.ctc.decoders.beam_search_decoder import BeamSearchDecoder
from utils.io.variable import var2np


#################################################################
# Useful documentation from
# https://discuss.pytorch.org/t/ctcloss-with-warp-ctc-help/8788
#################################################################
# The CTC loss function computes total CTC loss on a batch of sequences.
# Total loss is not the equal to the sum of the losses for individual samples.
# Not clear why.
# https://discuss.pytorch.org/t/how-to-fill-the-label-tensor-for-ctc-loss/5801
#
# ctc_loss(probs, labels, prob_sizes, label_sizes)
#
# probs
# -----
# Estimated probabilities.
# Tensor of size (seq_len, batch_size, n_alphabet+1).
# Note that each sample in the batch may have a different sequence length, so
#   the seq_len size of the tensor is maximum of all sequence lengths in the
#   batch.
# The tail of short sequences should be padded with zeros.
# The [0] index of the probabilities is reserved for "blanks" which is why the
#   3rd dimension is of size n_alphabet+1.
#
# labels
# ------
# Ground truth labels.
# A 1-D tensor composed of concatenated sequences of int labels
#   (not one-hot vectors).
# Scalars should range from 1 to n_alphabet.
# 0 is not used, as that is reserved for blanks.
# For example, if the label sequences for two samples are [1, 2] and [4, 5, 7]
#   then the tensor is [1, 2, 4, 5, 7].
#
# prob_sizes
# ----------
# Sequence lengths of the probabilities.
# A 1-D tensor of ints of length batch_size.
# The ith value specifies the sequence length of the probabilities of the ith
#   sample that are used in computing that sample's CTC loss.
# Values in the probs tensor that extend beyond this length are ignored.
#
# label_sizes
# ------------
# Sequence lengths of the labels.
# A 1-D tensor of ints of length batch_size.
# The ith value specifies the sequence length of the labels of the ith sample
#   that are used in computing that sample's CTC loss.
# The length of the labels vector should be equal to the cumulative sum of the
#   elements in the label_sizes vector.


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
            (except for a blank label)
        num_stack (int, optional): the number of frames to stack
        splice (int, optional): frames to splice. Default is 1 frame.
        parameter_init (float, optional): Range of uniform distribution to
            initialize weight parameters
        bottleneck_dim (int, optional):
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
                 num_stack=1,
                 splice=1,
                 parameter_init=0.1,
                 bottleneck_dim=None):

        super(ModelBase, self).__init__()

        self.input_size = input_size
        self.splice = splice
        self.encoder_type = encoder_type
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_units = num_units
        self.num_proj = num_proj
        self.num_layers = num_layers
        self.bottleneck_dim = bottleneck_dim
        self.dropout = dropout
        self.num_classes = num_classes + 1
        # NOTE: Add blank class
        self.blank_index = 0
        # NOTE: index 0 is reserved for blank in warpctc_pytorch
        self.parameter_init = parameter_init
        self.name = 'pt_ctc'

        # Load encoder
        encoder = load(encoder_type=encoder_type)
        if encoder_type in ['lstm', 'gru', 'rnn']:
            self.encoder = encoder(
                input_size=input_size,
                rnn_type=encoder_type,
                bidirectional=bidirectional,
                num_units=num_units,
                num_proj=num_proj,
                num_layers=num_layers,
                dropout=dropout,
                parameter_init=parameter_init,
                use_cuda=self.use_cuda,
                batch_first=True)
        else:
            raise NotImplementedError

        if self.bottleneck_dim is not None:
            self.bottleneck = torch.nn.Linear(
                num_units * self.num_directions, bottleneck_dim)
            self.fc = torch.nn.Linear(bottleneck_dim, self.num_classes)
        else:
            self.fc = torch.nn.Linear(
                num_units * self.num_directions, self.num_classes)

        # Set CTC decoders
        # self.decode_greedy = GreedyDecoder(blank_index=self.blank_index)
        self.decode_beam = BeamSearchDecoder(blank_index=self.blank_index)
        # TODO: set space index

    def forward(self, inputs, volatile=False):
        """
        Args:
            inputs (FloatTensor): A tensor of size `[B, T, input_size]`
            volatile (bool, optional): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            logits (FloatTensor): A tensor of size
                `[T, B, num_classes (including blank)]`
        """
        return self._forward(inputs, volatile)

    def _forward(self, inputs, volatile):
        """
        Args:
            inputs (FloatTensor): A tensor of size `[B, T, input_size]`
            volatile (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            logits (FloatTensor): A tensor of size
                `[T, B, num_classes (including blank)]`
        """
        encoder_outputs, final_state = self.encoder(inputs)
        batch_size, max_time = encoder_outputs.size()[:2]

        # Convert to 2D tensor
        encoder_outputs = encoder_outputs.contiguous()
        encoder_outputs = encoder_outputs.view(batch_size * max_time, -1)

        if self.bottleneck_dim is not None:
            logits = self.bottleneck(encoder_outputs)
            logits = self.fc(logits)
        else:
            logits = self.fc(encoder_outputs)

        # Reshape back to 3D tensor
        logits = logits.view(batch_size, max_time, -1)

        return logits

    def compute_loss(self, logits, labels, inputs_seq_len, labels_seq_len):
        """
        Args:
            logits (FloatTensor): A tensor of size `[B, T_in, num_classes]`
            labels (LongTensor): A tensor of size `[B, T_out]`
            inputs_seq_len (LongTensor): A tensor of size `[B]`
            labels_seq_len (LongTensor): A tensor of size `[B]`
        Returns:
            ctc_loss (FloatTensor): A tensor of size `[]`
        """
        batch_size, max_time = logits.size()[:2]
        ctc_loss_fn = CTCLoss()

        logits = logits.transpose(0, 1)
        # NOTE; logits must be a tensor of size `[T_in, B, num_classes]`

        ctc_loss = ctc_loss_fn(logits, labels, inputs_seq_len, labels_seq_len)

        # Average the loss by mini-batch
        ctc_loss /= batch_size

        return ctc_loss

    def posteriors(self, logits, temperature=1, blank_prior=None):
        """
        Args:
            logits (FloatTensor): A tensor of size `[B, T, num_classes]`
            temperature (float, optional): the temperature parameter for the
                softmax layer in the inference stage
            blank_prior (float, optional):
        Returns:
            probs (FloatTensor): A tensor of size `[]`
        """
        probs = self.softmax(logits / temperature)

        # Divide by blank prior
        if blank_prior is not None:
            raise NotImplementedError

        return probs

    def decode(self, inputs, inputs_seq_len, beam_width=1):
        """
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            inputs_seq_len (LongTensor): A tensor of size `[B]`
            beam_width (int, optional): the size of beam
        Returns:

        """
        logits = self._forward(inputs, volatile=True)
        # log_probs = F.log_softmax(logits, dim=logits.dim() - 1)
        log_probs = self.log_softmax(logits)
        # TODO: update pytorch version

        if beam_width == 1:
            # torch-based decoder
            return self._decode_greedy(log_probs, inputs_seq_len)
        else:
            # numpy-based decoder
            log_probs = var2np(log_probs)
            inputs_seq_len = var2np(inputs_seq_len)
            return self._decode_beam(log_probs, inputs_seq_len, beam_width)

    def _decode_greedy(self, log_probs, inputs_seq_len):
        """
        Args:
            log_probs (FloatTensor): log-scale probabilities
                A tensor of size `[B, num_classes (including blank)]`
            inputs_seq_len (LongTensor): A tensor of size `[B]`
        Returns:
            results (np.ndarray): A tensor of size `[B,]`
        """
        batch_size = log_probs.size(0)
        results = []

        # Pickup argmax class
        for i_batch in range(batch_size):
            indices = []
            time = var2np(inputs_seq_len)[i_batch]
            for t in range(time):
                _, argmax = torch.max(log_probs[i_batch, t], dim=0)
                indices.append(var2np(argmax).tolist()[0])

            # Step 1. Collapse repeated labels
            collapsed_indices = [x[0] for x in groupby(indices)]

            # Step 2. Remove all blank labels
            best_hyp = [x for x in filter(
                lambda x: x != self.blank_index, collapsed_indices)]
            results.append(best_hyp)

        # return np.squeeze(np.array(results), axis=2)
        return np.array(results)

    def _decode_beam(self, log_probs, inputs_seq_len, beam_width):
        """
        Args:
            log_probs (FloatTensor): log-scale probabilities
                A tensor of size `[B, num_classes (including blank)]`
            inputs_seq_len (LongTensor): A tensor of size `[B]`
            beam_width (int): the size of beam
        Returns:
            results (np.ndarray):
        """
        raise NotImplementedError
