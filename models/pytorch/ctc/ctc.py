#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""The Connectionist Temporal Classification model (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from warpctc_pytorch import CTCLoss
    ctc_loss = CTCLoss()
except ImportError:
    raise ImportError('Install warpctc_pytorch.')
# try:
#     import pytorch_ctc
# except ImportError:
#     raise ImportError('Install pytorch_ctc.')

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.pytorch.base import ModelBase
from models.pytorch.linear import LinearND
from models.pytorch.encoders.load_encoder import load
from models.pytorch.ctc.decoders.greedy_decoder import GreedyDecoder
from models.pytorch.ctc.decoders.beam_search_decoder import BeamSearchDecoder
# from models.pytorch.ctc.decoders.beam_search_decoder2 import BeamSearchDecoder
from utils.io.variable import np2var, var2np

NEG_INF = -float("inf")
LOG_0 = NEG_INF
LOG_1 = 0


class CTC(ModelBase):
    """The Connectionist Temporal Classification model.
    Args:
        input_size (int): the dimension of input features
        encoder_type (string): the type of the encoder. Set lstm or gru or rnn.
        encoder_bidirectional (bool): if True create a bidirectional encoder
        encoder_num_units (int): the number of units in each layer
        encoder_num_proj (int): the number of nodes in recurrent projection layer
        encoder_num_layers (int): the number of layers of the encoder
        fc_list (list):
        dropout (float): the probability to drop nodes
        num_classes (int): the number of classes of target labels
            (excluding the blank class)
        parameter_init (float, optional): Range of uniform distribution to
            initialize weight parameters
        subsample_list (list, optional): subsample in the corresponding layers (True)
            ex.) [False, True, True, False] means that subsample is conducted
                in the 2nd and 3rd layers.
        subsample_type (string, optional): drop or concat
        logits_temperature (float):
        num_stack (int, optional): the number of frames to stack
        splice (int, optional): frames to splice. Default is 1 frame.
        conv_channels (list, optional):
        conv_kernel_sizes (list, optional):
        conv_strides (list, optional):
        poolings (list, optional):
        activation (string, optional): The activation function of CNN layers.
            Choose from relu or prelu or hard_tanh or maxout
        batch_norm (bool, optional):
        label_smoothing_prob (float, optional):
        weight_noise_std (float, optional):
        residual (bool, optional):
        dense_residual (bool, optional):
    """

    def __init__(self,
                 input_size,
                 encoder_type,
                 encoder_bidirectional,
                 encoder_num_units,
                 encoder_num_proj,
                 encoder_num_layers,
                 fc_list,
                 dropout,
                 num_classes,
                 parameter_init=0.1,
                 subsample_list=[],
                 subsample_type='drop',
                 logits_temperature=1,
                 num_stack=1,
                 splice=1,
                 conv_channels=[],
                 conv_kernel_sizes=[],
                 conv_strides=[],
                 poolings=[],
                 activation='relu',
                 batch_norm=False,
                 label_smoothing_prob=0,
                 weight_noise_std=0,
                 residual=False,
                 dense_residual=False):

        super(ModelBase, self).__init__()

        # Setting for the encoder
        self.input_size = input_size
        self.num_stack = num_stack
        self.encoder_type = encoder_type
        self.num_directions = 2 if encoder_bidirectional else 1
        self.fc_list = fc_list
        self.subsample_list = subsample_list

        # Setting for CTC
        self.num_classes = num_classes + 1  # Add the blank class
        self.logits_temperature = logits_temperature

        # Setting for regualarization
        self.parameter_init = parameter_init
        self.weight_noise_injection = False
        self.weight_noise_std = float(weight_noise_std)
        self.label_smoothing_prob = label_smoothing_prob

        # Call the encoder function
        if encoder_type in ['lstm', 'gru', 'rnn']:
            self.encoder = load(encoder_type=encoder_type)(
                input_size=input_size,
                rnn_type=encoder_type,
                bidirectional=encoder_bidirectional,
                num_units=encoder_num_units,
                num_proj=encoder_num_proj,
                num_layers=encoder_num_layers,
                dropout=dropout,
                subsample_list=subsample_list,
                subsample_type=subsample_type,
                use_cuda=self.use_cuda,
                batch_first=True,
                merge_bidirectional=False,
                num_stack=num_stack,
                splice=splice,
                conv_channels=conv_channels,
                conv_kernel_sizes=conv_kernel_sizes,
                conv_strides=conv_strides,
                poolings=poolings,
                activation=activation,
                batch_norm=batch_norm,
                residual=residual,
                dense_residual=dense_residual)
        elif encoder_type == 'cnn':
            assert num_stack == 1
            assert splice == 1
            self.encoder = load(encoder_type=encoder_type)(
                input_size=input_size,
                conv_channels=conv_channels,
                conv_kernel_sizes=conv_kernel_sizes,
                conv_strides=conv_strides,
                poolings=poolings,
                dropout=dropout,
                use_cuda=self.use_cuda,
                activation=activation,
                batch_norm=batch_norm)
        else:
            raise NotImplementedError

        if len(fc_list) > 0:
            fc_layers = []
            for i in range(len(fc_list)):
                if i == 0:
                    if encoder_type == 'cnn':
                        bottle_input_size = self.encoder.output_size
                    else:
                        bottle_input_size = encoder_num_units * self.num_directions
                    # if batch_norm:
                    #     fc_layers.append(nn.BatchNorm1d(bottle_input_size))
                    fc_layers.append(LinearND(bottle_input_size, fc_list[i],
                                              dropout=dropout))
                else:
                    # if batch_norm:
                    #     fc_layers.append(nn.BatchNorm1d(fc_list[i - 1]))
                    fc_layers.append(LinearND(fc_list[i - 1], fc_list[i],
                                              dropout=dropout))
            self.fc_layers = nn.Sequential(*fc_layers)
            # TODO: remove a bias term in the case of batch normalization

            self.fc = LinearND(fc_list[-1], self.num_classes)
        else:
            self.fc = LinearND(
                encoder_num_units * self.num_directions, self.num_classes)

        # Initialize all weights with uniform distribution
        self.init_weights(
            parameter_init, distribution='uniform', ignore_keys=['bias'])

        # Initialize all biases with 0
        self.init_weights(0, distribution='uniform', keys=['bias'])

        # Recurrent weights are orthogonalized
        # self.init_weights(parameter_init, distribution='orthogonal',
        #                   keys=['lstm', 'weight'], ignore_keys=['bias'])

        # Initialize bias in forget gate with 1
        self.init_forget_gate_bias_with_one()

        # Set CTC decoders
        self._decode_greedy_np = GreedyDecoder(blank_index=0)
        self._decode_beam_np = BeamSearchDecoder(blank_index=0)
        # NOTE: index 0 is reserved for the blank class in warpctc_pytorch
        # TODO: set space index

    def forward(self, inputs, labels, inputs_seq_len, labels_seq_len,
                is_eval=False):
        """Forward computation.
        Args:
            inputs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            labels (np.ndarray): A tensor of size `[B, T_out]`
            inputs_seq_len (np.ndarray): A tensor of size `[B]`
            labels_seq_len (np.ndarray): A tensor of size `[B]`
            is_eval (bool, optional): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (FloatTensor or float): A tensor of size `[1]`
        """
        # Wrap by Variable
        xs = np2var(inputs, use_cuda=self.use_cuda, backend='pytorch')
        ys = np2var(labels, dtype='int', use_cuda=False, backend='pytorch')
        x_lens = np2var(
            inputs_seq_len, dtype='int', use_cuda=self.use_cuda, backend='pytorch')
        y_lens = np2var(
            labels_seq_len, dtype='int', use_cuda=False, backend='pytorch')

        # NOTE: index 0 is reserved for the blank class in warpctc_pytorch
        ys = ys + 1

        if is_eval:
            self.eval()
        else:
            self.train()

            # Gaussian noise injection
            if self.weight_noise_injection:
                self._inject_weight_noise(mean=0, std=self.weight_noise_std)

        # Encode acoustic features
        logits, x_lens, perm_idx = self._encode(xs, x_lens, volatile=is_eval)

        # Convert to time-major
        logits = logits.transpose(0, 1).contiguous()

        # Permutate indices
        if perm_idx is not None:
            ys = ys[perm_idx.cpu()]
            y_lens = y_lens[perm_idx.cpu()]

        # Concatenate all labels for warpctc_pytorch
        # `[B, T_out]` -> `[1,]`
        concatenated_labels = _concatenate_labels(ys, y_lens)

        # Output smoothing
        if self.logits_temperature != 1:
            logits = logits / self.logits_temperature

        # Compute CTC loss
        loss = ctc_loss(logits, concatenated_labels, x_lens.cpu(), y_lens)

        # Label smoothing (with uniform distribution)
        if self.label_smoothing_prob > 0:
            batch_size, label_num, num_classes = logits.size()
            log_probs = F.log_softmax(logits, dim=-1)
            uniform = Variable(torch.FloatTensor(
                batch_size, label_num, num_classes).fill_(np.log(1 / num_classes)))
            loss = loss * (1 - self.label_smoothing_prob) + F.kl_div(
                log_probs.cpu(), uniform,
                size_average=False, reduce=True) * self.label_smoothing_prob

        # Average the loss by mini-batch
        loss = loss / len(xs)

        if is_eval:
            loss = loss.data[0]

        return loss

    def _encode(self, xs, x_lens, volatile, is_multi_task=False):
        """Encode acoustic features.
        Args:
            xs (FloatTensor): A tensor of size `[B, T, input_size]`
            x_lens (IntTensor): A tensor of size `[B]`
            volatile (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
            is_multi_task (bool, optional):
        Returns:
            logits (FloatTensor): A tensor of size
                `[B, T, num_classes (including the blank class)]`
            x_lens (IntTensor): A tensor of size `[B]`
            logits_sub (FloatTensor): A tensor of size
                `[B, T, num_classes_sub (including the blank class)]`
            x_lens_sub (IntTensor): A tensor of size `[B]`
            perm_idx (LongTensor):
        """
        if is_multi_task:
            xs, x_lens, xs_sub, x_lens_sub, perm_idx = self.encoder(
                xs, x_lens, volatile)
        else:
            if self.encoder_type == 'cnn':
                xs, x_lens = self.encoder(xs, x_lens)
                perm_idx = None
            else:
                xs, x_lens, perm_idx = self.encoder(xs, x_lens, volatile)

        if len(self.fc_list) > 0:
            xs = self.fc_layers(xs)
        logits = self.fc(xs)

        if is_multi_task:
            logits_sub = self.fc_sub(xs_sub)
            return logits, x_lens, logits_sub, x_lens_sub, perm_idx
        else:
            return logits, x_lens, perm_idx

    def posteriors(self, inputs, inputs_seq_len, temperature=1,
                   blank_prior=None, is_sub_task=False):
        """Returns CTC posteriors (after the softmax layer).
        Args:
            inputs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            inputs_seq_len (np.ndarray): A tensor of size `[B]`
            temperature (float, optional): the temperature parameter for the
                softmax layer in the inference stage
            blank_prior (float, optional):
            is_sub_task (bool, optional):
        Returns:
            probs (np.ndarray): A tensor of size `[B, T, num_classes]`
        """
        # Wrap by Variable
        xs = np2var(
            inputs, use_cuda=self.use_cuda, volatile=True, backend='pytorch')
        x_lens = np2var(
            inputs_seq_len, dtype='int', use_cuda=self.use_cuda, volatile=True, backend='pytorch')

        # Change to evaluation mode
        self.eval()

        # Encode acoustic features
        if hasattr(self, 'main_loss_weight'):
            if is_sub_task:
                _, _, logits, _, perm_idx = self._encode(
                    xs, x_lens, volatile=True, is_multi_task=True)
            else:
                logits, _, _, _, perm_idx = self._encode(
                    xs, x_lens, volatile=True, is_multi_task=True)
        else:
            logits, _, perm_idx = self._encode(xs, x_lens, volatile=True)

        probs = F.softmax(logits / temperature, dim=-1)

        # Divide by blank prior
        if blank_prior is not None:
            raise NotImplementedError

        # Permutate indices to the original order
        if perm_idx is not None:
            probs = probs[perm_idx]

        return var2np(probs)

    def decode(self, inputs, inputs_seq_len, beam_width=1,
               max_decode_len=None, is_sub_task=False):
        """
        Args:
            inputs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            inputs_seq_len (np.ndarray): A tensor of size `[B]`
            beam_width (int, optional): the size of beam
            max_decode_len: not used (to make CTC compatible with attention)
            is_sub_task (bool, optional):
        Returns:
            best_hyps (np.ndarray):
        """
        # Wrap by Variable
        xs = np2var(
            inputs, use_cuda=self.use_cuda, volatile=True, backend='pytorch')
        x_lens = np2var(
            inputs_seq_len, dtype='int', use_cuda=self.use_cuda, volatile=True, backend='pytorch')

        # Change to evaluation mode
        self.eval()

        # Encode acoustic features
        if hasattr(self, 'main_loss_weight'):
            if is_sub_task:
                _, _, logits, x_lens,  perm_idx = self._encode(
                    xs, x_lens, volatile=True, is_multi_task=True)
            else:
                logits, x_lens, _, _, perm_idx = self._encode(
                    xs, x_lens, volatile=True, is_multi_task=True)
        else:
            logits, x_lens, perm_idx = self._encode(xs, x_lens, volatile=True)

        log_probs = F.log_softmax(logits, dim=-1)

        if beam_width == 1:
            best_hyps = self._decode_greedy_np(
                var2np(log_probs, backend='pytorch'),
                var2np(x_lens, backend='pytorch'))
        else:
            best_hyps = self._decode_beam_np(
                var2np(log_probs, backend='pytorch'),
                var2np(x_lens, backend='pytorch'), beam_width=beam_width)

        # NOTE: index 0 is reserved for the blank class in warpctc_pytorch
        best_hyps = best_hyps - 1

        # Permutate indices to the original order
        if perm_idx is not None:
            perm_idx = var2np(perm_idx)
            best_hyps = best_hyps[perm_idx]

        return best_hyps

    def decode_from_probs(self, probs, inputs_seq_len, beam_width=1,
                          max_decode_len=None):
        """
        Args:
            probs (np.ndarray):
            inputs_seq_len (np.ndarray):
            beam_width (int, optional):
            max_decode_len (int, optional):
        Returns:
            best_hyps (np.ndarray):
        """
        # TODO: Subsampling

        # Convert to log-scale
        log_probs = np.log(probs + 1e-10)

        if beam_width == 1:
            best_hyps = self._decode_greedy_np(log_probs, inputs_seq_len)
        else:
            best_hyps = self._decode_beam_np(
                log_probs, inputs_seq_len, beam_width=beam_width)

        # NOTE: index 0 is reserved for the blank class in warpctc_pytorch
        best_hyps = best_hyps - 1

        return best_hyps


def _concatenate_labels(ys, y_lens):
    """Concatenate all labels in mini-batch and convert to a 1D tensor.
    Args:
        ys (LongTensor): A tensor of size `[B, T_out]`
        y_lens (IntTensor): A tensor of size `[B]`
    Returns:
        concatenated_labels (): A tensor of size `[all_label_num]`
    """
    batch_size = ys.size(0)
    total_y_lens = y_lens.data.sum()
    concatenated_labels = Variable(torch.zeros(total_y_lens)).int()
    label_counter = 0
    for i_batch in range(batch_size):
        concatenated_labels[label_counter:label_counter +
                            y_lens.data[i_batch]] = ys[i_batch][:y_lens.data[i_batch]]
        label_counter += y_lens.data[i_batch]

    return concatenated_labels
