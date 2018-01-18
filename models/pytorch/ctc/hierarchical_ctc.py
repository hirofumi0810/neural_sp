#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Hierarchical CTC model (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from warpctc_pytorch import CTCLoss
    ctc_loss = CTCLoss()
except:
    raise ImportError('Install warpctc_pytorch.')
# try:
#     import pytorch_ctc
# except ImportError:
#     raise ImportError('Install pytorch_ctc.')

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from models.pytorch.ctc.ctc import CTC, _concatenate_labels
from models.pytorch.linear import LinearND
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
        encoder_bidirectional (bool): if True create a bidirectional encoder
        encoder_num_units (int): the number of units in each layer
        encoder_num_proj (int): the number of nodes in recurrent projection layer
        encoder_num_layers (int): the number of layers of the encoder of the main task
        encoder_num_layers_sub (int): the number of layers of the encoder of the sub task
        fc_list (list):
        dropout (float): the probability to drop nodes
        main_loss_weight (float): A weight parameter for the main CTC loss
        num_classes (int): the number of classes of target labels of the main task
            (excluding the blank class)
        num_classes_sub (int): the number of classes of target labels of the sub task
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
                 encoder_num_layers_sub,  # ***
                 fc_list,
                 dropout,
                 main_loss_weight,  # ***
                 num_classes,
                 num_classes_sub,  # ***
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

        super(HierarchicalCTC, self).__init__(
            input_size=input_size,
            encoder_type=encoder_type,
            encoder_bidirectional=encoder_bidirectional,
            encoder_num_units=encoder_num_units,
            encoder_num_proj=encoder_num_proj,
            encoder_num_layers=encoder_num_layers,
            dropout=dropout,
            num_classes=num_classes,
            parameter_init=parameter_init,
            subsample_list=subsample_list,
            subsample_type=subsample_type,
            fc_list=fc_list,
            logits_temperature=logits_temperature,
            batch_norm=batch_norm,
            label_smoothing_prob=label_smoothing_prob,
            weight_noise_std=weight_noise_std)

        # Setting for the encoder
        self.encoder_num_layers_sub = encoder_num_layers_sub

        # Setting for CTC
        self.num_classes_sub = num_classes_sub + 1  # Add the blank class

        # Setting for MTL
        self.main_loss_weight = main_loss_weight

        # Load the encoder
        # NOTE: overide encoder
        if encoder_type in ['lstm', 'gru', 'rnn']:
            self.encoder = load(encoder_type=encoder_type)(
                input_size=input_size,
                rnn_type=encoder_type,
                bidirectional=encoder_bidirectional,
                num_units=encoder_num_units,
                num_proj=encoder_num_proj,
                num_layers=encoder_num_layers,
                num_layers_sub=encoder_num_layers_sub,
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
        else:
            raise NotImplementedError

        self.fc_sub = LinearND(
            encoder_num_units * self.num_directions, self.num_classes_sub)

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

    def forward(self, inputs, labels, labels_sub, inputs_seq_len,
                labels_seq_len, labels_seq_len_sub, is_eval=False):
        """Forward computation.
        Args:
            inputs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            labels (np.ndarray): A tensor of size `[B, T_out]`
            labels_sub (np.ndarray): A tensor of size `[B, T_out_sub]`
            inputs_seq_len (np.ndarray): A tensor of size `[B]`
            labels_seq_len (np.ndarray): A tensor of size `[B]`
            labels_seq_len_sub (np.ndarray): A tensor of size `[B]`
            is_eval (bool, optional): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (FloatTensor or float): A tensor of size `[1]`
            loss_main (FloatTensor or float): A tensor of size `[1]`
            loss_sub (FloatTensor or float): A tensor of size `[1]`
        """
        # Wrap by Variable
        xs = np2var(inputs, use_cuda=self.use_cuda, backend='pytorch')
        ys = np2var(labels, dtype='int', use_cuda=False, backend='pytorch')
        ys_sub = np2var(
            labels_sub, dtype='int', use_cuda=False, backend='pytorch')
        x_lens = np2var(
            inputs_seq_len, dtype='int', use_cuda=self.use_cuda, backend='pytorch')
        y_lens = np2var(
            labels_seq_len, dtype='int', use_cuda=False, backend='pytorch')
        y_lens_sub = np2var(
            labels_seq_len_sub, dtype='int', use_cuda=False, backend='pytorch')

        ys = ys + 1
        ys_sub = ys_sub + 1
        # NOTE: index 0 is reserved for the blank class in warpctc_pytorch

        if is_eval:
            self.eval()
        else:
            self.train()

            # Gaussian noise injection
            if self.weight_noise_injection:
                self._inject_weight_noise(mean=0, std=self.weight_noise_std)

        # Encode acoustic features
        logits, x_lens, logits_sub, x_lens_sub, perm_idx = self._encode(
            xs, x_lens, volatile=is_eval, is_multi_task=True)

        # Convert to time-major
        logits = logits.transpose(0, 1).contiguous()
        logits_sub = logits_sub.transpose(0, 1).contiguous()

        # Permutate indices
        if perm_idx is not None:
            ys = ys[perm_idx.cpu()]
            ys_sub = ys_sub[perm_idx.cpu()]
            y_lens = y_lens[perm_idx.cpu()]
            y_lens_sub = y_lens_sub[perm_idx.cpu()]

        # Concatenate all labels for warpctc_pytorch
        # `[B, T_out]` -> `[1,]`
        concatenated_labels = _concatenate_labels(ys, y_lens)
        concatenated_labels_sub = _concatenate_labels(ys_sub, y_lens_sub)

        # Output smoothing
        if self.logits_temperature != 1:
            logits = logits / self.logits_temperature
            logits_sub = logits_sub / self.logits_temperature

        ##################################################
        # Main task
        ##################################################
        # Compute CTC loss in the main task
        loss_main = ctc_loss(logits, concatenated_labels, x_lens.cpu(), y_lens)

        # Label smoothing (with uniform distribution)
        if self.label_smoothing_prob > 0:
            batch_size, label_num, num_classes = logits.size()
            log_probs = F.log_softmax(logits, dim=-1)
            uniform = Variable(torch.FloatTensor(
                batch_size, label_num, num_classes).fill_(np.log(1 / num_classes)))
            loss_main = loss_main * (1 - self.label_smoothing_prob) + self.kl_div(
                log_probs.cpu(), uniform) * self.label_smoothing_prob

        ##################################################
        # Sub task
        ##################################################
        # Compute CTC loss in the sub task
        loss_sub = ctc_loss(
            logits_sub, concatenated_labels_sub, x_lens_sub.cpu(), y_lens_sub)

        # Label smoothing (with uniform distribution)
        if self.label_smoothing_prob > 0:
            label_num_sub, num_classes_sub = logits_sub.size()[1:]
            log_probs_sub = F.log_softmax(logits_sub, dim=-1)
            uniform_sub = Variable(torch.FloatTensor(
                batch_size, label_num, num_classes_sub).fill_(np.log(1 / num_classes_sub)))
            loss_sub = loss_sub * (1 - self.label_smoothing_prob) + F.kl_div(
                log_probs_sub.cpu(), uniform_sub,
                size_average=False, reduce=True) * self.label_smoothing_prob

        # Average the loss by mini-batch
        loss_main = loss_main * self.main_loss_weight / len(xs)
        loss_sub = loss_sub * (1 - self.main_loss_weight) / len(xs)
        loss = loss_main + loss_sub

        if is_eval:
            loss = loss.data[0]
            loss_main = loss_main.data[0]
            loss_sub = loss_sub.data[0]

        return loss, loss_main, loss_sub
