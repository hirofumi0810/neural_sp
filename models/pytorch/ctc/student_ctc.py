#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Student CTC model (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from warpctc_pytorch import CTCLoss
except:
    raise ImportError('Install warpctc_pytorch.')

import torch.nn.functional as F

from models.pytorch.linear import LinearND
from models.pytorch.ctc.ctc import CTC, _concatenate_labels
from utils.io.variable import np2var, var2np

NEG_INF = -float("inf")
LOG_0 = NEG_INF
LOG_1 = 0


class StudentCTC(CTC):
    """Student CTC model.
    Args:
        input_size (int): the dimension of input features
        encoder_type (string): the type of the encoder. Set lstm or gru or rnn.
        bidirectional (bool): if True create a bidirectional encoder
        num_units (int): the number of units in each layer
        num_proj (int): the number of nodes in recurrent projection layer
        num_layers (int): the number of layers of the encoder of the main task
        dropout (float): the probability to drop nodes
        main_loss_weight (float): A weight parameter for the main CTC loss
        num_classes (int): the number of classes of target labels of the main task
            (excluding a blank class)
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
            Choose from relu or prelu or hard_tanh
        batch_norm (bool, optional):
        weight_noise_std (flaot, optional):
    """

    def __init__(self,
                 input_size,
                 encoder_type,
                 bidirectional,
                 num_units,
                 num_proj,
                 num_layers,
                 fc_list,
                 dropout,
                 main_loss_weight,  # ***
                 num_classes,
                 parameter_init=0.1,
                 subsample_list=[],
                 subsample_type='concat',
                 logits_temperature=1,
                 num_stack=1,
                 splice=1,
                 conv_channels=[],
                 conv_kernel_sizes=[],
                 conv_strides=[],
                 poolings=[],
                 activation='relu',
                 batch_norm=False,
                 weight_noise_std=0):

        super(StudentCTC, self).__init__(
            input_size=input_size,
            encoder_type=encoder_type,
            bidirectional=bidirectional,
            num_units=num_units,
            num_proj=num_proj,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
            parameter_init=parameter_init,
            subsample_list=subsample_list,
            subsample_type=subsample_type,
            fc_list=fc_list,
            logits_temperature=logits_temperature,
            batch_norm=batch_norm,
            weight_noise_std=weight_noise_std)

        # Weight parameter for MTL with XE loss
        self.main_loss_weight = main_loss_weight

        if len(fc_list) > 0:
            self.fc_xe = LinearND(fc_list[-1], self.num_classes)
        else:
            self.fc_xe = LinearND(
                num_units * self.num_directions, self.num_classes)

        # Initialize parameters
        self.init_weights(parameter_init)

    def forward(self, inputs, labels, labels_xe, inputs_seq_len,
                labels_seq_len, is_eval=False):
        """Forward computation.
        Args:
            inputs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            labels (np.ndarray): A tensor of size `[B, T_out]`
            labels_xe (np.ndarray): A tensor of size `[B, T_out_sub]`
            inputs_seq_len (np.ndarray): A tensor of size `[B]`
            labels_seq_len (np.ndarray): A tensor of size `[B]`
            is_eval (bool, optional): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (FloatTensor or float): A tensor of size `[1]`
            loss_main (FloatTensor or float): A tensor of size `[1]`
            loss_sub (FloatTensor or float): A tensor of size `[1]`
        """
        # Wrap by Variable
        xs = np2var(inputs, use_cuda=self.use_cuda)
        ys = np2var(labels, dtype='int', use_cuda=False)
        ys_xe = np2var(labels_xe, dtype='float', use_cuda=False)
        x_lens = np2var(inputs_seq_len, dtype='int', use_cuda=self.use_cuda)
        y_lens = np2var(labels_seq_len, dtype='int', use_cuda=False)

        # NOTE: index 0 is reserved for blank in warpctc_pytorch
        ys = ys + 1

        if is_eval:
            self.eval()
        else:
            self.train()

            # Gaussian noise injection
            if self.weight_noise_injection:
                self._inject_weight_noise(mean=0, std=self.weight_noise_std)

        # Encode acoustic features
        logits, logits_xe, x_lens, perm_idx = self._encode(
            xs, x_lens, volatile=is_eval)

        # Permutate indices
        if perm_idx is not None:
            ys = ys[perm_idx.cpu()]
            ys_xe = ys_xe[perm_idx.cpu()]
            y_lens = y_lens[perm_idx.cpu()]

        # Concatenate all labels for warpctc_pytorch
        # `[B, T_out]` -> `[1,]`
        concatenated_labels = _concatenate_labels(ys, y_lens)

        # Output smoothing
        if self.logits_temperature != 1:
            logits = logits / self.logits_temperature
            logits_xe = logits_xe / self.logits_temperature

        # Compute CTC loss (hard targets)
        ctc_loss_fn = CTCLoss()
        loss_main = ctc_loss_fn(
            logits, concatenated_labels, x_lens.cpu(), y_lens)

        # Compute XE loss (soft targets)
        loss_xe = F.cross_entropy(logits_xe, labels_xe, size_average=False)

        # TODO: label smoothing (with uniform distribution)

        # Average the loss by mini-batch
        batch_size = logits.size(1)
        loss_main = loss_main * self.main_loss_weight / batch_size
        loss_xe = loss_xe * (1 - self.main_loss_weight) / batch_size
        loss = loss_main + loss_xe

        if is_eval:
            loss = loss.data[0]
            loss_main = loss_main.data[0]
            loss_xe = loss_xe.data[0]

        return loss, loss_main, loss_xe

    def _encode(self, xs, x_lens, volatile):
        """Encode acoustic features.
        Args:
            xs (FloatTensor): A tensor of size `[B, T, input_size]`
            x_lens (IntTensor): A tensor of size `[B]`
            volatile (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            logits (FloatTensor): A tensor of size
                `[B, T, num_classes (including the blank class)]`
            logits_xe (FloatTensor): A tensor of size
                `[B, T, num_classes_sub (including the blank class)]`
            x_lens (IntTensor): A tensor of size `[B]`
            perm_idx (LongTensor):
        """
        if self.encoder_type == 'cnn':
            encoder_outputs = self.encoder(xs)
            # NOTE: `[B, T, feature_dim]`
            perm_idx = None
        else:
            encoder_outputs, x_lens, perm_idx = self.encoder(
                xs, x_lens, volatile)

        if len(self.fc_list) > 0:
            encoder_outputs = self.fc_layers(encoder_outputs)
        logits = self.fc(encoder_outputs)
        logits_xe = self.fc_xe(encoder_outputs)

        return logits, logits_xe, x_lens, perm_idx

    def decode_xe(self, inputs, inputs_seq_len, beam_width=1,
                  max_decode_len=None, ):
        """
        Args:
            inputs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            inputs_seq_len (np.ndarray): A tensor of size `[B]`
            beam_width (int, optional): the size of beam
            max_decode_len: not used (to make CTC compatible with attention)
        Returns:
            best_hyps (np.ndarray):
        """
        # Wrap by Variable
        xs = np2var(inputs, use_cuda=self.use_cuda, volatile=True)
        x_lens = np2var(
            inputs_seq_len, dtype='int', use_cuda=self.use_cuda, volatile=True)

        # Encode acoustic features
        _, logits_xe, x_lens, perm_idx = self._encode(
            xs, x_lens, volatile=True)

        log_probs_xe = F.log_softmax(logits_xe, dim=logits_xe.dim() - 1)

        if beam_width == 1:
            best_hyps = self._decode_greedy_np(
                var2np(log_probs_xe), var2np(x_lens))
        else:
            best_hyps = self._decode_beam_np(
                var2np(log_probs_xe), var2np(x_lens), beam_width=beam_width)

        best_hyps = best_hyps - 1
        # NOTE: index 0 is reserved for blank in warpctc_pytorch

        # Permutate indices to the original order
        if perm_idx is not None:
            perm_idx = var2np(perm_idx)
            best_hyps = best_hyps[perm_idx]

        return best_hyps
