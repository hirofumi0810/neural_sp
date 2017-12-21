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
        subsample_list (list, optional): subsample in the corresponding layers (True)
            ex.) [False, True, True, False] means that subsample is conducted
                in the 2nd and 3rd layers.
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
        weight_noise_std (float, optional):
        residual (bool, optional):
        dense_residual (bool, optional):
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
                 subsample_list=[],
                 logits_temperature=1,
                 num_stack=1,
                 splice=1,
                 conv_channels=[],
                 conv_kernel_sizes=[],
                 conv_strides=[],
                 poolings=[],
                 activation='relu',
                 batch_norm=False,
                 weight_noise_std=0,
                 residual=False,
                 dense_residual=False):

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
            subsample_list=subsample_list,
            fc_list=fc_list,
            logits_temperature=logits_temperature,
            batch_norm=batch_norm,
            weight_noise_std=weight_noise_std)

        self.num_layers_sub = num_layers_sub

        # Setting for CTC
        self.num_classes_sub = num_classes_sub + 1
        # NOTE: Add blank class

        # Setting for MTL
        self.main_loss_weight = main_loss_weight

        # Load an instance
        if sum(subsample_list) == 0:
            encoder = load(encoder_type=encoder_type + '_hierarchical')
        else:
            encoder = load(encoder_type='p' + encoder_type + '_hierarchical')

        # Call the encoder function
        # NOTE: overide encoder
        if encoder_type in ['lstm', 'gru', 'rnn']:
            if sum(subsample_list) == 0:
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
                    activation=activation,
                    batch_norm=batch_norm,
                    residual=residual,
                    dense_residual=dense_residual)
            else:
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
                    subsample_list=subsample_list,
                    subsample_type='concat',
                    use_cuda=self.use_cuda,
                    batch_first=False,
                    num_stack=num_stack,
                    splice=splice,
                    conv_channels=conv_channels,
                    conv_kernel_sizes=conv_kernel_sizes,
                    conv_strides=conv_strides,
                    poolings=poolings,
                    activation=activation,
                    batch_norm=batch_norm)
        else:
            raise NotImplementedError

        self.fc_sub = LinearND(
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
        inputs_var = np2var(inputs, use_cuda=self.use_cuda)
        labels_var = np2var(labels, dtype='int', use_cuda=False)
        labels_sub_var = np2var(labels_sub, dtype='int', use_cuda=False)
        inputs_seq_len_var = np2var(
            inputs_seq_len, dtype='int', use_cuda=self.use_cuda)
        _inputs_seq_len_sub_var = inputs_seq_len_var.clone()
        labels_seq_len_var = np2var(
            labels_seq_len, dtype='int', use_cuda=False)
        labels_seq_len_sub_var = np2var(
            labels_seq_len_sub, dtype='int', use_cuda=False)

        labels_var = labels_var + 1
        labels_sub_var = labels_sub_var + 1
        # NOTE: index 0 is reserved for blank in warpctc_pytorch

        # Encode acoustic features
        logits, logits_sub, perm_indices = self._encode(
            inputs_var, inputs_seq_len_var,
            volatile=volatile, is_multi_task=True)

        # Permutate indices
        if perm_indices is not None:
            labels_var = labels_var[perm_indices.cpu()]
            labels_sub_var = labels_sub_var[perm_indices.cpu()]
            inputs_seq_len_var = inputs_seq_len_var[perm_indices]
            labels_seq_len_var = labels_seq_len_var[perm_indices.cpu()]
            labels_seq_len_sub_var = labels_seq_len_sub_var[perm_indices.cpu()]

        # Concatenate all labels for warpctc_pytorch
        # `[B, T_out]` -> `[1,]`
        concatenated_labels = _concatenate_labels(
            labels_var, labels_seq_len_var)
        concatenated_labels_sub = _concatenate_labels(
            labels_sub_var, labels_seq_len_sub_var)

        # Output smoothing
        if self.logits_temperature != 1:
            logits /= self.logits_temperature
            logits_sub /= self.logits_temperature

        # Modify inputs_seq_len for reducing time resolution
        if self.encoder.conv is not None:
            for i in range(len(inputs_seq_len)):
                _inputs_seq_len_sub_var.data[i] = self.encoder.conv_out_size(
                    _inputs_seq_len_sub_var.data[i], 1)
            for i in range(len(inputs_seq_len)):
                inputs_seq_len_var.data[i] = self.encoder.conv_out_size(
                    inputs_seq_len_var.data[i], 1)
        _inputs_seq_len_sub_var /= 2 ** sum(
            self.subsample_list[:self.num_layers_sub])
        inputs_seq_len_var /= 2 ** sum(self.subsample_list)
        # NOTE: floor is not needed because inputs_seq_len_var is IntTensor

        # Compute CTC loss
        batch_size = logits.size(1)
        ctc_loss_fn = CTCLoss()
        # Main task
        loss_main = ctc_loss_fn(logits, concatenated_labels,
                                inputs_seq_len_var.cpu(), labels_seq_len_var)
        loss_main = loss_main * self.main_loss_weight / batch_size

        # Sub task
        loss_sub = ctc_loss_fn(logits_sub, concatenated_labels_sub,
                               _inputs_seq_len_sub_var.cpu(), labels_seq_len_sub_var)
        loss_sub = loss_sub * (1 - self.main_loss_weight) / batch_size

        loss = loss_main + loss_sub

        return loss, loss_main, loss_sub
