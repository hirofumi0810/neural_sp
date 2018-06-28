#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Hierarchical CTC model (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from src.models.pytorch_v3.ctc.ctc import CTC
from src.models.pytorch_v3.linear import LinearND
from src.models.pytorch_v3.encoders.load_encoder import load
from src.models.pytorch_v3.ctc.ctc import my_warpctc
from src.models.pytorch_v3.criterion import cross_entropy_label_smoothing
from src.models.pytorch_v3.utils import np2var
from src.utils.io.inputs.frame_stacking import stack_frame
from src.utils.io.inputs.splicing import do_splice


class HierarchicalCTC(CTC):
    """Hierarchical CTC model.
    Args:
        input_type (string): speech or text
            speech means ASR, and text means NMT or P2W and so on...
        input_size (int): the dimension of input features
        encoder_type (string): the type of the encoder. Set lstm or gru or rnn.
        encoder_bidirectional (bool): if True create a bidirectional encoder
        encoder_num_units (int): the number of units in each layer
        encoder_num_proj (int): the number of nodes in recurrent projection layer
        encoder_num_layers (int): the number of layers of the encoder of the main task
        encoder_num_layers_sub (int): the number of layers of the encoder of the sub task
        fc_list (list):
        fc_list_sub (list):
        dropout_input (float): the probability to drop nodes in input-hidden connection
        dropout_encoder (float): the probability to drop nodes in hidden-hidden connection
        main_loss_weight (float): A weight parameter for the CTC loss in the main task
        sub_loss_weight (float): A weight parameter for the CTC loss in the sub task
        num_classes (int): the number of classes of target labels of the main task
            (excluding the blank class)
        num_classes_sub (int): the number of classes of target labels of the sub task
            (excluding the blank class)
        parameter_init_distribution (string): uniform or normal or orthogonal
            or constant distribution
        parameter_init (float): Range of uniform distribution to initialize
            weight parameters
        recurrent_weight_orthogonal (bool): if True, recurrent weights are
            orthogonalized
        init_forget_gate_bias_with_one (bool): if True, initialize the forget
            gate bias with 1
        subsample_list (list): subsample in the corresponding layers (True)
            ex.) [False, True, True, False] means that subsample is conducted
                in the 2nd and 3rd layers.
        subsample_type (string): drop or concat
        logits_temperature (float):
        num_stack (int): the number of frames to stack
        num_skip (int): the number of frames to skip
        splice (int): frames to splice. Default is 1 frame.
        input_channel (int): the number of channels of input features
        conv_channels (list):
        conv_kernel_sizes (list):
        conv_strides (list):
        poolings (list):
        activation (string): The activation function of CNN layers.
            Choose from relu or prelu or hard_tanh or maxout
        batch_norm (bool):
        label_smoothing_prob (float):
        weight_noise_std (float):
        encoder_residual (bool):
        encoder_dense_residual (bool):
        num_classes_input (int):
    """

    def __init__(self,
                 input_type,
                 input_size,
                 encoder_type,
                 encoder_bidirectional,
                 encoder_num_units,
                 encoder_num_proj,
                 encoder_num_layers,
                 encoder_num_layers_sub,  # ***
                 fc_list,
                 fc_list_sub,
                 dropout_input,
                 dropout_encoder,
                 main_loss_weight,  # ***
                 sub_loss_weight,  # ***
                 num_classes,
                 num_classes_sub,  # ***
                 parameter_init_distribution='uniform',
                 parameter_init=0.1,
                 recurrent_weight_orthogonal=False,
                 init_forget_gate_bias_with_one=True,
                 subsample_list=[],
                 subsample_type='drop',
                 logits_temperature=1,
                 num_stack=1,
                 num_skip=1,
                 splice=1,
                 input_channel=1,
                 conv_channels=[],
                 conv_kernel_sizes=[],
                 conv_strides=[],
                 poolings=[],
                 activation='relu',
                 batch_norm=False,
                 label_smoothing_prob=0,
                 weight_noise_std=0,
                 encoder_residual=False,
                 encoder_dense_residual=False,
                 num_classes_input=0):

        super(HierarchicalCTC, self).__init__(
            input_type=input_type,
            input_size=input_size,
            encoder_type=encoder_type,
            encoder_bidirectional=encoder_bidirectional,
            encoder_num_units=encoder_num_units,
            encoder_num_proj=encoder_num_proj,
            encoder_num_layers=encoder_num_layers,
            dropout_input=dropout_input,
            dropout_encoder=dropout_encoder,
            num_classes=num_classes,
            parameter_init=parameter_init,
            subsample_list=subsample_list,
            subsample_type=subsample_type,
            fc_list=fc_list,
            num_stack=num_stack,
            num_skip=num_skip,
            splice=splice,
            input_channel=input_channel,
            conv_channels=conv_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            conv_strides=conv_strides,
            poolings=poolings,
            logits_temperature=logits_temperature,
            batch_norm=batch_norm,
            label_smoothing_prob=label_smoothing_prob,
            weight_noise_std=weight_noise_std,
            num_classes_input=num_classes_input)
        self.model_type = 'hierarchical_ctc'

        # Setting for the encoder
        self.encoder_num_layers_sub = encoder_num_layers_sub
        self.fc_list_sub = fc_list_sub

        # Setting for CTC
        self.num_classes_sub = num_classes_sub + 1  # Add the blank class

        # Setting for MTL
        self.main_loss_weight = main_loss_weight
        self.sub_loss_weight = sub_loss_weight

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
                dropout_input=dropout_input,
                dropout_hidden=dropout_encoder,
                subsample_list=subsample_list,
                subsample_type=subsample_type,
                batch_first=True,
                merge_bidirectional=False,
                pack_sequence=True,
                num_stack=num_stack,
                splice=splice,
                input_channel=input_channel,
                conv_channels=conv_channels,
                conv_kernel_sizes=conv_kernel_sizes,
                conv_strides=conv_strides,
                poolings=poolings,
                activation=activation,
                batch_norm=batch_norm,
                residual=encoder_residual,
                dense_residual=encoder_dense_residual)
        elif encoder_type == 'cnn':
            assert num_stack == 1 and splice == 1
            self.encoder = load(encoder_type='cnn')(
                input_size=input_size,
                input_channel=input_channel,
                conv_channels=conv_channels,
                conv_kernel_sizes=conv_kernel_sizes,
                conv_strides=conv_strides,
                poolings=poolings,
                dropout_input=dropout_input,
                dropout_hidden=dropout_encoder,
                activation=activation,
                batch_norm=batch_norm)
        else:
            raise NotImplementedError

        # Fully-connected layers in the main task
        if len(fc_list) > 0:
            for i in range(len(fc_list)):
                if i == 0:
                    if encoder_type == 'cnn':
                        bottle_input_size = self.encoder.output_size
                    else:
                        bottle_input_size = self.encoder_num_units

                    # TODO: add batch norm layers

                    setattr(self, 'fc_0', LinearND(
                        bottle_input_size, fc_list[i],
                        dropout=dropout_encoder))
                else:
                    # TODO: add batch norm layers

                    setattr(self, 'fc_' + str(i),
                            LinearND(fc_list[i - 1], fc_list[i],
                                     dropout=dropout_encoder))
            # TODO: remove a bias term in the case of batch normalization

            self.fc_out = LinearND(fc_list[-1], self.num_classes)
        else:
            self.fc_out = LinearND(self.encoder_num_units, self.num_classes)

        # Fully-connected layers in the sub task
        if len(fc_list_sub) > 0:
            for i in range(len(fc_list_sub)):
                if i == 0:
                    if encoder_type == 'cnn':
                        bottle_input_size = self.encoder.output_size
                    else:
                        bottle_input_size = self.encoder_num_units

                    # TODO: add batch norm layers

                    setattr(self, 'fc_sub_0', LinearND(
                        bottle_input_size, fc_list_sub[i],
                        dropout=dropout_encoder))
                else:
                    # TODO: add batch norm layers

                    setattr(self, 'fc_sub_' + str(i),
                            LinearND(fc_list_sub[i - 1], fc_list_sub[i],
                                     dropout=dropout_encoder))
            # TODO: remove a bias term in the case of batch normalization

            self.fc_out_sub = LinearND(fc_list_sub[-1], self.num_classes_sub)
        else:
            self.fc_out_sub = LinearND(
                self.encoder_num_units, self.num_classes_sub)

        # Initialize weight matricess
        self.init_weights(parameter_init,
                          distribution=parameter_init_distribution,
                          ignore_keys=['bias'])

        # Initialize all biases with 0
        self.init_weights(0, distribution='constant', keys=['bias'])

        # Recurrent weights are orthogonalized
        if recurrent_weight_orthogonal:
            self.init_weights(parameter_init,
                              distribution='orthogonal',
                              keys=['lstm', 'weight'],
                              ignore_keys=['bias'])

        # Initialize bias in forget gate with 1
        if init_forget_gate_bias_with_one:
            self.init_forget_gate_bias_with_one()

    def forward(self, xs, ys, ys_sub, is_eval=False):
        """Forward computation.
        Args:
            xs (list): A list of length `[B]`, which contains arrays of size `[T, input_size]`
            ys (list): A list of lenght `[B]`, which contains arrays of size `[L]`
            ys_sub (list): A list of lenght `[B]`, which contains arrays of size `[L_sub]`
            is_eval (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (torch.autograd.Variable(float)): A tensor of size `[1]`
            loss_main (float):
            loss_sub (float):
        """
        if is_eval:
            self.eval()
        else:
            self.train()

            # Gaussian noise injection
            if self.weight_noise_injection:
                self.inject_weight_noise(mean=0, std=self.weight_noise_std)

        # Sort by lenghts in the descending order
        if is_eval and self.encoder_type != 'cnn' or self.input_type == 'text':
            perm_idx = sorted(list(range(0, len(xs), 1)),
                              key=lambda i: xs[i].shape[0], reverse=True)
            xs = [xs[i] for i in perm_idx]
            ys = [ys[i] for i in perm_idx]
            ys_sub = [ys_sub[i] for i in perm_idx]
            # NOTE: must be descending order for pack_padded_sequence
            # NOTE: assumed that xs is already sorted in the training stage

        # Encode input features
        logits_main, x_lens, logits_sub, x_lens_sub = self._encode(
            xs, is_multi_task=True)

        # Output smoothing
        if self.logits_temp != 1:
            logits_main /= self.logits_temp
            logits_sub /= self.logits_temp

        # Wrap by Variable
        ys = [np2var(np.fromiter(y, dtype=np.int64), self.device_id).long()
              for y in ys]
        x_lens = np2var(np.fromiter(x_lens, dtype=np.int32), -1).int()
        y_lens = np2var(np.fromiter([y.size(0)
                                     for y in ys], dtype=np.int32), -1).int()
        ys_sub = [np2var(np.fromiter(y, dtype=np.int64), self.device_id).long()
                  for y in ys_sub]
        x_lens_sub = np2var(np.fromiter(x_lens_sub, dtype=np.int32), -1).int()
        y_lens_sub = np2var(np.fromiter([y.size(0)
                                         for y in ys_sub], dtype=np.int32), -1).int()
        # NOTE: do not copy to GPUs

        # Concatenate all elements in ys for warpctc_pytorch
        ys = torch.cat(ys, dim=0).cpu().int() + 1
        ys_sub = torch.cat(ys_sub, dim=0).cpu().int() + 1
        # NOTE: index 0 is reserved for blank in warpctc_pytorch

        # Compute CTC loss in the main & sub task
        loss_main = my_warpctc(
            logits_main.transpose(0, 1).contiguous(),  # time-major
            ys, x_lens, y_lens, size_average=False) / len(xs)
        loss_sub = my_warpctc(
            logits_sub.transpose(0, 1).contiguous(),  # time-major
            ys_sub, x_lens_sub, y_lens_sub, size_average=False) / len(xs)

        # Label smoothing (with uniform distribution)
        if self.ls_prob > 0:
            raise NotImplementedError

            if self.device_id >= 0:
                loss_main = loss_main.cuda(self.device_id)
                loss_sub = loss_sub.cuda(self.device_id)

            # loss_ls_main = cross_entropy_label_smoothing(
            #     logits_main,
            #     y_lens=x_lens,  # NOTE: CTC is frame-synchronous
            #     label_smoothing_prob=self.ls_prob,
            #     distribution='uniform',
            #     size_average=False) / len(xs)
            # loss_main = loss_main * (1 - self.ls_prob) + loss_ls_main
            #
            # loss_ls_sub = cross_entropy_label_smoothing(
            #     logits_sub,
            #     y_lens=x_lens_sub,  # NOTE: CTC is frame-synchronous
            #     label_smoothing_prob=self.ls_prob,
            #     distribution='uniform',
            #     size_average=False) / len(xs)
            # loss_sub = loss_sub * (1 - self.ls_prob) + loss_ls_sub

        # Compute total loss
        loss_main = loss_main * self.main_loss_weight
        loss_sub = loss_sub * self.sub_loss_weight
        loss = loss_main + loss_sub

        if self.device_id >= 0:
            loss = loss.cuda(self.device_id)

        return loss, loss_main.data[0], loss_sub.data[0], 0., 0.
