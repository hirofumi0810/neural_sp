#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Hierarchical CTC model (chainer)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import chainer
from chainer import functions as F
from models.chainer.ctc.ctc_loss_from_chainer import connectionist_temporal_classification

from models.chainer.ctc.ctc import CTC
from models.chainer.linear import LinearND
from models.chainer.encoders.load_encoder import load
from models.chainer.criterion import cross_entropy_label_smoothing


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
                 encoder_dense_residual=False):

        super(HierarchicalCTC, self).__init__(
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
            splice=splice,
            input_channel=input_channel,
            conv_channels=conv_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            conv_strides=conv_strides,
            poolings=poolings,
            logits_temperature=logits_temperature,
            batch_norm=batch_norm,
            label_smoothing_prob=label_smoothing_prob,
            weight_noise_std=weight_noise_std)
        self.model_type = 'hierarchical_ctc'

        # Setting for the encoder
        self.encoder_num_layers_sub = encoder_num_layers_sub
        self.fc_list_sub = fc_list_sub

        # Setting for CTC
        self.num_classes_sub = num_classes_sub + 1  # Add the blank class

        # Setting for MTL
        self.main_loss_weight = main_loss_weight
        self.sub_loss_weight = sub_loss_weight

        with self.init_scope():
            # Overide
            delattr(self, 'encoder')

            # Load the encoder
            if encoder_type in ['lstm', 'gru', 'rnn']:
                self.encoder = load(encoder_type=encoder_type)(
                    input_size=input_size,  # 120 or 123
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
                    use_cuda=self.use_cuda,
                    merge_bidirectional=False,
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
                    use_cuda=self.use_cuda,
                    activation=activation,
                    batch_norm=batch_norm)
            else:
                raise NotImplementedError

            ##################################################
            # Fully-connected layers in the main task
            ##################################################
            # Overide
            delattr(self, 'fc_out')
            if len(fc_list) > 0:
                for i in range(len(fc_list)):
                    # Overide
                    delattr(self, 'fc_' + str(i))

                    if i == 0:
                        if encoder_type == 'cnn':
                            bottle_input_size = self.encoder.output_size
                        else:
                            bottle_input_size = self.encoder_num_units

                        # TODO: add batch norm layers

                        setattr(self, 'fc_' + str(i), LinearND(
                            bottle_input_size, fc_list[i],
                            dropout=dropout_encoder, use_cuda=self.use_cuda))
                    else:
                        # TODO: add batch norm layers

                        setattr(self, 'fc_' + str(i), LinearND(
                            fc_list[i - 1], fc_list[i],
                            dropout=dropout_encoder, use_cuda=self.use_cuda))
                # TODO: remove a bias term in the case of batch normalization

                self.fc_out = LinearND(fc_list[-1], self.num_classes,
                                       use_cuda=self.use_cuda)
            else:
                self.fc_out = LinearND(self.encoder_num_units, self.num_classes,
                                       use_cuda=self.use_cuda)

            ##################################################
            # Fully-connected layers in the sub task
            ##################################################
            if len(fc_list_sub) > 0:
                for i in range(len(fc_list_sub)):
                    if i == 0:
                        if encoder_type == 'cnn':
                            bottle_input_size = self.encoder.output_size
                        else:
                            bottle_input_size = self.encoder_num_units

                        # TODO: add batch norm layers

                        setattr(self, 'fc_sub_' + str(i), LinearND(
                            bottle_input_size, fc_list_sub[i],
                            dropout=dropout_encoder, use_cuda=self.use_cuda))
                    else:
                        # TODO: add batch norm layers

                        setattr(self, 'fc_sub_' + str(i), LinearND(
                            fc_list_sub[i - 1], fc_list_sub[i],
                            dropout=dropout_encoder, use_cuda=self.use_cuda))
                # TODO: remove a bias term in the case of batch normalization

                self.fc_out_sub = LinearND(
                    fc_list_sub[-1], self.num_classes_sub,
                    use_cuda=self.use_cuda)
            else:
                self.fc_out_sub = LinearND(
                    self.encoder_num_units, self.num_classes_sub,
                    use_cuda=self.use_cuda)

            ##################################################
            # Initialize parameters
            ##################################################
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

    def __call__(self, xs, ys, x_lens, y_lens, ys_sub, y_lens_sub, is_eval=False):
        """Forward computation.
        Args:
            xs (list of np.ndarray): A tensor of size `[B, T_in, input_size]`
            ys (np.ndarray): A tensor of size `[B, T_out]`
            x_lens (list or np.ndarray): A tensor of size `[B]`
            y_lens (np.ndarray): A tensor of size `[B]`
            ys_sub (np.ndarray): A tensor of size `[B, T_out_sub]`
            y_lens_sub (np.ndarray): A tensor of size `[B]`
            is_eval (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (chainer.Variable(float) or float): A tensor of size `[1]`
            loss_main (chainer.Variable(float) or float): A tensor of size `[1]`
            loss_sub (chainer.Variable(float) or float): A tensor of size `[1]`
        """
        if is_eval:
            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                loss, loss_main, loss_sub = self._forward(
                    xs, ys, ys_sub, x_lens, y_lens, y_lens_sub)
                loss = loss.data
                loss_main = loss_main.data
                loss_sub = loss_sub.data
        else:
            loss, loss_main, loss_sub = self._forward(
                xs, ys, ys_sub, x_lens, y_lens, y_lens_sub)
            # TODO: Gaussian noise injection

        return loss, loss_main, loss_sub

    def _forward(self, xs, ys, ys_sub, x_lens, y_lens, y_lens_sub):
        # Wrap by Variable
        xs = self.np2var(xs)
        ys = self.np2var(ys)
        ys_sub = self.np2var(ys_sub)
        y_lens = self.np2var(y_lens)
        y_lens_sub = self.np2var(y_lens_sub)

        # Encode acoustic features
        logits_main, x_lens, logits_sub, x_lens_sub = self._encode(
            xs, x_lens, is_multi_task=True)

        # Output smoothing
        if self.logits_temperature != 1:
            logits_main /= self.logits_temperature
            logits_sub /= self.logits_temperature

        ys = ys + 1
        ys_sub = ys_sub + 1
        # NOTE: index 0 is reserved for the blank class

        # Compute CTC loss in the main & sub task
        loss_main = connectionist_temporal_classification(
            x=F.separate(logits_main, axis=1),  # list of Variable
            t=ys,  # Variable
            blank_symbol=0,
            input_length=self.np2var(x_lens),  # Variable
            label_length=y_lens,  # Variable
            reduce='no')
        loss_main = F.sum(loss_main, axis=0) / len(xs)

        loss_sub = connectionist_temporal_classification(
            x=F.separate(logits_sub, axis=1),  # list of Variable
            t=ys_sub,  # Variable
            blank_symbol=0,
            input_length=self.np2var(x_lens_sub),  # Variable
            label_length=y_lens_sub,  # Variable
            reduce='no')
        loss_sub = F.sum(loss_sub, axis=0) / len(xs)

        # Label smoothing (with uniform distribution)
        if self.ls_prob > 0:
            # XE
            xe_loss_ls_main = cross_entropy_label_smoothing(
                logits_main,
                y_lens=self.np2var(x_lens),  # NOTE: CTC is frame-synchronous
                label_smoothing_prob=self.ls_prob,
                distribution='uniform',
                size_average=False) / len(xs)

            xe_loss_ls_sub = cross_entropy_label_smoothing(
                logits_sub,
                y_lens=self.np2var(x_lens_sub),
                label_smoothing_prob=self.ls_prob,
                distribution='uniform',
                size_average=False) / len(xs)

            loss_main = loss_main * (1 - self.ls_prob) + xe_loss_ls_main
            loss_sub = loss_sub * (1 - self.ls_prob) + xe_loss_ls_sub

        # Compute total loss
        loss_main = loss_main * self.main_loss_weight
        loss_sub = loss_sub * self.sub_loss_weight
        loss = loss_main + loss_sub

        return loss, loss_main, loss_sub
