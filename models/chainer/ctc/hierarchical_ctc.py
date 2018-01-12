#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Hierarchical CTC model (chainer)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from chainer import functions as F
# from models.chainer.ctc.ctc_loss_from_chainer import
# connectionist_temporal_classification as ctc

from models.chainer.ctc.ctc import CTC
from models.chainer.linear import LinearND
from models.chainer.encoders.load_encoder import load
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
                 label_smoothing_prob=0,
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
            subsample_type=subsample_type,
            fc_list=fc_list,
            logits_temperature=logits_temperature,
            batch_norm=batch_norm,
            label_smoothing_prob=label_smoothing_prob,
            weight_noise_std=weight_noise_std)

        # Setting for the encoder
        self.num_layers_sub = num_layers_sub

        # Setting for CTC
        self.num_classes_sub = num_classes_sub + 1  # Add the blank class

        # Setting for MTL
        self.main_loss_weight = main_loss_weight

        with self.init_scope():
            # Overide encoder
            delattr(self, 'encoder')

            # Load the encoder
            if encoder_type in ['lstm', 'gru', 'rnn']:
                self.encoder = load(encoder_type=encoder_type)(
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
                    subsample_type=subsample_type,
                    use_cuda=self.use_cuda,
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
                num_units * self.num_directions, self.num_classes_sub,
                dropout=dropout,
                parameter_init=parameter_init,
                use_cuda=self.use_cuda)

    def __call__(self, inputs, labels, labels_sub, inputs_seq_len,
                 labels_seq_len, labels_seq_len_sub, is_eval=False):
        """Forward computation.
        Args:
            inputs (list of np.ndarray):
                A list of tensors of size `[T_in, input_size]`
            labels (list of np.ndarray):
                A list of tensors of size `[T_out]`
            labels_sub (list of np.ndarray):
                A list of tensors of size `[T_out_sub]`
            inputs_seq_len (list of np.ndarray):
                A list of tensors of size `[1]`
            labels_seq_len (list of np.ndarray):
                A list of tensors of size `[1]`
            labels_seq_len_sub (list of np.ndarray):
                A list of tensors of size `[1]`
            is_eval (bool, optional): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            ctc_loss (chainer.Variable or float): A tensor of size `[1]`
            ctc_loss_main (chainer.Variable or float): A tensor of size `[1]`
            ctc_loss_sub (chainer.Variable or float): A tensor of size `[1]`
        """
        # Wrap by Variable
        xs = np2var(inputs, use_cuda=self.use_cuda, backend='chainer')
        ys = np2var(labels, use_cuda=self.use_cuda, backend='chainer')
        ys_sub = np2var(labels_sub, use_cuda=self.use_cuda, backend='chainer')
        x_lens = np2var(
            inputs_seq_len, use_cuda=self.use_cuda, backend='chainer')
        x_lens_sub = np2var(
            inputs_seq_len, use_cuda=self.use_cuda, backend='chainer')
        y_lens = np2var(
            labels_seq_len, use_cuda=self.use_cuda, backend='chainer')
        y_lens_sub = np2var(
            labels_seq_len_sub, use_cuda=self.use_cuda, backend='chainer')

        if is_eval:
            # TODO: add no_backprop_mode
            pass
        else:
            # TODO: Gaussian noise injection
            pass

        # Encode acoustic features
        logits, logits_sub = self._encode(xs, x_lens, is_multi_task=True)

        # Convert to time-major & list of Variable from Variable
        logits = F.separate(logits, axis=1)
        logits_sub = F.separate(logits_sub, axis=1)
        # or
        # logits = F.transpose(logits, axes=(1, 0, 2))
        # logits = [t[0] for t in F.split_axis(logits, len(logits), axis=0)]
        # logits_sub = F.transpose(logits_sub, axes=(1, 0, 2))
        # logits_sub = [t[0] for t in F.split_axis(
        #     logits_sub, len(logits_sub), axis=0)]

        # Convert to Variable from list of Variable
        ys = F.pad_sequence(ys, padding=-1024)  # 0 or -1?
        ys_sub = F.pad_sequence(ys_sub, padding=-1024)  # 0 or -1?
        ys = ys + 1
        ys_sub = ys_sub + 1
        # NOTE: index 0 is reserved for the blank class

        # Output smoothing
        if self.logits_temperature != 1:
            logits /= self.logits_temperature
            logits_sub /= self.logits_temperature

        # TODO: Modify x_lens for reducing time resolution
        # if self.encoder.conv is not None:
        #     for i in range(len(inputs_seq_len)):
        #         x_lens_sub.data[i] = self.encoder.get_conv_out_size(
        #             x_lens_sub.data[i], 1)
        #     for i in range(len(inputs_seq_len)):
        #         x_lens.data[i] = self.encoder.get_conv_out_size(
        #             x_lens.data[i], 1)
        # x_lens_sub /= 2 ** sum(self.subsample_list[:self.num_layers_sub])
        # x_lens /= 2 ** sum(self.subsample_list)
        # NOTE: floor is not needed because x_lens is IntTensor

        ##################################################
        # Main task
        ##################################################
        # Compute CTC loss in the main task
        ctc_loss_main = F.connectionist_temporal_classification(
            x=logits,  # list of Variable
            t=ys,  # Variable
            blank_symbol=self.blank_index,
            input_length=x_lens,
            label_length=y_lens,
            reduce='no')

        # TODO: Label smoothing (with uniform distribution)

        ##################################################
        # Sub task
        ##################################################
        # Compute CTC loss in the sub task
        ctc_loss_sub = F.connectionist_temporal_classification(
            x=logits_sub,  # list of Variable
            t=ys_sub,  # Variable
            blank_symbol=self.blank_index,
            input_length=x_lens_sub,
            label_length=y_lens_sub,
            reduce='no')

        # TODO: Label smoothing (with uniform distribution)

        # Total loss
        ctc_loss_main = ctc_loss_main * self.main_loss_weight
        ctc_loss_sub = ctc_loss_sub * (1 - self.main_loss_weight)
        ctc_loss = ctc_loss_main + ctc_loss_sub

        # Average the loss by mini-batch
        ctc_loss_main = F.sum(ctc_loss_main, axis=0) / len(inputs)
        ctc_loss_sub = F.sum(ctc_loss_sub, axis=0) / len(inputs)
        ctc_loss = F.sum(ctc_loss, axis=0) / len(inputs)

        if is_eval:
            ctc_loss = ctc_loss.data
            ctc_loss_main = ctc_loss_main.data
            ctc_loss_sub = ctc_loss_sub.data

        return ctc_loss, ctc_loss_main, ctc_loss_sub
