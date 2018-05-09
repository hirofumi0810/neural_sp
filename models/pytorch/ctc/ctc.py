#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""The Connectionist Temporal Classification model (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import warpctc_pytorch
except:
    raise ImportError('Install warpctc_pytorch.')

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.loss import _assert_no_grad

from models.pytorch.base import ModelBase
from models.pytorch.linear import LinearND
from models.pytorch.encoders.load_encoder import load
from models.pytorch.criterion import cross_entropy_label_smoothing
from models.pytorch.ctc.decoders.greedy_decoder import GreedyDecoder
from models.pytorch.ctc.decoders.beam_search_decoder import BeamSearchDecoder
# from models.pytorch.ctc.decoders.beam_search_decoder2 import BeamSearchDecoder


class _CTC(warpctc_pytorch._CTC):
    @staticmethod
    def forward(ctx, acts, labels, act_lens, label_lens, size_average=False):
        is_cuda = True if acts.is_cuda else False
        acts = acts.contiguous()
        loss_func = warpctc_pytorch.gpu_ctc if is_cuda else warpctc_pytorch.cpu_ctc
        grads = torch.zeros(acts.size()).type_as(acts)
        minibatch_size = acts.size(1)
        costs = torch.zeros(minibatch_size).cpu()
        loss_func(acts,
                  grads,
                  labels,
                  label_lens,
                  act_lens,
                  minibatch_size,
                  costs)
        if size_average:
            # Compute the avg. log-probability per frame and batch sample.
            costs = torch.FloatTensor([costs.mean()])
        else:
            costs = torch.FloatTensor([costs.sum()])
        ctx.grads = Variable(grads)
        return costs


def my_warpctc(acts, labels, act_lens, label_lens, size_average=False):
    """Chainer like CTC Loss
    acts: Tensor of (seqLength x batch x outputDim) containing output from network
    labels: 1 dimensional Tensor containing all the targets of the batch in one sequence
    act_lens: Tensor of size (batch) containing size of each output sequence from the network
    act_lens: Tensor of (batch) containing label length of each example
    """
    assert len(labels.size()) == 1  # labels must be 1 dimensional
    _assert_no_grad(labels)
    _assert_no_grad(act_lens)
    _assert_no_grad(label_lens)
    return _CTC.apply(acts, labels, act_lens, label_lens, size_average)


warpctc = warpctc_pytorch.CTCLoss()


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
        dropout_input (float): the probability to drop nodes in input-hidden connection
        dropout_encoder (float): the probability to drop nodes in hidden-hidden connection
        num_classes (int): the number of classes of target labels
            (excluding the blank class)
        parameter_init_distribution (string, optional): uniform or normal or
            orthogonal or constant distribution
        parameter_init (float, optional): Range of uniform distribution to
            initialize weight parameters
        recurrent_weight_orthogonal (bool, optional): if True, recurrent
            weights are orthogonalized
        init_forget_gate_bias_with_one (bool, optional): if True, initialize
            the forget gate bias with 1
        subsample_list (list, optional): subsample in the corresponding layers (True)
            ex.) [False, True, True, False] means that subsample is conducted
                in the 2nd and 3rd layers.
        subsample_type (string, optional): drop or concat
        logits_temperature (float):
        num_stack (int, optional): the number of frames to stack
        splice (int, optional): frames to splice. Default is 1 frame.
        input_channel (int, optional): the number of channels of input features
        conv_channels (list, optional):
        conv_kernel_sizes (list, optional):
        conv_strides (list, optional):
        poolings (list, optional):
        activation (string, optional): The activation function of CNN layers.
            Choose from relu or prelu or hard_tanh or maxout
        batch_norm (bool, optional):
        label_smoothing_prob (float, optional):
        weight_noise_std (float, optional):
        encoder_residual (bool, optional):
        encoder_dense_residual (bool, optional):
    """

    def __init__(self,
                 input_size,
                 encoder_type,
                 encoder_bidirectional,
                 encoder_num_units,
                 encoder_num_proj,
                 encoder_num_layers,
                 fc_list,
                 dropout_input,
                 dropout_encoder,
                 num_classes,
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

        super(ModelBase, self).__init__()
        self.model_type = 'ctc'

        # Setting for the encoder
        self.input_size = input_size
        self.num_stack = num_stack
        self.encoder_type = encoder_type
        self.encoder_num_units = encoder_num_units
        if encoder_bidirectional:
            self.encoder_num_units *= 2
        self.fc_list = fc_list
        self.subsample_list = subsample_list

        # Setting for CTC
        self.num_classes = num_classes + 1  # Add the blank class
        self.logits_temperature = logits_temperature

        # Setting for regualarization
        self.weight_noise_injection = False
        self.weight_noise_std = float(weight_noise_std)
        self.ls_prob = label_smoothing_prob

        # Call the encoder function
        if encoder_type in ['lstm', 'gru', 'rnn']:
            self.encoder = load(encoder_type=encoder_type)(
                input_size=input_size,
                rnn_type=encoder_type,
                bidirectional=encoder_bidirectional,
                num_units=encoder_num_units,
                num_proj=encoder_num_proj,
                num_layers=encoder_num_layers,
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
                dense_residual=encoder_dense_residual,
                nin=0)
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

        ##################################################
        # Fully-connected layers
        ##################################################
        if len(fc_list) > 0:
            for i in range(len(fc_list)):
                if i == 0:
                    if encoder_type == 'cnn':
                        bottle_input_size = self.encoder.output_size
                    else:
                        bottle_input_size = self.encoder_num_units

                    # if batch_norm:
                    #     setattr(self, 'bn_fc_0', nn.BatchNorm1d(
                    #         bottle_input_size))

                    setattr(self, 'fc_0', LinearND(
                        bottle_input_size, fc_list[i],
                        dropout=dropout_encoder))
                else:
                    # if batch_norm:
                    #     setattr(self, 'fc_bn_' + str(i),
                    #             nn.BatchNorm1d(fc_list[i - 1]))

                    setattr(self, 'fc_' + str(i), LinearND(
                        fc_list[i - 1], fc_list[i],
                        dropout=dropout_encoder))
            # TODO: remove a bias term in the case of batch normalization

            self.fc_out = LinearND(fc_list[-1], self.num_classes)
        else:
            self.fc_out = LinearND(self.encoder_num_units, self.num_classes)

        ##################################################
        # Initialize parameters
        ##################################################
        self.init_weights(parameter_init,
                          distribution=parameter_init_distribution,
                          ignore_keys=['bias'])

        # Initialize all biases with 0
        self.init_weights(0, distribution='constant', keys=['bias'])

        # Recurrent weights are orthogonalized
        if recurrent_weight_orthogonal and encoder_type != 'cnn':
            self.init_weights(parameter_init,
                              distribution='orthogonal',
                              keys=[encoder_type, 'weight'],
                              ignore_keys=['bias'])

        # Initialize bias in forget gate with 1
        if init_forget_gate_bias_with_one:
            self.init_forget_gate_bias_with_one()

        # Set CTC decoders
        self._decode_greedy_np = GreedyDecoder(blank_index=0)
        self._decode_beam_np = BeamSearchDecoder(blank_index=0)
        # NOTE: index 0 is reserved for the blank class in warpctc_pytorch
        # TODO: set space index

    def forward(self, xs, ys, x_lens, y_lens, is_eval=False):
        """Forward computation.
        Args:
            xs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            ys (np.ndarray): A tensor of size `[B, T_out]`
            x_lens (np.ndarray): A tensor of size `[B]`
            y_lens (np.ndarray): A tensor of size `[B]`
            is_eval (bool, optional): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (torch.FloatTensor or float): A tensor of size `[1]`
        """
        if is_eval:
            with torch.no_grad():
                loss = self._forward(xs, ys, x_lens, y_lens).item()
        else:
            self.train()

            # Gaussian noise injection
            if self.weight_noise_injection:
                self.inject_weight_noise(mean=0, std=self.weight_noise_std)

            loss = self._forward(xs, ys, x_lens, y_lens)

        return loss

    def _forward(self, xs, ys, x_lens, y_lens):
        # Wrap by Variable
        xs = self.np2var(xs)
        ys = self.np2var(ys, dtype='int', cpu=True)
        x_lens = self.np2var(x_lens, dtype='int')
        y_lens = self.np2var(y_lens, dtype='int', cpu=True)

        # NOTE: index 0 is reserved for the blank class in warpctc_pytorch
        ys = ys + 1

        # Encode acoustic features
        logits, x_lens, perm_idx = self._encode(xs, x_lens)

        # Output smoothing
        if self.logits_temperature != 1:
            logits /= self.logits_temperature

        # Permutate indices
        if perm_idx is not None:
            ys = ys[perm_idx.cpu()]
            y_lens = y_lens[perm_idx.cpu()]

        # Concatenate all labels for warpctc_pytorch
        # `[B, T_out]` -> `[1,]`
        concatenated_labels = _concatenate_labels(ys, y_lens)

        # Compute CTC loss
        loss = my_warpctc(logits.transpose(0, 1),  # time-major
                          concatenated_labels,
                          x_lens.cpu(),
                          y_lens,
                          size_average=False) / len(xs)

        if self.use_cuda:
            loss = loss.cuda()

        # Label smoothing (with uniform distribution)
        if self.ls_prob > 0:
            # XE
            loss_ls = cross_entropy_label_smoothing(
                logits,
                y_lens=x_lens,  # NOTE: CTC is frame-synchronous
                label_smoothing_prob=self.ls_prob,
                distribution='uniform',
                size_average=False) / len(xs)
            loss = loss * (1 - self.ls_prob) + loss_ls

        return loss

    def _encode(self, xs, x_lens, is_multi_task=False):
        """Encode acoustic features.
        Args:
            xs (torch.FloatTensor): A tensor of size `[B, T, input_size]`
            x_lens (torch.IntTensor): A tensor of size `[B]`
            is_multi_task (bool, optional): set True in MTL models
        Returns:
            logits (torch.FloatTensor): A tensor of size
                `[B, T, num_classes (including the blank class)]`
            x_lens (torch.IntTensor): A tensor of size `[B]`
            logits_sub (torch.FloatTensor): A tensor of size
                `[B, T, num_classes_sub (including the blank class)]`
            x_lens_sub (torch.IntTensor): A tensor of size `[B]`
            perm_idx (torch.LongTensor): A tensor of size `[B]`
        """
        if is_multi_task:
            if self.encoder_type == 'cnn':
                xs, x_lens = self.encoder(xs, x_lens)
                perm_idx = None
                xs_sub = xs
                x_lens_sub = x_lens
                # TODO: clone??
            else:
                xs, x_lens, xs_sub, x_lens_sub, perm_idx = self.encoder(
                    xs, x_lens, volatile=not self.training)
        else:
            if self.encoder_type == 'cnn':
                xs, x_lens = self.encoder(xs, x_lens)
                perm_idx = None
            else:
                xs, x_lens, perm_idx = self.encoder(
                    xs, x_lens, volatile=not self.training)

        # Path through fully-connected layers
        if len(self.fc_list) > 0:
            for i in range(len(self.fc_list)):
                # if self.batch_norm:
                #     xs = getattr(self, 'bn_fc_' + str(i))(xs)
                xs = getattr(self, 'fc_' + str(i))(xs)
        logits = self.fc_out(xs)

        if is_multi_task:
            # Path through fully-connected layers
            for i in range(len(self.fc_list_sub)):
                # if self.batch_norm:
                #     xs_sub = getattr(self, 'bn_fc_sub_' + str(i))(xs_sub)
                xs_sub = getattr(self, 'fc_sub_' + str(i))(xs_sub)
            logits_sub = self.fc_out_sub(xs_sub)

            return logits, x_lens, logits_sub, x_lens_sub, perm_idx
        else:
            return logits, x_lens, perm_idx

    def decode(self, xs, x_lens, beam_width=1,
               max_decode_len=None, task_index=0):
        """
        Args:
            xs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            x_lens (np.ndarray): A tensor of size `[B]`
            beam_width (int, optional): the size of beam
            max_decode_len: not used (to make CTC compatible with attention)
            task_index (bool, optional): the index of a task
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B]`
            None: this corresponds to aw in attention-based models
            perm_idx (np.ndarray): A tensor of size `[B]`
        """
        # Change to evaluation mode
        self.eval()

        # Wrap by Variable
        xs = self.np2var(xs)
        x_lens = self.np2var(x_lens, dtype='int')

        # Encode acoustic features
        if hasattr(self, 'main_loss_weight'):
            if task_index == 0:
                logits, x_lens, _, _, perm_idx = self._encode(
                    xs, x_lens, is_multi_task=True)
            elif task_index == 1:
                _, _, logits, x_lens, perm_idx = self._encode(
                    xs, x_lens, is_multi_task=True)
            else:
                raise NotImplementedError
        else:
            logits, x_lens, perm_idx = self._encode(xs, x_lens)

        if beam_width == 1:
            best_hyps = self._decode_greedy_np(
                self.var2np(logits), self.var2np(x_lens))
        else:
            best_hyps = self._decode_beam_np(
                self.var2np(F.log_softmax(logits, dim=-1)),
                self.var2np(x_lens), beam_width=beam_width)

        # NOTE: index 0 is reserved for the blank class in warpctc_pytorch
        best_hyps -= 1

        # Permutate indices to the original order
        if perm_idx is None:
            perm_idx = np.arange(0, len(xs), 1)
        else:
            perm_idx = self.var2np(perm_idx)

        return best_hyps, None, perm_idx
        # NOTE: None corresponds to aw in attention-based models

    def posteriors(self, xs, x_lens, temperature=1,
                   blank_scale=None, task_idx=0):
        """Returns CTC posteriors (after the softmax layer).
        Args:
            xs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            x_lens (np.ndarray): A tensor of size `[B]`
            temperature (float, optional): the temperature parameter for the
                softmax layer in the inference stage
            blank_scale (float, optional):
            task_idx (int, optional): the index ofta task
        Returns:
            probs (np.ndarray): A tensor of size `[B, T, num_classes]`
            x_lens (np.ndarray): A tensor of size `[B]`
            perm_idx (np.ndarray): A tensor of size `[B]`
        """
        # Change to evaluation mode
        self.eval()

        # Wrap by Variable
        xs = self.np2var(xs)
        x_lens = self.np2var(x_lens, dtype='int')

        # Encode acoustic features
        if hasattr(self, 'main_loss_weight'):
            if task_idx == 0:
                logits, x_lens, _, _, perm_idx = self._encode(
                    xs, x_lens, is_multi_task=True)
            elif task_idx == 1:
                _, _, logits, x_lens, perm_idx = self._encode(
                    xs, x_lens, is_multi_task=True)
            else:
                raise NotImplementedError
        else:
            logits, x_lens, perm_idx = self._encode(xs, x_lens)

        probs = F.softmax(logits / temperature, dim=-1)

        # Divide by blank prior
        if blank_scale is not None:
            raise NotImplementedError

        # Permutate indices to the original order
        if perm_idx is None:
            perm_idx = np.arange(0, len(xs), 1)
        else:
            perm_idx = self.var2np(perm_idx)

        return self.var2np(probs), self.var2np(x_lens), perm_idx

    def decode_from_probs(self, probs, x_lens, beam_width=1,
                          max_decode_len=None):
        """
        Args:
            probs (np.ndarray):
            x_lens (np.ndarray):
            beam_width (int, optional):
            max_decode_len (int, optional):
        Returns:
            best_hyps (np.ndarray):
        """
        # TODO: Subsampling

        # Convert to log-scale
        log_probs = np.log(probs + 1e-10)

        if beam_width == 1:
            best_hyps = self._decode_greedy_np(log_probs, x_lens)
        else:
            best_hyps = self._decode_beam_np(
                log_probs, x_lens, beam_width=beam_width)

        # NOTE: index 0 is reserved for the blank class in warpctc_pytorch
        best_hyps -= 1

        return best_hyps


def _concatenate_labels(ys, y_lens):
    """Concatenate all labels in mini-batch and convert to a 1D tensor.
    Args:
        ys (torch.IntTensor): A tensor of size `[B, T_out]`
        y_lens (torch.IntTensor): A tensor of size `[B]`
    Returns:
        concatenated_labels (torch.IntTensor): A tensor of size `[all_label_num]`
    """
    total_y_lens = y_lens.sum().item()
    concatenated_labels = torch.zeros(total_y_lens).int()
    label_counter = 0
    for b in range(ys.size(0)):
        concatenated_labels[label_counter:label_counter +
                            y_lens[b]] = ys[b, :y_lens[b]]
        label_counter += y_lens[b]
    return concatenated_labels
