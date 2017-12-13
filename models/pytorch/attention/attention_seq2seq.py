#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention-based sequence-to-sequence model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from warpctc_pytorch import CTCLoss
    ctc_loss_fn = CTCLoss()
except:
    raise ImportError('Install warpctc_pytorch.')

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.pytorch.base import ModelBase
from models.pytorch.encoders.load_encoder import load
from models.pytorch.attention.decoders.rnn_decoder import RNNDecoder
from models.pytorch.attention.attention_layer import AttentionMechanism
from models.pytorch.ctc.ctc import _concatenate_labels
from models.pytorch.ctc.decoders.greedy_decoder import GreedyDecoder
from models.pytorch.ctc.decoders.beam_search_decoder import BeamSearchDecoder
from utils.io.variable import np2var, var2np


LOG_1 = 0


class AttentionSeq2seq(ModelBase):
    """The Attention-besed model.
    Args:
        input_size (int): the dimension of input features
        encoder_type (string): the type of the encoder. Set lstm or gru or rnn.
        encoder_bidirectional (bool): if True, create a bidirectional encoder
        encoder_num_units (int): the number of units in each layer of the
            encoder
        encoder_num_proj (int): the number of nodes in the projection layer of
            the encoder.
        encoder_num_layers (int): the number of layers of the encoder
        encoder_dropout (float): the probability to drop nodes of the encoder
        attention_type (string): the type of attention
        attention_dim: (int) the dimension of the attention layer
        decoder_type (string): lstm or gru
        decoder_num_units (int): the number of units in each layer of the
            decoder
        decoder_num_layers (int): the number of layers of the decoder
        decoder_dropout (float): the probability to drop nodes of the decoder
        embedding_dim (int): the dimension of the embedding in target spaces
        embedding_dropout (int): the probability to drop nodes of the
            embedding layer
        num_classes (int): the number of nodes in softmax layer
            (excluding <SOS> and <EOS> classes)
        parameter_init (float, optional): the range of uniform distribution to
            initialize weight parameters (>= 0)
        subsample_list (list, optional): subsample in the corresponding layers (True)
            ex.) [False, True, True, False] means that subsample is conducted
                in the 2nd and 3rd layers.
        init_dec_state_with_enc_state (bool, optional): if True, initialize
            decoder state with the final encoder state.
        sharpening_factor (float, optional): a sharpening factor in the
            softmax layer for computing attention weights
        logits_temperature (float, optional): a parameter for smoothing the
            softmax layer in outputing probabilities
        sigmoid_smoothing (bool, optional): if True, replace softmax function
            in computing attention weights with sigmoid function for smoothing
        input_feeding (bool, optional): See detail in
            Luong, Minh-Thang, Hieu Pham, and Christopher D. Manning.
            "Effective approaches to attention-based neural machine translation."
                arXiv preprint arXiv:1508.04025 (2015).
        coverage_weight (float, optional): the weight parameter for coverage
            computation.
        ctc_loss_weight (float): A weight parameter for auxiliary CTC loss
        attention_conv_num_channels (int, optional): the number of channles of conv
            outputs. This is used for location-based attention.
        attention_conv_width (int, optional): the size of kernel.
            This must be the odd number.
        num_stack (int, optional): the number of frames to stack
        splice (int, optional): frames to splice. Default is 1 frame.
        conv_channels (list, optional):
        conv_kernel_sizes (list, optional):
        conv_strides (list, optional):
        poolings (list, optional):
        batch_norm (bool, optional):
        scheduled_sampling_prob (float, optional):
        weight_noise_std (flaot, optional):
    """

    def __init__(self,
                 input_size,
                 encoder_type,
                 encoder_bidirectional,
                 encoder_num_units,
                 encoder_num_proj,
                 encoder_num_layers,
                 encoder_dropout,
                 attention_type,
                 attention_dim,
                 decoder_type,
                 decoder_num_units,
                 decoder_num_layers,
                 decoder_dropout,
                 embedding_dim,
                 embedding_dropout,
                 num_classes,
                 parameter_init=0.1,
                 subsample_list=[],
                 init_dec_state_with_enc_state=True,
                 sharpening_factor=1,
                 logits_temperature=1,
                 sigmoid_smoothing=False,
                 input_feeding=False,
                 coverage_weight=0,
                 ctc_loss_weight=0,
                 attention_conv_num_channels=10,
                 attention_conv_width=101,
                 num_stack=1,
                 splice=1,
                 conv_channels=[],
                 conv_kernel_sizes=[],
                 conv_strides=[],
                 poolings=[],
                 batch_norm=False,
                 scheduled_sampling_prob=0,
                 weight_noise_std=0):

        super(ModelBase, self).__init__()

        # TODO:
        # clip_activation
        # time_major option

        # Setting for the encoder
        self.encoder_type = encoder_type
        self.encoder_bidirectional = encoder_bidirectional
        self.encoder_num_directions = 2 if encoder_bidirectional else 1
        self.encoder_num_units = encoder_num_units
        self.encoder_num_proj = encoder_num_proj
        self.encoder_num_layers = encoder_num_layers
        self.subsample_list = subsample_list

        # Setting for the decoder
        self.attention_type = attention_type
        self.attention_dim = attention_dim
        self.decoder_type = decoder_type
        self.decoder_num_units = decoder_num_units
        self.decoder_num_layers = decoder_num_layers
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes + 2
        # NOTE: Add <SOS> and <EOS>
        self.sos_index = num_classes + 1
        self.eos_index = num_classes

        # Setting for the attention
        self.init_dec_state_with_enc_state = init_dec_state_with_enc_state
        self.sharpening_factor = sharpening_factor
        self.logits_temperature = logits_temperature
        self.sigmoid_smoothing = sigmoid_smoothing
        self.input_feeding = input_feeding
        self.coverage_weight = coverage_weight
        self.attention_conv_num_channels = attention_conv_num_channels
        self.attention_conv_width = attention_conv_width
        self.scheduled_sampling_prob = scheduled_sampling_prob

        # Joint CTC-Attention
        self.ctc_loss_weight = ctc_loss_weight

        # Regualarization
        self.parameter_init = parameter_init
        self.weight_noise_injection = False
        self.weight_noise_std = float(weight_noise_std)

        ####################
        # Encoder
        ####################
        # Load an instance
        if sum(subsample_list) == 0:
            encoder = load(encoder_type=encoder_type)
        else:
            encoder = load(encoder_type='p' + encoder_type)

        # Call the encoder function
        if encoder_type in ['lstm', 'gru', 'rnn']:
            if sum(subsample_list) == 0:
                self.encoder = encoder(
                    input_size=input_size,  # 120 or 123
                    rnn_type=encoder_type,
                    bidirectional=encoder_bidirectional,
                    num_units=encoder_num_units,
                    num_proj=encoder_num_proj,
                    num_layers=encoder_num_layers,
                    dropout=encoder_dropout,
                    parameter_init=parameter_init,
                    use_cuda=self.use_cuda,
                    batch_first=True,
                    merge_bidirectional=True,
                    num_stack=num_stack,
                    splice=splice,
                    conv_channels=conv_channels,
                    conv_kernel_sizes=conv_kernel_sizes,
                    conv_strides=conv_strides,
                    poolings=poolings,
                    batch_norm=batch_norm)
            else:
                # Pyramidal encoder
                self.encoder = encoder(
                    input_size=input_size,  # 120 or 123
                    rnn_type=encoder_type,
                    bidirectional=encoder_bidirectional,
                    num_units=encoder_num_units,
                    num_proj=encoder_num_proj,
                    num_layers=encoder_num_layers,
                    dropout=encoder_dropout,
                    parameter_init=parameter_init,
                    subsample_list=subsample_list,
                    subsample_type='concat',
                    use_cuda=self.use_cuda,
                    batch_first=True,
                    merge_bidirectional=True,
                    num_stack=num_stack,
                    splice=splice,
                    conv_channels=conv_channels,
                    conv_kernel_sizes=conv_kernel_sizes,
                    conv_strides=conv_strides,
                    poolings=poolings,
                    batch_norm=batch_norm)
        else:
            raise NotImplementedError

        ####################
        # Decoder
        ####################
        self.decoder = RNNDecoder(
            embedding_dim=embedding_dim,
            rnn_type=decoder_type,
            num_units=decoder_num_units,
            num_layers=decoder_num_layers,
            dropout=decoder_dropout,
            parameter_init=parameter_init,
            use_cuda=self.use_cuda,
            batch_first=True)

        ##############################
        # Attention layer
        ##############################
        self.attend = AttentionMechanism(
            decoder_num_units=decoder_num_units,
            attention_type=attention_type,
            attention_dim=attention_dim,
            sharpening_factor=sharpening_factor,
            sigmoid_smoothing=sigmoid_smoothing,
            out_channels=attention_conv_num_channels,
            kernel_size=attention_conv_width)
        # NOTE: encoder's outputs will be mapped to the same dimension as the
        # decoder states

        ##################################################
        # Bridge layer between the encoder and decoder
        ##################################################
        if encoder_num_units != decoder_num_units:
            self.bridge = nn.Linear(
                encoder_num_units, decoder_num_units)
        else:
            self.bridge = None

        self.embedding = nn.Embedding(self.num_classes, embedding_dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        if input_feeding:
            self.input_feeeding = nn.Linear(
                decoder_num_units * 2, decoder_num_units)
            # NOTE: input-feeding approach

        self.fc = nn.Linear(decoder_num_units, self.num_classes - 1)
        # NOTE: <SOS> is removed because the decoder never predict <SOS> class
        # TODO: consider projection

        if ctc_loss_weight > 0:
            # self.fc_ctc = nn.Linear(
            # encoder_num_units * self.encoder_num_directions, num_classes + 1)
            self.fc_ctc = nn.Linear(encoder_num_units, num_classes + 1)

            # Set CTC decoders
            self._decode_ctc_greedy_np = GreedyDecoder(blank_index=0)
            self._decode_ctc_beam_np = BeamSearchDecoder(blank_index=0)

    def forward(self, inputs, labels, inputs_seq_len, labels_seq_len,
                volatile=False):
        """Forward computation.
        Args:
            inputs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            labels (np.ndarray): A tensor of size `[B, T_out]`
            inputs_seq_len (np.ndarray): A tensor of size `[B]`
            labels_seq_len (np.ndarray): A tensor of size `[B]`
            volatile (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (FloatTensor): A tensor of size `[1]`
        """
        # Wrap by Variable
        inputs_var = np2var(inputs, use_cuda=self.use_cuda)
        labels_var = np2var(labels, dtype='long', use_cuda=self.use_cuda)
        # NOTE: labels_var must be long
        inputs_seq_len_var = np2var(
            inputs_seq_len, dtype='int', use_cuda=self.use_cuda)
        labels_seq_len_var = np2var(
            labels_seq_len, dtype='int', use_cuda=self.use_cuda)

        # Gaussian noise injection
        if self.weight_noise_injection:
            self._inject_weight_noise(mean=0., std=self.weight_noise_std)

        # Encode acoustic features
        encoder_outputs, encoder_final_state, perm_indices = self._encode(
            inputs_var, inputs_seq_len_var, volatile=volatile)

        # Permutate indices
        if perm_indices is not None:
            labels_var = labels_var[perm_indices]
            inputs_seq_len_var = inputs_seq_len_var[perm_indices]
            labels_seq_len_var = labels_seq_len_var[perm_indices]

        # Teacher-forcing
        logits, attention_weights = self._decode_train(
            encoder_outputs, labels_var, encoder_final_state)

        # Output smoothing
        if self.logits_temperature != 1:
            logits /= self.logits_temperature

        # Compute XE sequence loss
        num_classes = logits.size(2)
        logits = logits.view((-1, num_classes))
        labels_1d = labels_var[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(logits, labels_1d,
                               ignore_index=self.sos_index,
                               size_average=False) * (1 - self.ctc_loss_weight)
        # NOTE: labels_var are padded by sos_index

        # Add coverage term
        if self.coverage_weight != 0:
            pass
            # coverage = self._compute_coverage(attention_weights)
            # loss += coverage_weight * coverage

        # Auxiliary CTC loss (optional)
        if self.ctc_loss_weight > 0:
            ctc_loss = self._compute_ctc_loss(
                encoder_outputs, labels_var,
                inputs_seq_len_var, labels_seq_len_var)
            loss += ctc_loss * self.ctc_loss_weight

        # Average the loss by mini-batch
        batch_size = encoder_outputs.size(0)
        loss /= batch_size

        return loss

    def _compute_ctc_loss(self, encoder_outputs, labels,
                          inputs_seq_len, labels_seq_len, is_sub_task=False):
        """
        Args:
            encoder_outputs (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            labels (LongTensor): A tensor of size `[B, T_out]`
            inputs_seq_len (IntTensor): A tensor of size `[B]`
            labels_seq_len (IntTensor): A tensor of size `[B]`
            is_sub_task (bool, optional):
        Returns:
            ctc_loss (FloatTensor):
        """
        batch_size, max_time = encoder_outputs.size()[:2]

        # Convert to 2D tensor
        encoder_outputs = encoder_outputs.contiguous()
        encoder_outputs = encoder_outputs.view(batch_size, max_time, -1)
        if is_sub_task:
            logits_ctc = self.fc_ctc_sub(encoder_outputs)
        else:
            logits_ctc = self.fc_ctc(encoder_outputs)

        # Reshape back to 3D tensor
        logits_ctc = logits_ctc.view(batch_size, max_time, -1)

        # Convert to batch-major
        logits_ctc = logits_ctc.transpose(0, 1)

        inputs_seq_len = inputs_seq_len.clone()
        labels = labels.clone()[:, 1:] + 1
        labels_seq_len = labels_seq_len.clone() - 2
        # NOTE: index 0 is reserved for blank
        # NOTE: Ignore <SOS> and <EOS>

        # Concatenate all labels for warpctc_pytorch
        # `[B, T_out]` -> `[1,]`
        concatenated_labels = _concatenate_labels(
            labels, labels_seq_len)

        # Subsampling
        if is_sub_task:
            if sum(self.subsample_list[:self.encoder_num_layers_sub]) > 0:
                inputs_seq_len /= sum(
                    self.subsample_list[:self.encoder_num_layers_sub]) ** 2
        else:
            if sum(self.subsample_list) > 0:
                inputs_seq_len /= sum(self.subsample_list) ** 2
        # NOTE: floor is not needed because inputs_seq_len is IntTensor

        ctc_loss = ctc_loss_fn(logits_ctc, concatenated_labels.cpu(),
                               inputs_seq_len.cpu(), labels_seq_len.cpu())

        if self.use_cuda:
            ctc_loss = ctc_loss.cuda()

        return ctc_loss

    def _encode(self, inputs, inputs_seq_len, volatile, is_multi_task=False):
        """Encode acoustic features.
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            inputs_seq_len (IntTensor): A tensor of size `[B]`
            volatile (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
            is_multi_task (bool, optional):
        Returns:
            encoder_outputs (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            encoder_final_state (FloatTensor): A tensor of size
                `[1, B, decoder_num_units (may be equal to encoder_num_units)]`
            encoder_outputs_sub (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            encoder_final_state_sub (FloatTensor): A tensor of size
                `[1, B, decoder_num_units_sub (may be equal to encoder_num_units)]`
            perm_indices (FloatTensor):
        """
        if is_multi_task:
            encoder_outputs, encoder_final_state, encoder_outputs_sub, encoder_final_state_sub, perm_indices = self.encoder(
                inputs, inputs_seq_len, volatile, mask_sequence=True)
        else:
            encoder_outputs, encoder_final_state, perm_indices = self.encoder(
                inputs, inputs_seq_len, volatile, mask_sequence=True)
        # NOTE: encoder_outputs:
        # `[B, T_in, encoder_num_units * encoder_num_directions]`
        # encoder_final_state: `[1, B, encoder_num_units]`
        # encoder_outputs_sub: `[B, T_in, encoder_num_units * encoder_num_directions]`
        # encoder_final_state_sub: `[1, B, encoder_num_units]`

        batch_size = encoder_outputs.size(0)

        # Bridge between the encoder and decoder in the main task
        if self.encoder_num_units != self.decoder_num_units:
            # Bridge between the encoder and decoder
            encoder_outputs = self.bridge(encoder_outputs)
            encoder_final_state = self.bridge(
                encoder_final_state.view(-1, self.encoder_num_units))
            encoder_final_state = encoder_final_state.view(1, batch_size, -1)
            # TODO: add self.bridge_init

        if is_multi_task:
            # Bridge between the encoder and decoder in the sub task
            if self.encoder_num_units != self.decoder_num_units_sub:
                # Bridge between the encoder and decoder
                encoder_outputs_sub = self.bridge_sub(encoder_outputs_sub)
                encoder_final_state_sub = self.bridge_sub(
                    encoder_final_state_sub.view(-1, self.encoder_num_units))
                encoder_final_state_sub = encoder_final_state_sub.view(
                    1, batch_size, -1)
                # TODO: add self.bridge_sub_init

            return encoder_outputs, encoder_final_state, encoder_outputs_sub, encoder_final_state_sub, perm_indices

        else:
            return encoder_outputs, encoder_final_state, perm_indices

    def _compute_coverage(self, attention_weights):
        batch_size, max_time_outputs, max_time_inputs = attention_weights.size()
        raise NotImplementedError

    def _decode_train(self, encoder_outputs, labels, encoder_final_state=None,
                      is_sub_task=False):
        """Decoding in the training stage.
        Args:
            encoder_outputs (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            labels (LongTensor): A tensor of size `[B, T_out]`
            encoder_final_state (FloatTensor, optional): A tensor of size
                `[1, B, encoder_num_units]`
            is_sub_task (bool, optional):
        Returns:
            logits (FloatTensor): A tensor of size `[B, T_out, num_classes]`
            attention_weights (FloatTensor): A tensor of size
                `[B, T_out, T_in]`
        """
        batch_size, max_time = encoder_outputs.size()[:2]
        labels_max_seq_len = labels.size(1)

        # Initialize decoder state
        decoder_state = self._init_decoder_state(encoder_final_state)

        # Initialize attention weights
        attention_weights_step = Variable(torch.zeros(batch_size, max_time))
        if self.use_cuda:
            attention_weights_step = attention_weights_step.cuda()
            # TODO: volatile, require_grad

        logits = []
        attention_weights = []
        for t in range(labels_max_seq_len - 1):

            if is_sub_task:
                if self.scheduled_sampling_prob > 0 and t > 0 and random.random() < self.scheduled_sampling_prob:
                    # Scheduled sampling
                    y_prev = torch.max(logits[-1], dim=2)[1]
                    y = self.embedding_sub(y_prev)
                else:
                    y = self.embedding_sub(labels[:, t:t + 1])
                y = self.embedding_dropout_sub(y)

                decoder_outputs, decoder_state, context_vector, attention_weights_step = self._decode_step(
                    encoder_outputs,
                    y,
                    decoder_state,
                    attention_weights_step,
                    is_sub_task=True)

                if self.input_feeding:
                    # Input-feeding approach
                    output = self.input_feeding_sub(
                        torch.cat([decoder_outputs, context_vector], dim=-1))
                    logits_step = self.fc_sub(F.tanh(output))
                else:
                    logits_step = self.fc_sub(decoder_outputs + context_vector)

            else:
                if self.scheduled_sampling_prob > 0 and t > 0 and random.random() < self.scheduled_sampling_prob:
                    # Scheduled sampling
                    y_prev = torch.max(logits[-1], dim=2)[1]
                    y = self.embedding(y_prev)
                else:
                    y = self.embedding(labels[:, t:t + 1])
                y = self.embedding_dropout(y)

                decoder_outputs, decoder_state, context_vector, attention_weights_step = self._decode_step(
                    encoder_outputs,
                    y,
                    decoder_state,
                    attention_weights_step)

                if self.input_feeding:
                    # Input-feeding approach
                    output = self.input_feeeding(
                        torch.cat([decoder_outputs, context_vector], dim=-1))
                    logits_step = self.fc(F.tanh(output))
                else:
                    logits_step = self.fc(decoder_outputs + context_vector)

            attention_weights.append(attention_weights_step)
            logits.append(logits_step)

        # Concatenate in T_out-dimension
        logits = torch.cat(logits, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)
        # NOTE; attention_weights in the training stage may be used for computing the
        # coverage, so do not convert to numpy yet.

        return logits, attention_weights

    def _init_decoder_state(self, encoder_final_state, volatile=False):
        """Initialize decoder state.
        Args:
            encoder_final_state (FloatTensor): A tensor of size
                `[1, B, encoder_num_units]`
            volatile (bool, optional): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            decoder_state (FloatTensor): A tensor of size
                `[1, B, decoder_num_units]`
        """
        if self.init_dec_state_with_enc_state and encoder_final_state is None:
            raise ValueError('Set the final state of the encoder.')

        batch_size = encoder_final_state.size(1)

        h_0 = Variable(torch.zeros(
            self.decoder_num_layers, batch_size, self.decoder_num_units))

        if volatile:
            h_0.volatile = True

        if self.use_cuda:
            h_0 = h_0.cuda()

        if self.init_dec_state_with_enc_state and self.encoder_type == self.decoder_type:
            # Initialize decoder state in the first layer with
            # the final state of the top layer of the encoder (forward)
            h_0[0, :, :] = encoder_final_state.squeeze(0)

        if self.decoder_type == 'lstm':
            c_0 = Variable(torch.zeros(
                self.decoder_num_layers, batch_size, self.decoder_num_units))

            if volatile:
                c_0.volatile = True

            if self.use_cuda:
                c_0 = c_0.cuda()

            decoder_state = (h_0, c_0)
        else:
            decoder_state = h_0

        return decoder_state

    def _decode_step(self, encoder_outputs, y, decoder_state,
                     attention_weights_step, is_sub_task=False):
        """Decoding step.
        Args:
            encoder_outputs (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            y (FloatTensor): A tensor of size `[B, 1, embedding_dim]`
            decoder_state (FloatTensor): A tensor of size
                `[decoder_num_layers, B, decoder_num_units]`
            attention_weights_step (FloatTensor): A tensor of size `[B, T_in]`
            is_sub_task (bool, optional):
        Returns:
            decoder_outputs (FloatTensor): A tensor of size
                `[B, 1, decoder_num_units]`
            decoder_state (FloatTensor): A tensor of size
                `[decoder_num_layers, B, decoder_num_units]`
            content_vector (FloatTensor): A tensor of size
                `[B, 1, encoder_num_units]`
            attention_weights_step (FloatTensor): A tensor of size `[B, T_in]`
        """
        if is_sub_task:
            decoder_outputs, decoder_state = self.decoder_sub(y, decoder_state)
            context_vector, attention_weights_step = self.attend_sub(
                encoder_outputs,
                decoder_outputs,
                attention_weights_step)
        else:
            decoder_outputs, decoder_state = self.decoder(y, decoder_state)
            context_vector, attention_weights_step = self.attend(
                encoder_outputs,
                decoder_outputs,
                attention_weights_step)

        return decoder_outputs, decoder_state, context_vector, attention_weights_step

    def _create_token(self, value, batch_size):
        """Create 1 token per batch dimension.
        Args:
            value (int): the  value to pad
            batch_size (int): the size of mini-batch
        Returns:
            y (LongTensor): A tensor of size `[B, 1]`
        """
        y = np.full((batch_size, 1), fill_value=value, dtype=np.int64)
        y = torch.from_numpy(y)
        y = Variable(y, requires_grad=False)
        y.volatile = True
        if self.use_cuda:
            y = y.cuda()
        # NOTE: y: `[B, 1]`

        return y

    def decode(self, inputs, inputs_seq_len, beam_width=1,
               max_decode_length=100):
        """Decoding in the inference stage.
        Args:
            inputs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            inputs_seq_len (np.ndarray): A tensor of size `[B]`
            beam_width (int, optional): the size of beam
            max_decode_length (int, optional): the length of output sequences
                to stop prediction when EOS token have not been emitted
        Returns:
            best_hyps (np.ndarray):
            perm_indices (np.ndarray):
        """
        # Wrap by Variable
        inputs_var = np2var(inputs, use_cuda=self.use_cuda, volatile=True)
        inputs_seq_len_var = np2var(
            inputs_seq_len, dtype='int', use_cuda=self.use_cuda, volatile=True)

        # Encode acoustic features
        encoder_outputs, encoder_final_state, perm_indices = self._encode(
            inputs_var, inputs_seq_len_var, volatile=True)

        # Permutate indices
        if perm_indices is not None:
            perm_indices = var2np(perm_indices)
            inputs_seq_len_var = inputs_seq_len_var[perm_indices]

        if beam_width == 1:
            best_hyps, _ = self._decode_infer_greedy(
                encoder_outputs, encoder_final_state, max_decode_length)
        else:
            best_hyps, _ = self._decode_infer_beam(
                encoder_outputs, encoder_final_state,
                inputs_seq_len_var,
                beam_width, max_decode_length)

        # Permutate indices to the original order
        if perm_indices is not None:
            best_hyps = best_hyps[perm_indices]

        return best_hyps

    def attention_weights(self, inputs, inputs_seq_len, beam_width=1,
                          max_decode_length=100):
        """Get attention weights for visualization.
        Args:
            inputs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            inputs_seq_len (np.ndarray): A tensor of size `[B]`
            beam_width (int, optional): the size of beam
            max_decode_length (int, optional): the length of output sequences
                to stop prediction when EOS token have not been emitted
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B, T_out]`
            attention_weights (np.ndarray): A tensor of size `[B, T_out, T_in]`
        """
        # Wrap by Variable
        inputs_var = np2var(inputs, use_cuda=self.use_cuda, volatile=True)
        inputs_seq_len_var = np2var(
            inputs_seq_len, dtype='int', use_cuda=self.use_cuda, volatile=True)

        # Encode acoustic features
        encoder_outputs, encoder_final_state, perm_indices = self._encode(
            inputs_var, inputs_seq_len_var, volatile=True)

        # Permutate indices
        if perm_indices is not None:
            perm_indices = var2np(perm_indices)
            inputs_seq_len_var = inputs_seq_len_var[perm_indices]

        if beam_width == 1:
            best_hyps, attention_weights = self._decode_infer_greedy(
                encoder_outputs, encoder_final_state, max_decode_length)
        else:
            best_hyps, attention_weights = self._decode_infer_beam(
                encoder_outputs, encoder_final_state,
                inputs_seq_len_var,
                beam_width, max_decode_length)

        # Permutate indices to the original order
        if perm_indices is not None:
            best_hyps = best_hyps[perm_indices]
            attention_weights = attention_weights[perm_indices]

        return best_hyps, attention_weights

    def _decode_infer_greedy(self, encoder_outputs, encoder_final_state,
                             max_decode_length, is_sub_task=False):
        """Greedy decoding in the inference stage.
        Args:
            encoder_outputs (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            encoder_final_state (FloatTensor): A tensor of size
                `[1, B, decoder_num_units (may be equal to encoder_num_units)]`
            max_decode_length (int): the length of output sequences
                to stop prediction when EOS token have not been emitted
            is_sub_task (bool, optional):
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B, T_out]`
            attention_weights (np.ndarray): A tensor of size `[B, T_out, T_in]`
        """
        batch_size, max_time = encoder_outputs.size()[:2]

        # Initialize decoder state
        decoder_state = self._init_decoder_state(
            encoder_final_state, volatile=True)

        # Initialize attention weights
        attention_weights_step = Variable(torch.zeros(batch_size, max_time))
        attention_weights_step.volatile = True
        if self.use_cuda:
            attention_weights_step = attention_weights_step.cuda()

        best_hyps = []
        attention_weights = []

        # Start from <SOS>
        sos = self.sos_index_sub if is_sub_task else self.sos_index
        eos = self.eos_index_sub if is_sub_task else self.eos_index
        y = self._create_token(value=sos, batch_size=batch_size)

        for _ in range(max_decode_length):

            if is_sub_task:
                y = self.embedding_sub(y)
                y = self.embedding_dropout_sub(y)
                # TODO: remove dropout??

                decoder_outputs, decoder_state, context_vector, attention_weights_step = self._decode_step(
                    encoder_outputs,
                    y,
                    decoder_state,
                    attention_weights_step,
                    is_sub_task=True)

                if self.input_feeding:
                    # Input-feeding approach
                    output = self.input_feeding_sub(
                        torch.cat([decoder_outputs, context_vector], dim=-1))
                    logits = self.fc_sub(F.tanh(output))
                else:
                    logits = self.fc_sub(decoder_outputs + context_vector)

            else:
                y = self.embedding(y)
                y = self.embedding_dropout(y)
                # TODO: remove dropout??

                decoder_outputs, decoder_state, context_vector, attention_weights_step = self._decode_step(
                    encoder_outputs,
                    y,
                    decoder_state,
                    attention_weights_step)

                if self.input_feeding:
                    # Input-feeding approach
                    output = self.input_feeeding(
                        torch.cat([decoder_outputs, context_vector], dim=-1))
                    logits = self.fc(F.tanh(output))
                else:
                    logits = self.fc(decoder_outputs + context_vector)

            logits = logits.squeeze(dim=1)
            # NOTE: `[B, 1, num_classes]` -> `[B, num_classes]`

            # Path through the softmax layer & convert to log-scale
            log_probs = F.log_softmax(logits, dim=logits.dim() - 1)

            # Pick up 1-best
            y = torch.max(log_probs, dim=1)[1]
            y = y.unsqueeze(dim=1)
            best_hyps.append(y)
            attention_weights.append(attention_weights_step)

            # Break if <EOS> is outputed in all mini-batch
            if torch.sum(y.data == eos) == y.numel():
                break

        # Concatenate in T_out dimension
        best_hyps = torch.cat(best_hyps, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)

        # Convert to numpy
        best_hyps = var2np(best_hyps)
        attention_weights = var2np(attention_weights)

        return best_hyps, attention_weights

    def _decode_infer_beam(self, encoder_outputs, encoder_final_state,
                           inputs_seq_len,
                           beam_width, max_decode_length):
        """Beam search decoding in the inference stage.
        Args:
            encoder_outputs (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            encoder_final_state (FloatTensor): A tensor of size
                `[1, B, decoder_num_units (may be equal to encoder_num_units)]`
            inputs_seq_len (IntTensor): A tensor of size `[B]`
            beam_width (int): the size of beam
            max_decode_length (int, optional): the length of output sequences
                to stop prediction when EOS token have not been emitted
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B, T_out]`
            attention_weights (np.ndarray): A tensor of size `[B, T_out, T_in]`
        """
        batch_size = encoder_outputs.size(0)

        beam = []
        for i_batch in range(batch_size):
            # Initialize decoder state
            decoder_state = self._init_decoder_state(
                encoder_final_state[:, i_batch:i_batch + 1, :], volatile=True)

            # Initialize attention weights
            max_time = inputs_seq_len.data[i_batch]
            attention_weights_step = Variable(torch.zeros(1, max_time))
            attention_weights_step.volatile = True
            if self.use_cuda:
                attention_weights_step = attention_weights_step.cuda()

            beam.append([([self.sos_index], LOG_1,
                          decoder_state, attention_weights_step)])

        complete = [[]] * batch_size
        for t in range(max_decode_length):
            new_beam = [[]] * batch_size
            for i_batch in range(batch_size):
                for hyp, score, decoder_state, attention_weights_step in beam[i_batch]:
                    y_prev = hyp[-1] if t > 0 else self.sos_index
                    y = self._create_token(value=y_prev, batch_size=1)
                    y = self.embedding(y)
                    y = self.embedding_dropout(y)
                    # TODO: remove dropout??

                    max_time = inputs_seq_len.data[i_batch]
                    decoder_outputs, decoder_state, context_vector, attention_weights_step = self._decode_step(
                        encoder_outputs[i_batch:i_batch + 1, :max_time],
                        y,
                        decoder_state,
                        attention_weights_step)

                    if self.input_feeding:
                        # Input-feeding approach
                        output = self.input_feeeding(
                            torch.cat([decoder_outputs, context_vector], dim=-1))
                        logits = self.fc(F.tanh(output))
                    else:
                        logits = self.fc(decoder_outputs + context_vector)

                    logits = logits.squeeze(dim=1)
                    # NOTE: `[B, 1, num_classes]` -> `[B, num_classes]`

                    # Path through the softmax layer & convert to log-scale
                    log_probs = F.log_softmax(logits, dim=logits.dim() - 1)
                    log_probs = var2np(log_probs).tolist()[0]

                    for i, log_prob in enumerate(log_probs):
                        new_hyp = hyp + [i]
                        new_score = np.logaddexp(score, log_prob)
                        new_beam[i_batch].append(
                            (new_hyp, new_score, decoder_state,
                             attention_weights_step))
                new_beam[i_batch] = sorted(
                    new_beam[i_batch], key=lambda x: x[1], reverse=True)

                # Remove complete hypotheses
                for cand in new_beam[i_batch][:beam_width]:
                    if cand[0][-1] == self.eos_index:
                        complete[i_batch].append(cand)
                if len(complete[i_batch]) >= beam_width:
                    complete[i_batch] = complete[i_batch][:beam_width]
                    break
                beam[i_batch] = list(filter(lambda x: x[0][-1] !=
                                            self.eos_index, new_beam[i_batch]))
                beam[i_batch] = beam[i_batch][:beam_width]

        best_hyps = []
        attention_weights = [] * batch_size
        for i_batch in range(batch_size):
            complete[i_batch] = sorted(
                complete[i_batch], key=lambda x: x[1], reverse=True)
            if len(complete[i_batch]) == 0:
                complete[i_batch] = beam[i_batch]
            hyp, score, _, attention_weights_step = complete[i_batch][0]
            best_hyps.append(hyp[1:])
            # NOTE: remove <SOS>
            attention_weights.append(attention_weights_step)

        return np.array(best_hyps), np.array(attention_weights)

    def decode_ctc(self, inputs, inputs_seq_len, beam_width=1):
        """
        Args:
            inputs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            inputs_seq_len (np.ndarray): A tensor of size `[B]`
            beam_width (int, optional): the size of beam
        Returns:
            best_hyps (np.ndarray):
        """
        # Wrap by Variable
        inputs_var = np2var(inputs, use_cuda=self.use_cuda, volatile=True)
        inputs_seq_len_var = np2var(
            inputs_seq_len, dtype='int', use_cuda=self.use_cuda, volatile=True)

        assert self.ctc_loss_weight > 0

        # Encode acoustic features
        encoder_outputs, encoder_final_state, perm_indices = self._encode(
            inputs_var, inputs_seq_len_var, volatile=True)

        # Permutate indices
        if perm_indices is not None:
            perm_indices = var2np(perm_indices)
            inputs_seq_len_var = inputs_seq_len_var[perm_indices]

        # Path through the softmax layer
        batch_size, max_time = encoder_outputs.size()[:2]
        encoder_outputs = encoder_outputs.contiguous()
        encoder_outputs = encoder_outputs.view(batch_size * max_time, -1)
        logits_ctc = self.fc_ctc(encoder_outputs)
        logits_ctc = logits_ctc.view(batch_size, max_time, -1)
        log_probs = F.log_softmax(logits_ctc, dim=logits_ctc.dim() - 1)

        # Subsampling
        if sum(self.subsample_list) > 0:
            inputs_seq_len_var /= sum(self.subsample_list) ** 2
        # NOTE: floor is not needed because inputs_seq_len_var is IntTensor

        if beam_width == 1:
            best_hyps = self._decode_ctc_greedy_np(
                var2np(log_probs), var2np(inputs_seq_len_var))
        else:
            best_hyps = self._decode_ctc_beam_np(
                var2np(log_probs), var2np(inputs_seq_len_var),
                beam_width=beam_width)

        best_hyps = best_hyps - 1
        # NOTE: index 0 is reserved for blank in warpctc_pytorch

        # Permutate indices to the original order
        if perm_indices is not None:
            best_hyps = best_hyps[perm_indices]

        return best_hyps
