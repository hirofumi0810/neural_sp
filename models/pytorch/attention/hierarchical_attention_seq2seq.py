#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Hierarchical attention-based sequence-to-sequence model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.pytorch.attention.attention_seq2seq import AttentionSeq2seq
from models.pytorch.encoders.load_encoder import load
from models.pytorch.attention.decoders.rnn_decoder import RNNDecoder
from models.pytorch.attention.attention_layer import AttentionMechanism
# from models.pytorch.ctc.ctc import _concatenate_labels
from utils.io.variable import var2np


class HierarchicalAttentionSeq2seq(AttentionSeq2seq):

    def __init__(self,
                 input_size,
                 encoder_type,
                 encoder_bidirectional,
                 encoder_num_units,
                 encoder_num_proj,
                 encoder_num_layers,
                 encoder_num_layers_sub,  # ***
                 encoder_dropout,
                 attention_type,
                 attention_dim,
                 decoder_type,
                 decoder_num_units,
                 decoder_num_layers,
                 decoder_num_units_sub,  # ***
                 decoder_num_layers_sub,  # ***
                 decoder_dropout,
                 embedding_dim,
                 embedding_dim_sub,  # ***
                 embedding_dropout,
                 main_loss_weight,  # ***
                 num_classes,
                 num_classes_sub,  # ***
                 parameter_init=0.1,
                 subsample_list=[],
                 init_dec_state_with_enc_state=True,
                 sharpening_factor=1,
                 logits_temperature=1,
                 sigmoid_smoothing=False,
                 input_feeding=False,
                 coverage_weight=0,
                 ctc_loss_weight=0,
                 ctc_loss_weight_sub=0,
                 conv_num_channels=10,
                 conv_width=101,
                 num_stack=1,
                 splice=1,
                 channels=[],
                 kernel_sizes=[],
                 strides=[],
                 batch_norm=False,
                 scheduled_sampling_prob=0):

        super(HierarchicalAttentionSeq2seq, self).__init__(
            input_size=input_size,
            encoder_type=encoder_type,
            encoder_bidirectional=encoder_bidirectional,
            encoder_num_units=encoder_num_units,
            encoder_num_proj=encoder_num_proj,
            encoder_num_layers=encoder_num_layers,
            encoder_dropout=encoder_dropout,
            attention_type=attention_type,
            attention_dim=attention_dim,
            decoder_type=decoder_type,
            decoder_num_units=decoder_num_units,
            decoder_num_layers=decoder_num_layers,
            decoder_dropout=decoder_dropout,
            embedding_dim=embedding_dim,
            embedding_dropout=embedding_dropout,
            num_classes=num_classes,
            parameter_init=parameter_init,
            subsample_list=subsample_list,
            init_dec_state_with_enc_state=init_dec_state_with_enc_state,
            sharpening_factor=sharpening_factor,
            logits_temperature=logits_temperature,
            sigmoid_smoothing=sigmoid_smoothing,
            input_feeding=input_feeding,
            coverage_weight=coverage_weight,
            ctc_loss_weight=ctc_loss_weight,
            conv_num_channels=conv_num_channels,
            conv_width=conv_width,
            num_stack=num_stack,
            splice=splice,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            batch_norm=batch_norm,
            scheduled_sampling_prob=scheduled_sampling_prob)

        # Setting for the encoder
        self.encoder_num_layers_sub = encoder_num_layers_sub

        # Setting for the decoder
        self.decoder_num_units_sub = decoder_num_units_sub
        self.decoder_num_layers_sub = decoder_num_layers_sub
        self.embedding_dim_sub = embedding_dim_sub
        self.num_classes_sub = num_classes_sub + 2
        # NOTE: add <SOS> and <EOS>
        self.sos_index_sub = num_classes_sub + 1
        self.eos_index_sub = num_classes_sub

        # Setting for MTL
        self.main_loss_weight = main_loss_weight
        self.sub_loss_weight = 1 - main_loss_weight - \
            ctc_loss_weight - ctc_loss_weight_sub
        self.ctc_loss_weight_sub = ctc_loss_weight_sub
        assert self.ctc_loss_weight * self.ctc_loss_weight_sub == 0

        #########################
        # Encoder
        # NOTE: overide encoder
        #########################
        # Load an instance
        if sum(subsample_list) == 0:
            encoder = load(encoder_type=encoder_type + '_hierarchical')
        else:
            encoder = load(encoder_type='p' + encoder_type + '_hierarchical')

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
                    num_layers_sub=encoder_num_layers_sub,
                    dropout=encoder_dropout,
                    parameter_init=parameter_init,
                    use_cuda=self.use_cuda,
                    batch_first=True,
                    merge_bidirectional=True,
                    num_stack=num_stack,
                    splice=splice,
                    channels=channels,
                    kernel_sizes=kernel_sizes,
                    strides=strides,
                    batch_norm=batch_norm)
            else:
                self.encoder = encoder(
                    input_size=input_size,   # 120 or 123
                    rnn_type=encoder_type,
                    bidirectional=encoder_bidirectional,
                    num_units=encoder_num_units,
                    num_proj=encoder_num_proj,
                    num_layers=encoder_num_layers,
                    num_layers_sub=encoder_num_layers_sub,
                    dropout=encoder_dropout,
                    parameter_init=parameter_init,
                    subsample_list=subsample_list,
                    subsample_type='concat',
                    use_cuda=self.use_cuda,
                    batch_first=True,
                    merge_bidirectional=True,
                    num_stack=num_stack,
                    splice=splice,
                    channels=channels,
                    kernel_sizes=kernel_sizes,
                    strides=strides,
                    batch_norm=batch_norm)
        else:
            raise NotImplementedError

        ##############################
        # Decoder in the sub task
        ##############################
        self.decoder_sub = RNNDecoder(
            embedding_dim=embedding_dim_sub,
            rnn_type=decoder_type,
            num_units=decoder_num_units_sub,
            num_layers=decoder_num_layers,
            dropout=decoder_dropout,
            parameter_init=parameter_init,
            use_cuda=self.use_cuda,
            batch_first=True)

        ###################################
        # Attention layer in the sub task
        ###################################
        self.attend_sub = AttentionMechanism(
            encoder_num_units=encoder_num_units,
            decoder_num_units=decoder_num_units_sub,
            attention_type=attention_type,
            attention_dim=attention_dim,
            sharpening_factor=sharpening_factor,
            sigmoid_smoothing=sigmoid_smoothing,
            out_channels=conv_num_channels,
            kernel_size=conv_width)

        ##################################################
        # Bridge layer between the encoder and decoder
        ##################################################
        if encoder_num_units != decoder_num_units_sub:
            self.bridge_sub = nn.Linear(
                encoder_num_units, decoder_num_units_sub)
        else:
            self.bridge_sub = None

        self.embedding_sub = nn.Embedding(
            self.num_classes_sub, embedding_dim_sub)
        self.embedding_dropout_sub = nn.Dropout(embedding_dropout)

        if input_feeding:
            self.input_feeding_sub = nn.Linear(
                decoder_num_units_sub * 2, decoder_num_units_sub)
            # NOTE: input-feeding approach
            self.fc_sub = nn.Linear(
                decoder_num_units_sub, self.num_classes_sub - 1)
        else:
            self.fc_sub = nn.Linear(
                decoder_num_units_sub, self.num_classes_sub - 1)
        # NOTE: <SOS> is removed because the decoder never predict <SOS> class

        if ctc_loss_weight_sub > 0:
            # self.fc_ctc = nn.Linear(
            # encoder_num_units * self.encoder_num_directions, num_classes_sub
            # + 1)
            self.fc_ctc_sub = nn.Linear(encoder_num_units, num_classes_sub + 1)

    def forward(self, inputs, labels, labels_sub, inputs_seq_len,
                labels_seq_len, labels_seq_len_sub, volatile=False):
        """Forward computation.
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            labels (LongTensor): A tensor of size `[B, T_out]`
            labels_sub (LongTensor): A tensor of size `[B, T_out_sub]`
            inputs_seq_len (IntTensor): A tensor of size `[B]`
            labels_seq_len (IntTensor): A tensor of size `[B]`
            labels_seq_len_sub (IntTensor): A tensor of size `[B]`
            volatile (bool, optional): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (FloatTensor): A tensor of size `[1]`
            loss_main (FloatTensor): A tensor of size `[1]`
            loss_sub (FloatTensor): A tensor of size `[1]`
        """
        # Encode acoustic features
        encoder_outputs, encoder_final_state, encoder_outputs_sub, encoder_final_state_sub, perm_indices = self._encode(
            inputs, inputs_seq_len, volatile=volatile)

        # Permutate indices
        labels = labels[perm_indices]
        labels_sub = labels_sub[perm_indices]
        inputs_seq_len = inputs_seq_len[perm_indices]
        labels_seq_len = labels_seq_len[perm_indices]
        labels_seq_len_sub = labels_seq_len_sub[perm_indices]

        # Teacher-forcing (main task)
        logits, attention_weights = self._decode_train(
            encoder_outputs, labels, encoder_final_state)

        # Teacher-forcing (sub task)
        logits_sub, attention_weights_sub = self._decode_train_sub(
            encoder_outputs_sub, labels_sub, encoder_final_state_sub)

        # Output smoothing
        if self.logits_temperature != 1:
            logits /= self.logits_temperature
            logits_sub /= self.logits_temperature

        batch_size = encoder_outputs.size(0)

        # Compute XE sequence loss in the main task
        num_classes = logits.size(2)
        logits = logits.view((-1, num_classes))
        labels_1d = labels[:, 1:].contiguous().view(-1)
        loss_main = F.cross_entropy(logits, labels_1d,
                                    ignore_index=self.sos_index,
                                    size_average=False)
        # NOTE: labels are padded by sos_index

        # Compute XE sequence loss in the sub task
        num_classes_sub = logits_sub.size(2)
        logits_sub = logits_sub.view((-1, num_classes_sub))
        labels_sub_1d = labels_sub[:, 1:].contiguous().view(-1)
        loss_sub = F.cross_entropy(logits_sub, labels_sub_1d,
                                   ignore_index=self.sos_index_sub,
                                   size_average=False)

        loss = loss_main * self.main_loss_weight + loss_sub * self.sub_loss_weight

        # Add coverage term
        if self.coverage_weight != 0:
            pass
            # coverage = self._compute_coverage(attention_weights)
            # loss += coverage_weight * coverage

        # Auxiliary CTC loss (optional)
        if self.ctc_loss_weight > 0:
            ctc_loss = self._compute_ctc_loss(
                encoder_outputs, labels, inputs_seq_len, labels_seq_len)
            loss += ctc_loss * self.ctc_loss_weight
        elif self.ctc_loss_weight_sub > 0:
            ctc_loss_sub = self._compute_ctc_loss(
                encoder_outputs_sub, labels_sub,
                inputs_seq_len, labels_seq_len_sub, is_sub_task=True)
            loss += ctc_loss_sub * self.ctc_loss_weight_sub

        # Average the loss by mini-batch
        loss /= batch_size

        return loss, loss_main * self.main_loss_weight / batch_size, loss_sub * self.sub_loss_weight / batch_size

    def _encode(self, inputs, inputs_seq_len, volatile):
        """Encode acoustic features.
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            inputs_seq_len (IntTensor): A tensor of size `[B]`
            volatile (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            encoder_outputs (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            encoder_final_state (FloatTensor): A tensor of size
                `[1, B, decoder_num_units (may be equal to encoder_num_units)]`
            encoder_outputs_sub (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            encoder_final_state_sub (FloatTensor): A tensor of size
                `[1, B, decoder_num_units_sub (may be equal to encoder_num_units)]`
            perm_indices_sub (FloatTensor):
        """
        encoder_outputs, encoder_final_state, encoder_outputs_sub, encoder_final_state_sub, perm_indices = self.encoder(
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

        # Bridge between the encoder and decoder in the sub task
        if self.encoder_num_units != self.decoder_num_units_sub:
            # Bridge between the encoder and decoder
            encoder_outputs_sub = self.bridge_sub(encoder_outputs_sub)
            encoder_final_state_sub = self.bridge_sub(
                encoder_final_state_sub.view(-1, self.encoder_num_units))
            encoder_final_state_sub = encoder_final_state_sub.view(
                1, batch_size, -1)

        return encoder_outputs, encoder_final_state, encoder_outputs_sub, encoder_final_state_sub, perm_indices

    def _decode_train_sub(self, encoder_outputs, labels,
                          encoder_final_state=None):
        """Decoding when training in the sub task.
        Args:
            encoder_outputs (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            labels (LongTensor): A tensor of size `[B, T_out_sub]`
            encoder_final_state (FloatTensor, optional): A tensor of size
                `[1, B, encoder_num_units]`
        Returns:
            logits (FloatTensor): A tensor of size `[B, T_out_sub, num_classes]`
            attention_weights_sub (FloatTensor): A tensor of size
                `[B, T_out_sub, T_in]`
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

            # Scheduled sampling
            if self.scheduled_sampling_prob > 0 and t > 0 and random.random() < self.scheduled_sampling_prob:
                y_prev = torch.max(logits[-1], dim=2)[1]
                y = self.embedding_sub(y_prev)
            else:
                y = self.embedding_sub(labels[:, t:t + 1])
                y = self.embedding_dropout_sub(y)

            decoder_outputs, decoder_state, context_vector, attention_weights_step = self._decode_step_sub(
                encoder_outputs,
                y,
                decoder_state,
                attention_weights_step)

            if self.input_feeding:
                # Input-feeding approach
                output = self.input_feeding_sub(
                    torch.cat([decoder_outputs, context_vector], dim=-1))
                logits_step = self.fc_sub(F.tanh(output))
            else:
                logits_step = self.fc_sub(decoder_outputs + context_vector)

            attention_weights.append(attention_weights_step)
            logits.append(logits_step)

        # Concatenate in T_out-dimension
        logits = torch.cat(logits, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)
        # NOTE; attention_weights in the training stage may be used for computing the
        # coverage, so do not convert to numpy yet.

        return logits, attention_weights

    def _decode_step_sub(self, encoder_outputs, y, decoder_state,
                         attention_weights_step):
        """
        Args:
            encoder_outputs (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            y (FloatTensor): A tensor of size `[B, 1, embedding_dim]`
            decoder_state (FloatTensor): A tensor of size
                `[decoder_num_layers_sub, B, decoder_num_units_sub]`
            attention_weights_step (FloatTensor): A tensor of size `[B, T_in]`
        Returns:
            decoder_outputs (FloatTensor): A tensor of size
                `[B, 1, decoder_num_units_sub]`
            decoder_state (FloatTensor): A tensor of size
                `[decoder_num_layers_sub, B, decoder_num_units_sub]`
            content_vector (FloatTensor): A tensor of size
                `[B, 1, encoder_num_units]`
            attention_weights_step (FloatTensor): A tensor of size `[B, T_in]`
        """
        decoder_outputs, decoder_state = self.decoder_sub(y, decoder_state)

        # decoder_outputs: `[B, 1, decoder_num_units]`
        context_vector, attention_weights_step = self.attend_sub(
            encoder_outputs,
            decoder_outputs,
            attention_weights_step)

        return decoder_outputs, decoder_state, context_vector, attention_weights_step

    def decode_sub(self, inputs, inputs_seq_len, beam_width=1,
                   max_decode_length=100):
        """
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            inputs_seq_len (IntTensor): A tensor of size `[B]`
            beam_width (int, optional): the size of beam
            max_decode_length (int, optional): the length of output sequences
                to stop prediction when EOS token have not been emitted
        Returns:
            best_hyps ():
            perm_indices ():
        """
        # Encode acoustic features
        _, _, encoder_outputs, encoder_final_state, perm_indices = self._encode(
            inputs, inputs_seq_len, volatile=True)

        if beam_width == 1:
            best_hyps, _ = self._decode_infer_greedy_sub(
                encoder_outputs, encoder_final_state, max_decode_length)
        else:
            best_hyps, _ = self._decode_infer_beam_sub(
                encoder_outputs, encoder_final_state, beam_width, max_decode_length)

        return best_hyps, perm_indices

    def _decode_infer_greedy_sub(self, encoder_outputs, encoder_final_state,
                                 max_decode_length):
        """Greedy decoding when inference.
        Args:
            encoder_outputs (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units_sub]`
            encoder_final_state (FloatTensor): A tensor of size
                `[1, B, decoder_num_units_sub (may be equal to encoder_num_units_sub)]`
            max_decode_length (int): the length of output sequences
                to stop prediction when EOS token have not been emitted
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B, T_out_sub]`
            attention_weights (np.ndarray): A tensor of size
                `[B, T_out_sub, T_in]`
        """
        batch_size, max_time = encoder_outputs.size()[:2]

        # Initialize decoder state
        decoder_state = self._init_decoder_state(
            encoder_final_state, volatile=True)

        # Initialize attention weights
        attention_weights_step = Variable(torch.zeros(batch_size, max_time))
        if self.use_cuda:
            attention_weights_step = attention_weights_step.cuda()
            # TODO: volatile, require_grad

        best_hyps = []
        attention_weights = []

        # Start from <SOS>
        y = self._create_token(value=self.sos_index_sub, batch_size=batch_size)

        for _ in range(max_decode_length):
            y = self.embedding_sub(y)
            y = self.embedding_dropout_sub(y)
            # TODO: remove dropout??

            decoder_outputs, decoder_state, context_vector, attention_weights_step = self._decode_step_sub(
                encoder_outputs,
                y,
                decoder_state,
                attention_weights_step)

            if self.input_feeding:
                # Input-feeding approach
                output = self.input_feeding_sub(
                    torch.cat([decoder_outputs, context_vector], dim=-1))
                logits = self.fc_sub(F.tanh(output))
            else:
                logits = self.fc_sub(decoder_outputs + context_vector)

            logits = logits.squeeze(dim=1)
            # NOTE: `[B, 1, num_classes]` -> `[B, num_classes]`

            # Path through the softmax layer & convert to log-scale
            log_probs = self.log_softmax(logits)

            # Pick up 1-best
            y = torch.max(log_probs, dim=1)[1]
            y = y.unsqueeze(dim=1)
            best_hyps.append(y)
            attention_weights.append(attention_weights_step)

            # Break if <EOS> is outputed in all mini-batch
            if torch.sum(y.data == self.eos_index_sub) == y.numel():
                break

        # Concatenate in T_out-dimension
        best_hyps = torch.cat(best_hyps, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)

        # Convert to numpy
        best_hyps = var2np(best_hyps)
        attention_weights = var2np(attention_weights)

        return best_hyps, attention_weights

    def _decode_infer_beam_sub(self, encoder_outputs, encoder_final_state,
                               beam_width, max_decode_length):
        raise NotImplementedError
