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
from models.pytorch.attention.char2word import LSTMChar2Word
from utils.io.variable import np2var, var2np


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
                 ctc_loss_weight_sub=0,  # ***
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
                 composition_case=None,
                 space_index=None):

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
            attention_conv_num_channels=attention_conv_num_channels,
            attention_conv_width=attention_conv_width,
            num_stack=num_stack,
            splice=splice,
            conv_channels=conv_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            conv_strides=conv_strides,
            poolings=poolings,
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

        # Setting for composition model
        if composition_case is not None and space_index is None:
            raise ValueError
        if composition_case not in [None, 'hidden', 'embedding']:
            raise ValueError
        self.composition_case = composition_case
        self.space_index = space_index
        # NOTE: composition_case:
        # None: normal hierarchical attention model
        # hidden: leveraga character-level hidden states
        # embedding: leveraga character embeddings

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
                    conv_channels=conv_channels,
                    conv_kernel_sizes=conv_kernel_sizes,
                    conv_strides=conv_strides,
                    poolings=poolings,
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
                    conv_channels=conv_channels,
                    conv_kernel_sizes=conv_kernel_sizes,
                    conv_strides=conv_strides,
                    poolings=poolings,
                    batch_norm=batch_norm)
        else:
            raise NotImplementedError

        ##############################
        # Decoder in the main task
        ##############################
        if composition_case in [None, 'hidden']:
            decoder_input_size = embedding_dim
        elif composition_case == 'c2w_finding_function':
            decoder_input_size = embedding_dim_sub * 2
        else:
            raise NotImplementedError

        self.decoder = RNNDecoder(
            embedding_dim=decoder_input_size,
            rnn_type=decoder_type,
            num_units=decoder_num_units,
            num_layers=decoder_num_layers,
            dropout=decoder_dropout,
            parameter_init=parameter_init,
            use_cuda=self.use_cuda,
            batch_first=True)

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
            out_channels=attention_conv_num_channels,
            kernel_size=attention_conv_width)

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

        ##################################################
        # Composition layers
        ##################################################
        if composition_case == 'hidden':
            self.merge_context_vector = nn.Linear(
                encoder_num_units * 2, encoder_num_units)
        elif composition_case == 'embedding':
            self.c2w = LSTMChar2Word(
                num_units=256,
                num_layers=1,
                bidirectional=True,
                char_embedding_dim=embedding_dim_sub,
                word_embedding_dim=embedding_dim,
                use_cuda=self.use_cuda)
            self.gating_fn = nn.Linear(embedding_dim, embedding_dim)

        elif composition_case == 'hidden_embedding':
            raise NotImplementedError

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
        labels_var = np2var(labels, dtype='long', use_cuda=self.use_cuda)
        labels_sub_var = np2var(
            labels_sub, dtype='long', use_cuda=self.use_cuda)
        inputs_seq_len_var = np2var(
            inputs_seq_len, dtype='int', use_cuda=self.use_cuda)
        labels_seq_len_var = np2var(
            labels_seq_len, dtype='int', use_cuda=self.use_cuda)
        labels_seq_len_sub_var = np2var(
            labels_seq_len_sub, dtype='int', use_cuda=self.use_cuda)

        # Encode acoustic features
        encoder_outputs, encoder_final_state, encoder_outputs_sub, encoder_final_state_sub, perm_indices = self._encode(
            inputs_var, inputs_seq_len_var,
            volatile=volatile, is_multi_task=True)

        # Permutate indices
        if perm_indices is not None:
            labels_var = labels_var[perm_indices]
            labels_sub_var = labels_sub_var[perm_indices]
            inputs_seq_len_var = inputs_seq_len_var[perm_indices]
            labels_seq_len_var = labels_seq_len_var[perm_indices]
            labels_seq_len_sub_var = labels_seq_len_sub_var[perm_indices]

        # Teacher-forcing (main task)
        if self.composition_case is None:
            logits, attention_weights = self._decode_train(
                encoder_outputs, labels_var, encoder_final_state)
        else:
            logits, attention_weights = self._decode_train_composition(
                encoder_outputs, encoder_outputs_sub,
                labels_var, labels_sub_var,
                labels_seq_len_var, labels_seq_len_sub_var,
                encoder_final_state)

        # Teacher-forcing (sub task)
        logits_sub, attention_weights_sub = self._decode_train(
            encoder_outputs_sub, labels_sub_var, encoder_final_state_sub,
            is_sub_task=True)

        # Output smoothing
        if self.logits_temperature != 1:
            logits /= self.logits_temperature
            logits_sub /= self.logits_temperature

        # Compute XE sequence loss in the main task
        num_classes = logits.size(2)
        logits = logits.view((-1, num_classes))
        labels_1d = labels_var[:, 1:].contiguous().view(-1)
        loss_main = F.cross_entropy(logits, labels_1d,
                                    ignore_index=self.sos_index,
                                    size_average=False)
        # NOTE: labels_var are padded by sos_index

        # Compute XE sequence loss in the sub task
        num_classes_sub = logits_sub.size(2)
        logits_sub = logits_sub.view((-1, num_classes_sub))
        labels_sub_1d = labels_sub_var[:, 1:].contiguous().view(-1)
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
                encoder_outputs, labels, inputs_seq_len_var, labels_seq_len_var)
            loss += ctc_loss * self.ctc_loss_weight
        elif self.ctc_loss_weight_sub > 0:
            ctc_loss_sub = self._compute_ctc_loss(
                encoder_outputs_sub, labels_sub_var,
                inputs_seq_len_var, labels_seq_len_sub_var, is_sub_task=True)
            loss += ctc_loss_sub * self.ctc_loss_weight_sub

        # Average the loss by mini-batch
        batch_size = encoder_outputs.size(0)
        loss /= batch_size

        return (loss, loss_main * self.main_loss_weight / batch_size,
                loss_sub * self.sub_loss_weight / batch_size)

    def _decode_train_composition(self, encoder_outputs, encoder_outputs_sub,
                                  labels, labels_sub,
                                  labels_seq_len, labels_seq_len_sub,
                                  encoder_final_state=None):
        """Decoding of composition models in the training stage.
        Args:
            encoder_outputs (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            encoder_outputs_sub (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units_sub]`
            labels (LongTensor): A tensor of size `[B, T_out]`
            labels_sub (LongTensor): A tensor of size `[B, T_out_sub]`
            labels_seq_len (np.ndarray): A tensor of size `[B]`
            labels_seq_len_sub (np.ndarray): A tensor of size `[B]`
            encoder_final_state (FloatTensor, optional): A tensor of size
                `[1, B, encoder_num_units]`
        Returns:
            logits (FloatTensor): A tensor of size `[B, T_out, num_classes]`
            attention_weights (FloatTensor): A tensor of size
                `[B, T_out, T_in]`
        """
        batch_size = encoder_outputs.size(0)

        logits = []
        attention_weights = []

        if self.composition_case == 'hidden':

            labels_max_seq_len = labels.size(1)

            # Initialize decoder state
            decoder_state = self._init_decoder_state(encoder_final_state)

            # Initialize attention weights
            max_time = encoder_outputs.size(1)
            attention_weights_step = Variable(
                torch.zeros(batch_size, max_time))
            if self.use_cuda:
                attention_weights_step = attention_weights_step.cuda()
                # TODO: volatile, require_grad

            for t in range(labels_max_seq_len - 1):

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

                # Compute character-level context vector
                context_vector_sub = torch.sum(
                    encoder_outputs_sub *
                    attention_weights_step.unsqueeze(dim=2),
                    dim=1, keepdim=True)

                # Mix word-level and character-level context vector
                context_vector = torch.cat(
                    [context_vector, context_vector_sub], dim=context_vector.dim() - 1)
                context_vector = self.merge_context_vector(context_vector)

                if self.input_feeding:
                    # Input-feeding approach
                    output = self.input_feeeding(
                        torch.cat([decoder_outputs, context_vector], dim=-1))
                    logits_step = self.fc(F.tanh(output))
                else:
                    logits_step = self.fc(decoder_outputs + context_vector)

            else:
                raise NotImplementedError

            attention_weights.append(attention_weights_step)
            logits.append(logits_step)

        elif self.composition_case == 'embedding':

            # TODO: cache the same word

            for i_batch in range(batch_size):
                labels_seq_len_i = labels_seq_len[i_batch].data[0]
                labels_seq_len_sub_i = labels_seq_len_sub[i_batch].data[0]
                char_counter = 0

                # Initialize decoder state
                decoder_state = self._init_decoder_state(
                    encoder_final_state[:, i_batch:i_batch + 1, :])

                # Initialize attention weights
                max_time = encoder_outputs[i_batch].size(0)
                attention_weights_step = Variable(torch.zeros(1, max_time))
                if self.use_cuda:
                    attention_weights_step = attention_weights_step.cuda()
                    # TODO: volatile, require_grad

                for t in range(labels_seq_len_i - 1):

                    # Divide character by space
                    char_embeddings = []
                    while True:
                        char_embedding = self.embedding_sub(
                            labels_sub[i_batch:i_batch + 1, char_counter:char_counter + 1])
                        char_embedding = self.embedding_dropout_sub(
                            char_embedding)
                        # NOTE: char_embedding: `[1, 1, embedding_dim_sub]`
                        char_embeddings.append(char_embedding)
                        if labels_sub[i_batch, char_counter].data[0] == self.space_index:
                            # last character of the inner word
                            break
                        if char_counter == labels_seq_len_sub_i - 1:
                            # last character of the last word
                            break
                        char_counter += 1
                    char_embeddings = torch.cat(char_embeddings, dim=1)
                    # NOTE: char_embeddings: `[1, char_num, embedding_dim_sub]`

                    # Get Word representation from a sequence of character
                    word_repr = self.c2w(char_embeddings)
                    # NOTE: word_repr: `[1, 1, embedding_dim_sub * 2]`

                    if self.scheduled_sampling_prob > 0 and t > 0 and random.random() < self.scheduled_sampling_prob:
                        # Scheduled sampling
                        y_prev = torch.max(logits[-1], dim=2)[1]
                        y = self.embedding(y_prev)
                    else:
                        y = self.embedding(
                            labels[i_batch:i_batch + 1, t:t + 1])
                    y = self.embedding_dropout(y)

                    # Mix word embedding and word representation form C2W model
                    gate = F.sigmoid(self.gating_fn(y.view((-1, y.size(-1)))))
                    y = (1 - gate) * y + gate * word_repr

                    decoder_outputs, decoder_state, context_vector, attention_weights_step = self._decode_step(
                        encoder_outputs[i_batch: i_batch + 1],
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

    def decode(self, inputs, inputs_seq_len, beam_width=1,
               max_decode_length=100, is_sub_task=False):
        """Decoding in the inference stage.
        Args:
            inputs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            inputs_seq_len (np.ndarray): A tensor of size `[B]`
            beam_width (int, optional): the size of beam
            max_decode_length (int, optional): the length of output sequences
                to stop prediction when EOS token have not been emitted
            is_sub_task (bool, optional):
        Returns:
            best_hyps (np.ndarray):
        """
        # Wrap by Variable
        inputs_var = np2var(inputs, use_cuda=self.use_cuda, volatile=True)
        inputs_seq_len_var = np2var(
            inputs_seq_len, dtype='int', use_cuda=self.use_cuda, volatile=True)

        # Encode acoustic features
        if is_sub_task:
            _, _, encoder_outputs, encoder_final_state, perm_indices = self._encode(
                inputs_var, inputs_seq_len_var, volatile=True, is_multi_task=True)
        else:
            encoder_outputs, encoder_final_state, encoder_outputs_sub, encoder_final_state_sub, perm_indices = self._encode(
                inputs_var, inputs_seq_len_var, volatile=True, is_multi_task=True)

        # Permutate indices
        if perm_indices is not None:
            perm_indices = var2np(perm_indices)
            inputs_seq_len_var = inputs_seq_len_var[perm_indices]

        if beam_width == 1:
            if is_sub_task or self.composition_case is None:
                best_hyps, _ = self._decode_infer_greedy(
                    encoder_outputs, encoder_final_state, max_decode_length,
                    is_sub_task=is_sub_task)
            else:
                best_hyps, _ = self._decode_infer_greedy_composition(
                    encoder_outputs, encoder_outputs_sub,
                    encoder_final_state, encoder_final_state_sub,
                    max_decode_length)
        else:
            if is_sub_task or self.composition_case is None:
                best_hyps, _ = self._decode_infer_beam(
                    encoder_outputs, encoder_final_state,
                    inputs_seq_len_var,
                    beam_width, max_decode_length)
            else:
                raise NotImplementedError

        # Permutate indices to the original order
        if perm_indices is not None:
            best_hyps = best_hyps[perm_indices]

        return best_hyps

    def _decode_infer_greedy_composition(self, encoder_outputs, encoder_outputs_sub,
                                         encoder_final_state, encoder_final_state_sub,
                                         max_decode_length):
        """Greedy decoding in the inference stage.
        Args:
            encoder_outputs (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            encoder_outputs_sub (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units_sub]`
            encoder_final_state (FloatTensor): A tensor of size
                `[1, B, decoder_num_units (may be equal to encoder_num_units)]`
            encoder_final_state_sub (FloatTensor): A tensor of size
                `[1, B, decoder_num_units_sub (may be equal to encoder_num_units_sub)]`
            max_decode_length (int): the length of output sequences
                to stop prediction when EOS token have not been emitted
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B, T_out]`
            attention_weights (np.ndarray): A tensor of size `[B, T_out, T_in]`
        """
        batch_size = encoder_outputs.size(0)

        best_hyps = []
        attention_weights = []

        if self.composition_case == 'hidden':

            # Initialize decoder state
            decoder_state = self._init_decoder_state(
                encoder_final_state, volatile=True)

            # Initialize attention weights
            max_time = encoder_outputs.size(1)
            attention_weights_step = Variable(
                torch.zeros(batch_size, max_time))
            attention_weights_step.volatile = True
            if self.use_cuda:
                attention_weights_step = attention_weights_step.cuda()

            # Start from <SOS>
            y = self._create_token(value=self.sos_index, batch_size=batch_size)

            for _ in range(max_decode_length):

                y = self.embedding(y)
                y = self.embedding_dropout(y)
                # TODO: remove dropout??

                decoder_outputs, decoder_state, context_vector, attention_weights_step = self._decode_step(
                    encoder_outputs,
                    y,
                    decoder_state,
                    attention_weights_step)

                # Compute character-level context vector
                context_vector_sub = torch.sum(
                    encoder_outputs_sub *
                    attention_weights_step.unsqueeze(dim=2),
                    dim=1, keepdim=True)

                # Mix word-level and character-level context vector
                context_vector = torch.cat(
                    [context_vector, context_vector_sub], dim=context_vector.dim() - 1)
                context_vector = self.merge_context_vector(context_vector)

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
                if torch.sum(y.data == self.eos_index) == y.numel():
                    break

        elif self.composition_case == 'embedding_case':

            # TODO:
            # charの出力結果を各時刻に使っていくならば，単語の数が一致していなければならない．
            # これはresolving OOVではなかった問題
            # これを解決するため，限られた領域のみデコードするみたいなテクニックが必要
            # または，文字列などの離散的な情報ではなく，デコードする前の連続量の情報を使うなどの工夫が必要

            # Proccess per utterance
            for i_batch in range(batch_size):
                max_time = encoder_outputs.size(1)
                max_time_sub = encoder_outputs_sub.size(1)

                # Initialize decoder state
                decoder_state = self._init_decoder_state(
                    encoder_final_state[:, i_batch:i_batch + 1], volatile=True)
                decoder_state_sub = self._init_decoder_state(
                    encoder_final_state_sub[:, i_batch:i_batch + 1], volatile=True)

                # Initialize attention weights
                attention_weights_step = Variable(torch.zeros(1, max_time))
                attention_weights_step_sub = Variable(
                    torch.zeros(1, max_time_sub))
                attention_weights_step.volatile = True
                attention_weights_step_sub.volatile = True
                if self.use_cuda:
                    attention_weights_step = attention_weights_step.cuda()
                    attention_weights_step_sub = attention_weights_step_sub.cuda()

                # Start from <SOS>
                y = self._create_token(value=self.sos_index, batch_size=1)
                y_sub = self._create_token(
                    value=self.sos_index_sub, batch_size=1)

                end_flag = False
                for _ in range(max_decode_length):

                    # Decode the character sequence until space is emitted
                    char_counter = 0
                    total_char_counter = 0
                    char_embeddings = []
                    while True:
                        y_sub = self.embedding_sub(y_sub)
                        y_sub = self.embedding_dropout_sub(y_sub)
                        # TODO: remove dropout??

                        decoder_outputs_sub, decoder_state_sub, context_vector_sub, attention_weights_step_sub = self._decode_step_sub(
                            encoder_outputs_sub[i_batch:i_batch +
                                                1, :max_time_sub],
                            y_sub,
                            decoder_state_sub,
                            attention_weights_step_sub)

                        if self.input_feeding:
                            # Input-feeding approach
                            output_sub = self.input_feeding_sub(
                                torch.cat([decoder_outputs_sub, context_vector_sub], dim=-1))
                            logits_sub = self.fc_sub(F.tanh(output_sub))
                        else:
                            logits_sub = self.fc_sub(
                                decoder_outputs_sub + context_vector_sub)

                        logits_sub = logits_sub.squeeze(dim=1)
                        # NOTE: `[B, 1, num_classes_sub]` -> `[B,
                        # num_classes_sub]`

                        # Path through the softmax layer & convert to log-scale
                        log_probs_sub = F.log_softmax(
                            logits_sub, dim=logits_sub.dim() - 1)

                        # Pick up 1-best
                        y_sub = torch.max(log_probs_sub, dim=1)[1]
                        print(y_sub.data[0])
                        y_sub = y_sub.unsqueeze(dim=1)

                        if y_sub.data[0] == self.eos_index_sub:
                            end_flag = True
                            break
                        if y_sub.data[0] == self.space_index:
                            break

                        emb_char = self.embedding_sub(y_sub)
                        emb_char = self.embedding_dropout_sub(emb_char)
                        # NOTE: emb_char: `[1, 1, embedding_dim_sub]`
                        char_embeddings.append(emb_char)
                        char_counter += 1
                        total_char_counter += 1

                        if total_char_counter >= 60:
                            end_flag = True
                            break
                        if char_counter > 10:
                            break

                    if len(char_embeddings) == 0:
                        break

                    char_embeddings = torch.cat(char_embeddings, dim=1)
                    # NOTE: char_embeddings: `[1, char_num, embedding_dim_sub]`

                    # Get Word representation from a sequence of character
                    word_repr = self.char2word(char_embeddings, volatile=True)
                    # NOTE: word_repr: `[1, 1, embedding_dim_sub * 2]`

                    decoder_outputs, decoder_state, context_vector, attention_weights_step = self._decode_step(
                        encoder_outputs[i_batch: i_batch + 1, :max_time],
                        word_repr,
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

                    # Break if <EOS> is outputed in the character-level decoder or
                    # length of character sequence exceed a threshold
                    if end_flag:
                        break

                    # Break if <EOS> is outputed in all mini-batch
                    if torch.sum(y.data == self.eos_index) == y.numel():
                        break

        # Concatenate in T_out dimension
        best_hyps = torch.cat(best_hyps, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)

        # Convert to numpy
        best_hyps = var2np(best_hyps)
        attention_weights = var2np(attention_weights)

        return best_hyps, attention_weights
