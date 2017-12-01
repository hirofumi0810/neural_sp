#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Hierarchical attention-based sequence-to-sequence model with word-character composition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.pytorch.attention.hierarchical_attention_seq2seq import HierarchicalAttentionSeq2seq
from models.pytorch.attention.decoders.rnn_decoder import RNNDecoder
from utils.io.variable import var2np
from models.pytorch.encoders.rnn_utils import _init_hidden


class HierarchicalAttentionSeq2seqC2W(HierarchicalAttentionSeq2seq):

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
                 space_index,  # ***
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
                 scheduled_sampling_prob=0,
                 c2w_case=0):

        super(HierarchicalAttentionSeq2seqC2W, self).__init__(
            input_size=input_size,
            encoder_type=encoder_type,
            encoder_bidirectional=encoder_bidirectional,
            encoder_num_units=encoder_num_units,
            encoder_num_proj=encoder_num_proj,
            encoder_num_layers=encoder_num_layers,
            encoder_num_layers_sub=encoder_num_layers_sub,
            encoder_dropout=encoder_dropout,
            attention_type=attention_type,
            attention_dim=attention_dim,
            decoder_type=decoder_type,
            decoder_num_units=decoder_num_units,
            decoder_num_layers=decoder_num_layers,
            decoder_num_units_sub=decoder_num_units_sub,
            decoder_num_layers_sub=decoder_num_layers_sub,
            decoder_dropout=decoder_dropout,
            embedding_dim=embedding_dim,
            embedding_dim_sub=embedding_dim_sub,
            embedding_dropout=embedding_dropout,
            main_loss_weight=main_loss_weight,
            num_classes=num_classes,
            num_classes_sub=num_classes_sub,
            parameter_init=parameter_init,
            subsample_list=subsample_list,
            init_dec_state_with_enc_state=init_dec_state_with_enc_state,
            sharpening_factor=sharpening_factor,
            logits_temperature=logits_temperature,
            sigmoid_smoothing=sigmoid_smoothing,
            input_feeding=input_feeding,
            coverage_weight=coverage_weight,
            ctc_loss_weight=ctc_loss_weight,
            ctc_loss_weight_sub=ctc_loss_weight_sub,
            conv_num_channels=conv_num_channels,
            conv_width=conv_width,
            num_stack=num_stack,
            splice=splice,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            batch_norm=batch_norm,
            scheduled_sampling_prob=scheduled_sampling_prob)

        self.space_index = space_index

        # Setting for C2W
        self.c2w_case = c2w_case

        ##############################
        # Decoder in the main task
        ##############################
        self.decoder = RNNDecoder(
            embedding_dim=embedding_dim_sub * 2,
            rnn_type=decoder_type,
            num_units=decoder_num_units,
            num_layers=decoder_num_layers,
            dropout=decoder_dropout,
            parameter_init=parameter_init,
            use_cuda=self.use_cuda,
            batch_first=True)

        if c2w_case == 0:
            # Ling's LSTM-based C2W composition model
            self.c2w = nn.LSTM(
                embedding_dim_sub,
                hidden_size=256,
                num_layers=1,
                bias=True,
                batch_first=True,
                # dropout=dropout,
                bidirectional=True)

            self.c2w_fw = nn.Linear(256, embedding_dim)
            self.c2w_bw = nn.Linear(256, embedding_dim)

        elif c2w_case == 1:
            # Kim's CNN + Highway network
            pass

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
            encoder_outputs, labels, labels_sub, encoder_final_state)

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

    def _decode_train(self, encoder_outputs, labels, labels_sub,
                      encoder_final_state=None):
        """Decoding in the training stage.
        Args:
            encoder_outputs (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            labels (LongTensor): A tensor of size `[B, T_out]`
            labels_sub (LongTensor): A tensor of size `[B, T_out_sub]`
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

        for i_batch in range(batch_size):
            labels_max_seq_len = labels.size(1)
            labels_sub_max_seq_len = labels_sub.size(1)
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

            for t in range(labels_max_seq_len - 1):

                # Divide character by space
                char_embeddings = []
                while True:
                    emb_char = self.embedding_sub(
                        labels_sub[i_batch:i_batch + 1, char_counter:char_counter + 1])
                    emb_char = self.embedding_dropout_sub(emb_char)
                    # NOTE: emb_char: `[1, 1, embedding_dim_sub]`
                    char_embeddings.append(emb_char)
                    if labels_sub[i_batch, char_counter].data[0] == self.space_index:
                        break
                    if char_counter == labels_sub_max_seq_len - 1:
                        break
                    char_counter += 1
                char_embeddings = torch.cat(char_embeddings, dim=1)
                # NOTE: char_embeddings: `[1, char_num, embedding_dim_sub]`

                # Get Word representation from a sequence of character
                word_repr = self.char2word(char_embeddings)
                # NOTE: word_repr: `[1, 1, embedding_dim_sub * 2]`

                decoder_outputs, decoder_state, context_vector, attention_weights_step = self._decode_step(
                    encoder_outputs[i_batch: i_batch + 1],
                    word_repr,
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

    def char2word(self, char_embeddings, volatile=False):
        """
        Args:
            char_embeddings (FloatTensor): A tensor of size
                `[1 (batch_size), char_num, embedding_dim_sub]`
            volatile (bool, optional):
        Returns:
            word_repr (FloatTensor): A tensor of size
                `[1 (batch_size), 1, embedding_dim * 2]`
        """
        if self.c2w_case == 0:
            # Initialize hidden states (and memory cells) per mini-batch
            h_0 = _init_hidden(batch_size=1,
                               rnn_type='lstm',
                               num_units=256,
                               num_directions=2,
                               num_layers=1,
                               use_cuda=self.use_cuda,
                               volatile=volatile)

            _, (h_n, _) = self.c2w(char_embeddings, hx=h_0)
            # NOTE: h_n: `[2(num_directions), 1(batch_size), 256]`

            h_n = h_n.transpose(0, 1).contiguous()
            final_state_fw = h_n[:, 0, :]
            final_state_bw = h_n[:, 1, :]
            # NOTE: `[1, 256]`

            word_repr = self.c2w_fw(final_state_fw)
            word_repr += self.c2w_bw(final_state_bw)
            word_repr = word_repr.unsqueeze(1)

        elif self.c2w_case == 1:
            pass

        return word_repr

    def decode(self, inputs, inputs_seq_len, beam_width=1,
               max_decode_length=100):
        """Decoding in the inference stage.
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
        encoder_outputs, encoder_final_state, encoder_outputs_sub, encoder_final_state_sub, perm_indices = self._encode(
            inputs, inputs_seq_len, volatile=True)

        if beam_width == 1:
            best_hyps, _ = self._decode_infer_greedy(
                encoder_outputs, encoder_final_state,
                encoder_outputs_sub, encoder_final_state_sub,
                max_decode_length)
        else:
            raise NotImplementedError
            # best_hyps, _ = self._decode_infer_beam(
            #     encoder_outputs, encoder_final_state,
            #     inputs_seq_len[perm_indices],
            #     beam_width, max_decode_length)

        return best_hyps, perm_indices

    def _decode_infer_greedy(self, encoder_outputs, encoder_final_state,
                             encoder_outputs_sub, encoder_final_state_sub,
                             max_decode_length):
        """Greedy decoding in the inference stage.
        Args:
            encoder_outputs (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            encoder_final_state (FloatTensor): A tensor of size
                `[1, B, decoder_num_units (may be equal to encoder_num_units)]`
            encoder_outputs_sub (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units_sub]`
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
            attention_weights_step_sub = Variable(torch.zeros(1, max_time_sub))
            attention_weights_step.volatile = True
            attention_weights_step_sub.volatile = True
            if self.use_cuda:
                attention_weights_step = attention_weights_step.cuda()
                attention_weights_step_sub = attention_weights_step_sub.cuda()

            # Start from <SOS>
            y = self._create_token(value=self.sos_index, batch_size=1)
            y_sub = self._create_token(value=self.sos_index_sub, batch_size=1)

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
                    # NOTE: `[B, 1, num_classes_sub]` -> `[B, num_classes_sub]`

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
        attention_weights_step.volatile = True
        if self.use_cuda:
            attention_weights_step = attention_weights_step.cuda()

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
            # NOTE: `[B, 1, num_classes_sub]` -> `[B, num_classes_sub]`

            # Path through the softmax layer & convert to log-scale
            log_probs = F.log_softmax(logits, dim=logits.dim() - 1)

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
