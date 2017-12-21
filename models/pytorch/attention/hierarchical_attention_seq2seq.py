#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Hierarchical attention-based sequence-to-sequence model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.pytorch.linear import LinearND
from models.pytorch.attention.attention_seq2seq import AttentionSeq2seq
from models.pytorch.encoders.load_encoder import load
from models.pytorch.attention.decoders.rnn_decoder import RNNDecoder
from models.pytorch.attention.attention_layer import AttentionMechanism
from models.pytorch.ctc.decoders.greedy_decoder import GreedyDecoder
from models.pytorch.ctc.decoders.beam_search_decoder import BeamSearchDecoder
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
                 main_loss_weight,  # ***
                 num_classes,
                 num_classes_sub,  # ***
                 parameter_init=0.1,
                 subsample_list=[],
                 init_dec_state_with_enc_state=True,
                 sharpening_factor=1,
                 logits_temperature=1,
                 sigmoid_smoothing=False,
                 coverage_weight=0,
                 ctc_loss_weight_sub=0,  # ***
                 attention_conv_num_channels=10,
                 attention_conv_width=101,
                 num_stack=1,
                 splice=1,
                 conv_channels=[],
                 conv_kernel_sizes=[],
                 conv_strides=[],
                 poolings=[],
                 activation='relu',
                 batch_norm=False,
                 scheduled_sampling_prob=0,
                 scheduled_sampling_ramp_max_step=0,
                 label_smoothing_prob=0,
                 weight_noise_std=0,
                 encoder_residual=False,
                 encoder_dense_residual=False,
                 decoder_residual=False,
                 decoder_dense_residual=False):

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
            num_classes=num_classes,
            parameter_init=parameter_init,
            subsample_list=subsample_list,
            init_dec_state_with_enc_state=init_dec_state_with_enc_state,
            sharpening_factor=sharpening_factor,
            logits_temperature=logits_temperature,
            sigmoid_smoothing=sigmoid_smoothing,
            coverage_weight=coverage_weight,
            ctc_loss_weight=0,
            attention_conv_num_channels=attention_conv_num_channels,
            attention_conv_width=attention_conv_width,
            num_stack=num_stack,
            splice=splice,
            conv_channels=conv_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            conv_strides=conv_strides,
            poolings=poolings,
            batch_norm=batch_norm,
            scheduled_sampling_prob=scheduled_sampling_prob,
            scheduled_sampling_ramp_max_step=scheduled_sampling_ramp_max_step,
            label_smoothing_prob=label_smoothing_prob,
            weight_noise_std=weight_noise_std,
            encoder_residual=encoder_residual,
            encoder_dense_residual=encoder_dense_residual,
            decoder_residual=decoder_residual,
            decoder_dense_residual=decoder_dense_residual)

        # Setting for the encoder
        self.encoder_num_layers_sub = encoder_num_layers_sub

        # Setting for the decoder
        self.decoder_num_units_sub = decoder_num_units_sub
        self.decoder_num_layers_sub = decoder_num_layers_sub
        self.embedding_dim_sub = embedding_dim_sub
        self.num_classes_sub = num_classes_sub + 2
        self.sos_index_sub = num_classes_sub + 1
        self.eos_index_sub = num_classes_sub
        # NOTE: add <SOS> and <EOS>

        # Setting for MTL
        self.main_loss_weight = main_loss_weight
        self.sub_loss_weight = 1 - main_loss_weight - ctc_loss_weight_sub
        self.ctc_loss_weight_sub = ctc_loss_weight_sub

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
                    num_stack=num_stack,
                    splice=splice,
                    conv_channels=conv_channels,
                    conv_kernel_sizes=conv_kernel_sizes,
                    conv_strides=conv_strides,
                    poolings=poolings,
                    activation=activation,
                    batch_norm=batch_norm,
                    residual=encoder_residual,
                    dense_residual=encoder_dense_residual)
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
                    num_stack=num_stack,
                    splice=splice,
                    conv_channels=conv_channels,
                    conv_kernel_sizes=conv_kernel_sizes,
                    conv_strides=conv_strides,
                    poolings=poolings,
                    activation=activation,
                    batch_norm=batch_norm,
                    residual=encoder_residual,
                    dense_residual=encoder_dense_residual)
        else:
            raise NotImplementedError

        self.is_bridge_sub = False
        if self.sub_loss_weight > 0:
            ##############################
            # Decoder in the sub task
            ##############################
            self.decoder_sub = RNNDecoder(
                embedding_dim=embedding_dim_sub + decoder_num_units_sub,
                rnn_type=decoder_type,
                num_units=decoder_num_units_sub,
                num_layers=decoder_num_layers,
                dropout=decoder_dropout,
                parameter_init=parameter_init,
                use_cuda=self.use_cuda,
                batch_first=True,
                residual=decoder_residual,
                dense_residual=decoder_dense_residual)

            ###################################
            # Attention layer in the sub task
            ###################################
            self.attend_sub = AttentionMechanism(
                decoder_num_units=decoder_num_units_sub,
                attention_type=attention_type,
                attention_dim=attention_dim,
                sharpening_factor=sharpening_factor,
                sigmoid_smoothing=sigmoid_smoothing,
                out_channels=attention_conv_num_channels,
                kernel_size=attention_conv_width)

            #################################################################
            # Bridge layer between the encoder and decoder in the sub task
            #################################################################
            if encoder_bidirectional or encoder_num_units != decoder_num_units_sub:
                if encoder_bidirectional:
                    self.bridge_sub = LinearND(
                        encoder_num_units * 2, decoder_num_units_sub)
                else:
                    self.bridge_sub = LinearND(
                        encoder_num_units, decoder_num_units_sub)
                self.bridge_init_sub = LinearND(
                    encoder_num_units, decoder_num_units_sub)
                self.is_bridge_sub = True

            self.embedding_sub = nn.Embedding(
                self.num_classes_sub, embedding_dim_sub)

            self.proj_layer_sub = LinearND(
                decoder_num_units_sub * 2, decoder_num_units_sub)
            self.fc_sub = LinearND(
                decoder_num_units_sub, self.num_classes_sub - 1)
            # NOTE: <SOS> is removed because the decoder never predict <SOS>
            # class

        if ctc_loss_weight_sub > 0:
            if self.is_bridge_sub:
                self.fc_ctc_sub = LinearND(
                    decoder_num_units_sub, num_classes_sub + 1)
            else:
                self.fc_ctc_sub = LinearND(
                    encoder_num_units * self.encoder_num_directions, num_classes_sub + 1)

            # Set CTC decoders
            self._decode_ctc_greedy_np = GreedyDecoder(blank_index=0)
            self._decode_ctc_beam_np = BeamSearchDecoder(blank_index=0)

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

        # Gaussian noise injection
        if self.weight_noise_injection:
            self._inject_weight_noise(mean=0., std=self.weight_noise_std)

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

        ##################################################
        # Main task
        ##################################################
        # Teacher-forcing
        logits, attention_weights = self._decode_train(
            encoder_outputs, encoder_final_state, labels_var)

        # Output smoothing
        if self.logits_temperature != 1:
            logits /= self.logits_temperature

        # Compute XE sequence loss in the main task
        batch_size, label_num, num_classes = logits.size()
        logits = logits.view((-1, num_classes))
        labels_1d = labels_var[:, 1:].contiguous().view(-1)
        xe_loss_main = F.cross_entropy(
            logits, labels_1d,
            ignore_index=self.sos_index,
            size_average=False) * (1 - self.label_smoothing_prob)
        # NOTE: labels_var are padded by sos_index

        # Label smoothing (with uniform distribution)
        if self.label_smoothing_prob > 0:
            log_probs = F.log_softmax(logits, dim=-1)
            uniform = Variable(torch.ones(
                batch_size, label_num, num_classes)) / num_classes
            if self.use_cuda:
                uniform = uniform.cuda()
            kl = nn.KLDivLoss(size_average=False, reduce=True)
            xe_loss_main += kl(log_probs, uniform) * self.label_smoothing_prob

        # Add coverage term
        if self.coverage_weight != 0:
            pass
            # coverage = self._compute_coverage(attention_weights)
            # loss += coverage_weight * coverage
            # TODO: sub taskも入れる？

        xe_loss_main = xe_loss_main * self.main_loss_weight / batch_size
        loss = xe_loss_main

        ##################################################
        # Sub task (attention)
        ##################################################
        if self.sub_loss_weight > 0:
            # Teacher-forcing
            logits_sub, attention_weights_sub = self._decode_train(
                encoder_outputs_sub, encoder_final_state_sub, labels_sub_var,
                is_sub_task=True)

            # Output smoothing
            if self.logits_temperature != 1:
                logits_sub /= self.logits_temperature

            # Compute XE sequence loss in the sub task
            batch_size, label_num_sub, num_classes_sub = logits_sub.size()
            logits_sub = logits_sub.view((-1, num_classes_sub))
            labels_sub_1d = labels_sub_var[:, 1:].contiguous().view(-1)
            xe_loss_sub = F.cross_entropy(
                logits_sub, labels_sub_1d,
                ignore_index=self.sos_index_sub,
                size_average=False) * (1 - self.label_smoothing_prob)
            # NOTE: labels_var are padded by sos_index_sub

            # Label smoothing (with uniform distribution)
            if self.label_smoothing_prob > 0:
                log_probs_sub = F.log_softmax(logits_sub, dim=-1)
                uniform_sub = Variable(torch.ones(
                    batch_size, label_num_sub, num_classes_sub)) / num_classes_sub
                if self.use_cuda:
                    uniform_sub = uniform_sub.cuda()
                kl_div_sub = nn.KLDivLoss(size_average=False, reduce=True)
                xe_loss_sub += kl_div_sub(log_probs_sub, uniform_sub) * \
                    self.label_smoothing_prob

            xe_loss_sub = xe_loss_sub * self.sub_loss_weight / batch_size
            loss += xe_loss_sub

        ##################################################
        # Sub task (CTC)
        ##################################################
        if self.ctc_loss_weight_sub > 0:
            ctc_loss_sub = self._compute_ctc_loss(
                encoder_outputs_sub, labels_sub_var,
                inputs_seq_len_var, labels_seq_len_sub_var, is_sub_task=True)
            # NOTE: including modifying inputs_seq_len_sub

            ctc_loss_sub = ctc_loss_sub * self.ctc_loss_weight_sub / batch_size
            loss += ctc_loss_sub

        self._step += 1

        if self.sub_loss_weight > self.ctc_loss_weight_sub:
            return loss, xe_loss_main, xe_loss_sub
        else:
            return loss, xe_loss_main, ctc_loss_sub

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
            best_hyps (np.ndarray): A tensor of size `[]`
        """
        # Wrap by Variable
        inputs_var = np2var(inputs, use_cuda=self.use_cuda, volatile=True)
        inputs_seq_len_var = np2var(
            inputs_seq_len, dtype='int', use_cuda=self.use_cuda, volatile=True)

        # Encode acoustic features
        if is_sub_task:
            _, _, encoder_outputs, encoder_final_state, perm_indices = self._encode(
                inputs_var, inputs_seq_len_var, volatile=True,
                is_multi_task=True)
        else:
            encoder_outputs, encoder_final_state, _, _, perm_indices = self._encode(
                inputs_var, inputs_seq_len_var, volatile=True,
                is_multi_task=True)

        # Permutate indices
        if perm_indices is not None:
            perm_indices = var2np(perm_indices)
            inputs_seq_len_var = inputs_seq_len_var[perm_indices]

        if beam_width == 1:
            if is_sub_task:
                if self.sub_loss_weight > self.ctc_loss_weight_sub:
                    ########################################
                    # Decode by attention decoder
                    ########################################
                    best_hyps, _ = self._decode_infer_greedy(
                        encoder_outputs, encoder_final_state, max_decode_length,
                        is_sub_task=True)
                else:
                    ########################################
                    # Decode by CTC decoder
                    ########################################
                    # Path through the softmax layer
                    batch_size, max_time = encoder_outputs.size()[:2]
                    encoder_outputs = encoder_outputs.contiguous()
                    encoder_outputs = encoder_outputs.view(
                        batch_size * max_time, -1)
                    logits_ctc = self.fc_ctc_sub(encoder_outputs)
                    logits_ctc = logits_ctc.view(batch_size, max_time, -1)
                    log_probs = F.log_softmax(
                        logits_ctc, dim=logits_ctc.dim() - 1)

                    # Modify inputs_seq_len for reducing time resolution
                    if self.encoder.conv is not None:
                        for i in range(len(inputs_seq_len)):
                            inputs_seq_len_var.data[i] = self.encoder.conv_out_size(
                                inputs_seq_len_var.data[i], 1)
                    inputs_seq_len_var /= 2 ** sum(
                        self.subsample_list[:self.encoder_num_layers_sub])
                    # NOTE: floor is not needed because inputs_seq_len_var is
                    # IntTensor

                    if beam_width == 1:
                        best_hyps = self._decode_ctc_greedy_np(
                            var2np(log_probs), var2np(inputs_seq_len_var))
                    else:
                        best_hyps = self._decode_ctc_beam_np(
                            var2np(log_probs), var2np(inputs_seq_len_var),
                            beam_width=beam_width)

                    best_hyps = best_hyps - 1
                    # NOTE: index 0 is reserved for blank in warpctc_pytorch
            else:
                best_hyps, _ = self._decode_infer_greedy(
                    encoder_outputs, encoder_final_state, max_decode_length)
        else:
            if is_sub_task:
                raise NotImplementedError
            else:
                best_hyps, attention_weights = self._decode_infer_beam(
                    encoder_outputs, encoder_final_state,
                    inputs_seq_len_var,
                    beam_width, max_decode_length)

        # Remove <SOS>
        if not (is_sub_task and self.sub_loss_weight <= self.ctc_loss_weight_sub):
            best_hyps = best_hyps[:, 1:]

        # Permutate indices to the original order
        if perm_indices is not None:
            best_hyps = best_hyps[perm_indices]

        return best_hyps
