#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Hierarchical attention-based sequence-to-sequence model (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch

from models.pytorch.linear import LinearND, Embedding, Embedding_LS
from models.pytorch.attention.attention_seq2seq import AttentionSeq2seq
from models.pytorch.encoders.load_encoder import load
# from models.pytorch.attention.rnn_decoder import RNNDecoder
from models.pytorch.attention.rnn_decoder_nstep import RNNDecoder
from models.pytorch.attention.attention_layer import AttentionMechanism
from models.pytorch.ctc.decoders.greedy_decoder import GreedyDecoder
from models.pytorch.ctc.decoders.beam_search_decoder import BeamSearchDecoder


class HierarchicalAttentionSeq2seq(AttentionSeq2seq):

    def __init__(self,
                 input_size,
                 encoder_type,
                 encoder_bidirectional,
                 encoder_num_units,
                 encoder_num_proj,
                 encoder_num_layers,
                 encoder_num_layers_sub,  # ***
                 attention_type,
                 attention_dim,
                 decoder_type,
                 decoder_num_units,
                 decoder_num_layers,
                 decoder_num_units_sub,  # ***
                 decoder_num_layers_sub,  # ***
                 embedding_dim,
                 embedding_dim_sub,  # ***
                 dropout_input,
                 dropout_encoder,
                 dropout_decoder,
                 dropout_embedding,
                 main_loss_weight,  # ***
                 num_classes,
                 num_classes_sub,  # ***
                 parameter_init_distribution='uniform',
                 parameter_init=0.1,
                 recurrent_weight_orthogonal=False,
                 init_forget_gate_bias_with_one=True,
                 subsample_list=[],
                 subsample_type='drop',
                 init_dec_state='zero',
                 sharpening_factor=1,
                 logits_temperature=1,
                 sigmoid_smoothing=False,
                 coverage_weight=0,
                 ctc_loss_weight_sub=0,  # ***
                 attention_conv_num_channels=10,
                 attention_conv_width=201,
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
                 decoder_dense_residual=False,
                 decoding_order='spell_attend',
                 curriculum_training=False):

        super(HierarchicalAttentionSeq2seq, self).__init__(
            input_size=input_size,
            encoder_type=encoder_type,
            encoder_bidirectional=encoder_bidirectional,
            encoder_num_units=encoder_num_units,
            encoder_num_proj=encoder_num_proj,
            encoder_num_layers=encoder_num_layers,
            attention_type=attention_type,
            attention_dim=attention_dim,
            decoder_type=decoder_type,
            decoder_num_units=decoder_num_units,
            decoder_num_layers=decoder_num_layers,
            embedding_dim=embedding_dim,
            dropout_input=dropout_input,
            dropout_encoder=dropout_encoder,
            dropout_decoder=dropout_decoder,
            dropout_embedding=dropout_embedding,
            num_classes=num_classes,
            parameter_init=parameter_init,
            subsample_list=subsample_list,
            subsample_type=subsample_type,
            init_dec_state=init_dec_state,
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
            decoder_dense_residual=decoder_dense_residual,
            decoding_order=decoding_order)
        self.model_type = 'hierarchical_attention'

        # Setting for the encoder
        self.encoder_num_layers_sub = encoder_num_layers_sub

        # Setting for the decoder
        self.decoder_num_units_sub = decoder_num_units_sub
        self.decoder_num_layers_sub = decoder_num_layers_sub
        self.embedding_dim_sub = embedding_dim_sub
        self.num_classes_sub = num_classes_sub + 1  # Add <EOS> class
        self.sos_sub = num_classes_sub
        self.eos_sub = num_classes_sub
        # NOTE: <SOS> and <EOS> have the same index

        # Setting for MTL
        self.main_loss_weight = main_loss_weight
        self.main_loss_weight_tmp = main_loss_weight
        self.sub_loss_weight = 1 - main_loss_weight - ctc_loss_weight_sub
        self.sub_loss_weight_tmp = 1 - main_loss_weight - ctc_loss_weight_sub
        self.ctc_loss_weight_sub = ctc_loss_weight_sub
        self.ctc_loss_weight_sub_tmp = ctc_loss_weight_sub
        if curriculum_training and scheduled_sampling_ramp_max_step == 0:
            raise ValueError('Set scheduled_sampling_ramp_max_step.')
        self.curriculum_training = curriculum_training

        ##############################
        # Encoder
        # NOTE: overide encoder
        ##############################
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
                # pack_sequence=False if init_dec_state == 'zero' else True,
                pack_sequence=True,
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

        if self.init_dec_state != 'zero':
            self.W_dec_init_sub = LinearND(
                self.encoder_num_units, decoder_num_units_sub)

        if self.sub_loss_weight > 0:
            ##############################
            # Decoder (sub)
            ##############################
            self.decoder_sub = RNNDecoder(
                input_size=self.encoder_num_units + embedding_dim_sub,
                rnn_type=decoder_type,
                num_units=decoder_num_units_sub,
                num_layers=decoder_num_layers_sub,
                dropout=dropout_decoder,
                residual=decoder_residual,
                dense_residual=decoder_dense_residual)

            ###################################
            # Attention layer (sub)
            ###################################
            self.attend_sub = AttentionMechanism(
                encoder_num_units=self.encoder_num_units,
                decoder_num_units=decoder_num_units_sub,
                attention_type=attention_type,
                attention_dim=attention_dim,
                sharpening_factor=sharpening_factor,
                sigmoid_smoothing=sigmoid_smoothing,
                out_channels=attention_conv_num_channels,
                kernel_size=attention_conv_width)

            ##############################
            # Embedding (sub)
            ##############################
            if label_smoothing_prob > 0:
                self.embed_sub = Embedding_LS(num_classes=self.num_classes_sub,
                                              embedding_dim=embedding_dim_sub,
                                              dropout=dropout_embedding,
                                              label_smoothing_prob=label_smoothing_prob)
            else:
                self.embed_sub = Embedding(num_classes=self.num_classes_sub,
                                           embedding_dim=embedding_dim_sub,
                                           dropout=dropout_embedding,
                                           ignore_index=-1)

            ##############################
            # Output layer (sub)
            ##############################
            self.W_d_sub = LinearND(decoder_num_units_sub, decoder_num_units_sub,
                                    dropout=dropout_decoder)
            self.W_c_sub = LinearND(self.encoder_num_units, decoder_num_units_sub,
                                    dropout=dropout_decoder)
            self.fc_sub = LinearND(decoder_num_units_sub, self.num_classes_sub)

        ##############################
        # CTC (sub)
        ##############################
        if ctc_loss_weight_sub > 0:
            self.fc_ctc_sub = LinearND(
                self.encoder_num_units, num_classes_sub + 1)

            # Set CTC decoders
            self._decode_ctc_greedy_np = GreedyDecoder(blank_index=0)
            self._decode_ctc_beam_np = BeamSearchDecoder(blank_index=0)
            # NOTE: index 0 is reserved for the blank class

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
                              keys=[encoder_type, 'weight'],
                              ignore_keys=['bias'])
            self.init_weights(parameter_init,
                              distribution='orthogonal',
                              keys=[decoder_type, 'weight'],
                              ignore_keys=['bias'])

        # Initialize bias in forget gate with 1
        if init_forget_gate_bias_with_one:
            self.init_forget_gate_bias_with_one()

    def forward(self, xs, ys, ys_sub, x_lens, y_lens, y_lens_sub, is_eval=False):
        """Forward computation.
        Args:
            xs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            ys (np.ndarray): A tensor of size `[B, T_out]`
            ys_sub (np.ndarray): A tensor of size `[B, T_out_sub]`
            x_lens (np.ndarray): A tensor of size `[B]`
            y_lens (np.ndarray): A tensor of size `[B]`
            y_lens_sub (np.ndarray): A tensor of size `[B]`
            is_eval (bool, optional): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (torch.autograd.Variable(float) or float): A tensor of size `[1]`
            loss_main (torch.autograd.Variable(float) or float): A tensor of size `[1]`
            loss_sub (torch.autograd.Variable(float) or float): A tensor of size `[1]`
        """
        if is_eval:
            self.eval()
        else:
            self.train()

            # Gaussian noise injection
            if self.weight_noise_injection:
                self.inject_weight_noise(mean=0, std=self.weight_noise_std)

        # NOTE: ys and ys_sub are padded with -1 here
        # ys_in and ys_sub_in areb padded with <EOS> in order to convert to
        # one-hot vector, and added <SOS> before the first token
        # ys_out and ys_sub_out are padded with -1, and added <EOS>
        # after the last token
        ys_in = self._create_var((ys.shape[0], ys.shape[1] + 1),
                                 fill_value=self.eos, dtype='long')
        ys_sub_in = self._create_var((ys_sub.shape[0], ys_sub.shape[1] + 1),
                                     fill_value=self.eos_sub, dtype='long')
        ys_out = self._create_var((ys.shape[0], ys.shape[1] + 1),
                                  fill_value=-1, dtype='long')
        ys_sub_out = self._create_var((ys_sub.shape[0], ys_sub.shape[1] + 1),
                                      fill_value=-1, dtype='long')
        for b in range(len(xs)):
            ys_in.data[b, 0] = self.sos
            ys_in.data[b, 1:y_lens[b] + 1] = torch.from_numpy(
                ys[b, :y_lens[b]])
            ys_sub_in.data[b, 0] = self.sos_sub
            ys_sub_in.data[b, 1:y_lens_sub[b] + 1] = torch.from_numpy(
                ys_sub[b, :y_lens_sub[b]])

            ys_out.data[b, :y_lens[b]] = torch.from_numpy(ys[b, :y_lens[b]])
            ys_out.data[b, y_lens[b]] = self.eos
            ys_sub_out.data[b, :y_lens_sub[b]] = torch.from_numpy(
                ys_sub[b, :y_lens_sub[b]])
            ys_sub_out.data[b, y_lens_sub[b]] = self.eos_sub

        if self.use_cuda:
            ys_in = ys_in.cuda()
            ys_sub_in = ys_sub_in.cuda()
            ys_out = ys_out.cuda()
            ys_sub_out = ys_sub_out.cuda()

        # Wrap by Variable
        xs = self.np2var(xs)
        x_lens = self.np2var(x_lens, dtype='int')
        y_lens = self.np2var(y_lens, dtype='int')
        y_lens_sub = self.np2var(y_lens_sub, dtype='int')

        # Encode acoustic features
        xs, x_lens, xs_sub, x_lens_sub, perm_idx = self._encode(
            xs, x_lens, is_multi_task=True)

        # Permutate indices
        if perm_idx is not None:
            ys_in = ys_in[perm_idx]
            ys_out = ys_out[perm_idx]
            y_lens = y_lens[perm_idx]

            ys_sub_in = ys_sub_in[perm_idx]
            ys_sub_out = ys_sub_out[perm_idx]
            y_lens_sub = y_lens_sub[perm_idx]

        ##################################################
        # Main task
        ##################################################
        # Compute XE loss
        loss_main = self.compute_xe_loss(
            xs, ys_in, ys_out, x_lens, y_lens, size_average=True)

        loss_main = loss_main * self.main_loss_weight_tmp
        loss = loss_main.clone()

        ##################################################
        # Sub task (attention, optional)
        ##################################################
        if self.sub_loss_weight > 0:
            # Compute XE loss
            loss_sub = self.compute_xe_loss(
                xs_sub, ys_sub_in, ys_sub_out, x_lens_sub, y_lens_sub,
                is_sub_task=True, size_average=True)

            loss_sub = loss_sub * self.sub_loss_weight_tmp
            loss += loss_sub

        ##################################################
        # Sub task (CTC, optional)
        ##################################################
        if self.ctc_loss_weight_sub > 0:
            ctc_loss_sub = self.compute_ctc_loss(
                xs_sub, ys_sub_in[:, 1:] + 1,
                x_lens_sub, y_lens_sub, is_sub_task=True, size_average=True)

            ctc_loss_sub = ctc_loss_sub * self.ctc_loss_weight_sub_tmp
            loss += ctc_loss_sub

        if is_eval:
            loss = loss.data[0]
            loss_main = loss_main.data[0]
            if self.sub_loss_weight > 0:
                loss_sub = loss_sub.data[0]
            if self.ctc_loss_weight_sub > 0:
                ctc_loss_sub = ctc_loss_sub.data[0]
        else:
            # Update the probability of scheduled sampling
            self._step += 1
            if self.sample_prob > 0:
                self._sample_prob = min(
                    self.sample_prob,
                    self.sample_prob / self.sample_ramp_max_step * self._step)

            # Curriculum training (gradually from char to word task)
            if self.curriculum_training:
                # main
                self.main_loss_weight_tmp = min(
                    self.main_loss_weight,
                    0.0 + self.main_loss_weight / self.sample_ramp_max_step * self._step * 2)
                # sub (attention)
                self.sub_loss_weight_tmp = max(
                    self.sub_loss_weight,
                    1.0 - (1 - self.sub_loss_weight) / self.sample_ramp_max_step * self._step * 2)
                # sub (CTC)
                self.ctc_loss_weight_sub_tmp = max(
                    self.ctc_loss_weight_sub,
                    1.0 - (1 - self.ctc_loss_weight_sub) / self.sample_ramp_max_step * self._step * 2)

        if self.sub_loss_weight > self.ctc_loss_weight_sub:
            return loss, loss_main, loss_sub
        else:
            return loss, loss_main, ctc_loss_sub

    def decode(self, xs, x_lens, beam_width, max_decode_len, is_sub_task=False):
        """Decoding in the inference stage.
        Args:
            xs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            x_lens (np.ndarray): A tensor of size `[B]`
            beam_width (int): the size of beam
            max_decode_len (int): the length of output sequences
                to stop prediction when EOS token have not been emitted
            is_sub_task (bool, optional):
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B]`
            perm_idx (np.ndarray): A tensor of size `[B]`
        """
        # Change to evaluation mode
        self.eval()

        if is_sub_task and self.ctc_loss_weight_sub > self.sub_loss_weight:
            # Decode by CTC decoder
            best_hyps, perm_idx = self.decode_ctc(
                xs, x_lens, beam_width, is_sub_task=True)
        else:
            # Wrap by Variable
            xs = self.np2var(xs)
            x_lens = self.np2var(x_lens, dtype='int')

            # Encode acoustic features
            if is_sub_task:
                _, _, enc_out, x_lens, perm_idx = self._encode(
                    xs, x_lens, is_multi_task=True)
            else:
                enc_out, x_lens, _, _, perm_idx = self._encode(
                    xs, x_lens, is_multi_task=True)

            # Decode by attention decoder
            if beam_width == 1:
                best_hyps, _ = self._decode_infer_greedy(
                    enc_out, x_lens, max_decode_len,
                    is_sub_task=is_sub_task)
            else:
                best_hyps = self._decode_infer_beam(
                    enc_out, x_lens, beam_width, max_decode_len,
                    is_sub_task=is_sub_task)

            # Permutate indices to the original order
            if perm_idx is None:
                perm_idx = np.arange(0, len(xs), 1)
            else:
                perm_idx = self.var2np(perm_idx)

        return best_hyps, perm_idx
