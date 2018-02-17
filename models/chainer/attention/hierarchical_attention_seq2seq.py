#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Hierarchical attention-based sequence-to-sequence model (chainer)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import chainer
from chainer import functions as F

from models.chainer.linear import LinearND, Embedding, Embedding_LS
from models.chainer.criterion import cross_entropy_label_smoothing
from models.chainer.attention.attention_seq2seq import AttentionSeq2seq
from models.chainer.encoders.load_encoder import load
from models.chainer.attention.rnn_decoder import RNNDecoder
from models.chainer.attention.attention_layer import AttentionMechanism
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
            decoder_dense_residual=decoder_dense_residual)
        self.model_type = 'hierarchical_attention'

        # Setting for the encoder
        self.encoder_num_layers_sub = encoder_num_layers_sub

        # Setting for the decoder
        self.decoder_num_units_sub = decoder_num_units_sub
        self.decoder_num_layers_sub = decoder_num_layers_sub
        self.embedding_dim_sub = embedding_dim_sub
        self.num_classes_sub = num_classes_sub + 2  # Add <SOS> and <EOS> class
        self.sos_index_sub = 0
        self.eos_index_sub = num_classes_sub + 1

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

        #########################
        # Encoder
        # NOTE: overide encoder
        #########################
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
                residual=encoder_residual,
                dense_residual=encoder_dense_residual)
        else:
            raise NotImplementedError

        if self.init_dec_state != 'zero':
            self.W_dec_init_sub = LinearND(
                decoder_num_units_sub, decoder_num_units_sub)

        self.is_bridge_sub = False
        if self.sub_loss_weight > 0:
            ##############################
            # Decoder (sub)
            ##############################
            self.decoder_sub = RNNDecoder(
                input_size=decoder_num_units_sub + embedding_dim_sub,
                rnn_type=decoder_type,
                num_units=decoder_num_units_sub,
                num_layers=decoder_num_layers_sub,
                dropout=dropout_decoder,
                use_cuda=self.use_cuda,
                residual=decoder_residual,
                dense_residual=decoder_dense_residual)

            ###################################
            # Attention layer (sub)
            ###################################
            self.attend_sub = AttentionMechanism(
                decoder_num_units=decoder_num_units_sub,
                attention_type=attention_type,
                attention_dim=attention_dim,
                use_cuda=self.use_cuda,
                sharpening_factor=sharpening_factor,
                sigmoid_smoothing=sigmoid_smoothing,
                out_channels=attention_conv_num_channels,
                kernel_size=attention_conv_width)

            #################################################################
            # Bridge layer between the encoder and decoder (sub)
            #################################################################
            if encoder_bidirectional or encoder_num_units != decoder_num_units_sub:
                if encoder_bidirectional:
                    self.bridge_sub = LinearND(
                        encoder_num_units * 2, decoder_num_units_sub,
                        dropout=dropout_encoder, use_cuda=self.use_cuda)
                else:
                    self.bridge_sub = LinearND(
                        encoder_num_units, decoder_num_units_sub,
                        dropout=dropout_encoder, use_cuda=self.use_cuda)
                self.is_bridge_sub = True

            if label_smoothing_prob > 0:
                self.embed_sub = Embedding_LS(
                    num_classes=self.num_classes_sub,
                    embedding_dim=embedding_dim_sub,
                    dropout=dropout_embedding,
                    label_smoothing_prob=label_smoothing_prob,
                    use_cuda=self.use_cuda)
            else:
                self.embed_sub = Embedding(
                    num_classes=self.num_classes_sub,
                    embedding_dim=embedding_dim_sub,
                    dropout=dropout_embedding,
                    use_cuda=self.use_cuda)

            self.proj_layer_sub = LinearND(
                decoder_num_units_sub * 2, decoder_num_units_sub,
                dropout=dropout_decoder, use_cuda=self.use_cuda)
            self.fc_sub = LinearND(
                decoder_num_units_sub, self.num_classes_sub,
                use_cuda=self.use_cuda)

        if ctc_loss_weight_sub > 0:
            if self.is_bridge_sub:
                self.fc_ctc_sub = LinearND(
                    decoder_num_units_sub, num_classes_sub + 1,
                    use_cuda=self.use_cuda)
            else:
                self.fc_ctc_sub = LinearND(
                    encoder_num_units * self.encoder_num_directions, num_classes_sub + 1,
                    use_cuda=self.use_cuda)

            self.blank_index = 0

            # Set CTC decoders
            self._decode_ctc_greedy_np = GreedyDecoder(
                blank_index=self.blank_index)
            self._decode_ctc_beam_np = BeamSearchDecoder(
                blank_index=self.blank_index)

        # Initialize parameters
        self.init_weights(parameter_init,
                          distribution=parameter_init_distribution,
                          ignore_keys=['bias'])

        # Initialize all biases with 0
        self.init_weights(0, distribution='constant', keys=['bias'])

        # Recurrent weights are orthogonalized
        if recurrent_weight_orthogonal:
            self.init_weights(parameter_init, distribution='orthogonal',
                              keys=[encoder_type, 'weight'], ignore_keys=['bias'])
            self.init_weights(parameter_init, distribution='orthogonal',
                              keys=[decoder_type, 'weight'], ignore_keys=['bias'])

        # Initialize bias in forget gate with 1
        if init_forget_gate_bias_with_one:
            self.init_forget_gate_bias_with_one()

    def __call__(self, xs, ys, ys_sub, x_lens, y_lens, y_lens_sub, is_eval=False):
        """Forward computation.
        Args:
            xs (list of np.ndarray): A tensor of size `[B, T_in, input_size]`
            ys (np.ndarray): A tensor of size `[B, T_out]`
            ys_sub (np.ndarray): A tensor of size `[B, T_out_sub]`
            x_lens (np.ndarray): A tensor of size `[B]`
            y_lens (np.ndarray): A tensor of size `[B]`
            y_lens_sub (np.ndarray): A tensor of size `[B]`
            is_eval (bool, optional): if True, the history will not be saved.
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

        return loss, loss_main, loss_sub

    def _forward(self, xs, ys, ys_sub, x_lens, y_lens, y_lens_sub):
        # Wrap by Variable
        xs = np2var(xs, use_cuda=self.use_cuda, backend='chainer')
        ys = np2var(ys, use_cuda=self.use_cuda, backend='chainer')
        ys_sub = np2var(ys_sub, use_cuda=self.use_cuda, backend='chainer')
        y_lens = np2var(y_lens, use_cuda=self.use_cuda, backend='chainer')
        y_lens_sub = np2var(
            y_lens_sub, use_cuda=self.use_cuda, backend='chainer')

        # NOTE: index 0 is reserved for blank and <SOS>
        ys = ys + 1
        ys_sub = ys_sub + 1

        # Encode acoustic features
        xs, x_lens, xs_sub, x_lens_sub = self._encode(
            xs, x_lens, is_multi_task=True)

        ##################################################
        # Main task
        ##################################################
        # Teacher-forcing
        logits_main, att_weights = self._decode_train(xs, x_lens, ys)

        # Output smoothing
        if self.logits_temperature != 1:
            logits_main /= self.logits_temperature

        # Compute XE sequence loss in the main task
        loss_main = F.softmax_cross_entropy(
            x=logits_main.reshape((-1, logits_main.shape[2])),
            t=ys[:, 1:].reshape(-1),  # NOTE: Exclude <SOS>
            normalize=True, cache_score=True, class_weight=None,
            ignore_label=self.sos_index, reduce='no')
        # NOTE: ys are padded by <SOS>
        # NOTE: len(loss_main) = batch_size * max_time
        loss_main = F.sum(loss_main, axis=0) / len(xs)

        # Label smoothing (with uniform distribution)
        if self.label_smoothing_prob > 0:
            # XE
            loss_ls_main = cross_entropy_label_smoothing(
                logits_main,
                label_smoothing_prob=self.label_smoothing_prob,
                distribution='uniform',
                size_average=False) / len(xs)
            loss_main = loss_main * (1 - self.label_smoothing_prob) + \
                loss_ls_main
            # print(loss_ls_main)

        # Add coverage term
        if self.coverage_weight != 0:
            raise NotImplementedError
            # TODO: sub taskも入れる？

        loss_main = loss_main * self.main_loss_weight_tmp
        loss = loss_main
        # TODO: copy?

        ##################################################
        # Sub task (attention, optional)
        ##################################################
        if self.sub_loss_weight > 0:
            # Teacher-forcing
            logits_sub, att_weights_sub = self._decode_train(
                xs_sub, x_lens_sub, ys_sub, is_sub_task=True)

            # Output smoothing
            if self.logits_temperature != 1:
                logits_sub /= self.logits_temperature

            # Compute XE sequence loss (sub)
            loss_sub = F.softmax_cross_entropy(
                x=logits_sub.reshape((-1, logits_sub.shape[2])),
                t=ys_sub[:, 1:].reshape(-1),  # NOTE: Exclude <SOS>
                normalize=True, cache_score=True, class_weight=None,
                ignore_label=self.sos_index_sub, reduce='no')
            # NOTE: ys_sub are padded by <SOS>
            # NOTE: len(loss_sub) = batch_size * max_time_sub
            loss_sub = F.sum(loss_sub, axis=0) / len(xs)

            # Label smoothing (with uniform distribution)
            if self.label_smoothing_prob > 0:
                # XE
                loss_ls_sub = cross_entropy_label_smoothing(
                    logits_sub,
                    label_smoothing_prob=self.label_smoothing_prob,
                    distribution='uniform',
                    size_average=False) / len(xs)
                loss_sub = loss_sub * (1 - self.label_smoothing_prob) + \
                    loss_ls_sub
                # print(loss_ls_sub)

            loss_sub = loss_sub * self.sub_loss_weight_tmp
            loss += loss_sub

        ##################################################
        # Sub task (CTC, optional)
        ##################################################
        if self.ctc_loss_weight_sub > 0:
            x_lens_sub = np2var(
                x_lens_sub, use_cuda=self.use_cuda, backend='chainer')

            logits_ctc_sub = self.fc_ctc_sub(xs_sub)
            ctc_loss_sub = self.compute_ctc_loss(
                logits_ctc_sub, ys_sub, x_lens_sub, y_lens_sub,
                size_average=False) / len(xs)
            ctc_loss_sub = ctc_loss_sub * self.ctc_loss_weight_sub_tmp
            loss += ctc_loss_sub

        if self.sub_loss_weight > self.ctc_loss_weight_sub:
            return loss, loss_main, loss_sub
        else:
            return loss, loss_main, ctc_loss_sub

    def decode(self, xs, x_lens, beam_width, max_decode_len, is_sub_task=False):
        """Decoding in the inference stage.
        Args:
            xs (list of np.ndarray): A tensor of size `[B, T_in, input_size]`
            x_lens (np.ndarray): A tensor of size `[B]`
            beam_width (int): the size of beam
            max_decode_len (int): the length of output sequences
                to stop prediction when EOS token have not been emitted
            is_sub_task (bool, optional):
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B]`
            perm_idx (np.ndarray): For interface with pytorch, not used
        """
        with chainer.no_backprop_mode(), chainer.using_config('train', False):

            # Wrap by Variable
            xs = np2var(xs, use_cuda=self.use_cuda, backend='chainer')

            # Encode acoustic features
            if is_sub_task:
                _, _, enc_out, x_lens = self._encode(
                    xs, x_lens, is_multi_task=True)
            else:
                enc_out, x_lens, _, _ = self._encode(
                    xs, x_lens, is_multi_task=True)

            if is_sub_task:
                # Decode by attention decoder
                if self.sub_loss_weight > self.ctc_loss_weight_sub:
                    if beam_width == 1:
                        best_hyps, _ = self._decode_infer_greedy(
                            enc_out, x_lens, max_decode_len, is_sub_task=True)
                    else:
                        best_hyps = self._decode_infer_beam(
                            enc_out, x_lens, beam_width, max_decode_len, is_sub_task=True)

                # Decode by CTC decoder
                else:
                    # Path through the softmax layer
                    batch_size, max_time = enc_out.shape[:2]
                    enc_out = enc_out.reshape(batch_size * max_time, -1)
                    logits_ctc = self.fc_ctc_sub(enc_out)
                    logits_ctc = logits_ctc.reshape(
                        batch_size, max_time, -1)
                    log_probs = F.log_softmax(logits_ctc)

                    if beam_width == 1:
                        best_hyps = self._decode_ctc_greedy_np(
                            var2np(log_probs, backend='chainer'), x_lens)
                    else:
                        best_hyps = self._decode_ctc_beam_np(
                            var2np(log_probs, backend='chainer'),
                            x_lens, beam_width=beam_width)
            else:
                if beam_width == 1:
                    best_hyps, _ = self._decode_infer_greedy(
                        enc_out, x_lens, max_decode_len)
                else:
                    best_hyps = self._decode_infer_beam(
                        enc_out, x_lens, beam_width, max_decode_len)

        # NOTE: index 0 is reserved for the blank and <SOS>
        best_hyps -= 1

        perm_idx = np.arange(0, len(xs), 1)
        return best_hyps, perm_idx
