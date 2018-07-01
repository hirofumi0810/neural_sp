#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Hierarchical attention-based sequence-to-sequence model (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
import math
from scipy.stats import entropy

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from src.models.pytorch_v3.attention.attention_seq2seq import AttentionSeq2seq
from src.models.pytorch_v3.linear import LinearND, Embedding
from src.models.pytorch_v3.encoders.load_encoder import load
from src.models.pytorch_v3.attention.rnn_decoder import RNNDecoder
from src.models.pytorch_v3.attention.attention_layer import AttentionMechanism, MultiheadAttentionMechanism
from src.models.pytorch_v3.ctc.decoders.greedy_decoder import GreedyDecoder
from src.models.pytorch_v3.ctc.decoders.beam_search_decoder import BeamSearchDecoder
from src.models.pytorch_v3.utils import np2var, var2np
from src.models.pytorch_v3.lm.rnnlm import RNNLM


class HierarchicalAttentionSeq2seq(AttentionSeq2seq):

    def __init__(self,
                 input_type,
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
                 decoder_num_units_sub,  # ***
                 decoder_num_layers,
                 decoder_num_layers_sub,  # ***
                 embedding_dim,
                 embedding_dim_sub,  # ***
                 dropout_input,
                 dropout_encoder,
                 dropout_decoder,
                 dropout_embedding,
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
                 bridge_layer=False,
                 init_dec_state='zero',
                 sharpening_factor=1,
                 logits_temperature=1,
                 sigmoid_smoothing=False,
                 coverage_weight=0,
                 ctc_loss_weight_sub=0,  # ***
                 attention_conv_num_channels=10,
                 attention_conv_width=201,
                 num_stack=1,
                 num_skip=1,
                 splice=1,
                 input_channel=1,
                 conv_channels=[],
                 conv_kernel_sizes=[],
                 conv_strides=[],
                 poolings=[],
                 activation='relu',
                 batch_norm=False,
                 scheduled_sampling_prob=0,
                 scheduled_sampling_max_step=0,
                 label_smoothing_prob=0,
                 label_smoothing_type='unigram',
                 weight_noise_std=0,
                 encoder_residual=False,
                 encoder_dense_residual=False,
                 decoder_residual=False,
                 decoder_dense_residual=False,
                 decoding_order='bahdanau',
                 bottleneck_dim=256,
                 bottleneck_dim_sub=256,  # ***
                 backward_sub=False,  # ***
                 num_heads=1,
                 num_heads_sub=1,  # ***
                 rnnlm_fusion_type=None,
                 rnnlm_config=None,
                 rnnlm_config_sub=None,  # ***
                 rnnlm_weight=0,
                 rnnlm_weight_sub=0,  # ***
                 num_classes_input=0):

        super(HierarchicalAttentionSeq2seq, self).__init__(
            input_type=input_type,
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
            bridge_layer=bridge_layer,
            init_dec_state=init_dec_state,
            sharpening_factor=sharpening_factor,
            logits_temperature=logits_temperature,
            sigmoid_smoothing=sigmoid_smoothing,
            coverage_weight=coverage_weight,
            ctc_loss_weight=0,
            attention_conv_num_channels=attention_conv_num_channels,
            attention_conv_width=attention_conv_width,
            num_stack=num_stack,
            num_skip=num_skip,
            splice=splice,
            input_channel=input_channel,
            conv_channels=conv_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            conv_strides=conv_strides,
            poolings=poolings,
            scheduled_sampling_prob=scheduled_sampling_prob,
            scheduled_sampling_max_step=scheduled_sampling_max_step,
            label_smoothing_prob=label_smoothing_prob,
            label_smoothing_type=label_smoothing_type,
            weight_noise_std=weight_noise_std,
            encoder_residual=encoder_residual,
            encoder_dense_residual=encoder_dense_residual,
            decoder_residual=decoder_residual,
            decoder_dense_residual=decoder_dense_residual,
            decoding_order=decoding_order,
            bottleneck_dim=bottleneck_dim,
            backward_loss_weight=0,
            num_heads=num_heads,
            rnnlm_fusion_type=rnnlm_fusion_type,
            rnnlm_config=rnnlm_config,
            rnnlm_weight=rnnlm_weight,
            num_classes_input=num_classes_input)
        self.model_type = 'hierarchical_attention'

        # Setting for the encoder
        self.encoder_num_units_sub = encoder_num_units
        if encoder_bidirectional:
            self.encoder_num_units_sub *= 2
        self.encoder_num_layers_sub = encoder_num_layers_sub

        # Setting for the decoder in the sub task
        self.decoder_num_units_1 = decoder_num_units_sub
        self.decoder_num_layers_1 = decoder_num_layers_sub
        self.num_classes_sub = num_classes_sub + 1  # Add <EOS> class
        self.sos_1 = num_classes_sub
        self.eos_1 = num_classes_sub
        # NOTE: <SOS> and <EOS> have the same index
        self.backward_1 = backward_sub
        dir_sub = 'bwd' if backward_sub else 'fwd'

        # Setting for the decoder initialization in the sub task
        if backward_sub:
            if init_dec_state == 'first':
                self.init_dec_state_1_bwd = 'final'
            elif init_dec_state == 'final':
                self.init_dec_state_1_bwd = 'first'
            else:
                self.init_dec_state_1_bwd = init_dec_state
            if encoder_type != decoder_type:
                self.init_dec_state_1_bwd = 'zero'
        else:
            self.init_dec_state_1_fwd = init_dec_state
            if encoder_type != decoder_type:
                self.init_dec_state_1_fwd = 'zero'

        # Setting for the attention in the sub task
        self.num_heads_1 = num_heads_sub

        # Setting for MTL
        self.main_loss_weight = main_loss_weight
        self.sub_loss_weight = sub_loss_weight
        self.ctc_loss_weight_sub = ctc_loss_weight_sub
        if backward_sub:
            self.bwd_weight_1 = sub_loss_weight

        # Setting for the RNNLM fusion
        self.rnnlm_1_fwd = None
        self.rnnlm_1_bwd = None
        self.rnnlm_weight_1 = rnnlm_weight_sub

        # RNNLM fusion
        if rnnlm_fusion_type:
            assert rnnlm_config_sub is not None
            self.rnnlm_1_fwd = RNNLM(
                embedding_dim=rnnlm_config_sub['embedding_dim'],
                rnn_type=rnnlm_config_sub['rnn_type'],
                bidirectional=rnnlm_config_sub['bidirectional'],
                num_units=rnnlm_config_sub['num_units'],
                num_layers=rnnlm_config_sub['num_layers'],
                dropout_embedding=rnnlm_config_sub['dropout_embedding'],
                dropout_hidden=rnnlm_config_sub['dropout_hidden'],
                dropout_output=rnnlm_config_sub['dropout_output'],
                num_classes=rnnlm_config_sub['num_classes'],
                parameter_init_distribution=rnnlm_config_sub['parameter_init_distribution'],
                parameter_init=rnnlm_config_sub['parameter_init'],
                recurrent_weight_orthogonal=rnnlm_config_sub['recurrent_weight_orthogonal'],
                init_forget_gate_bias_with_one=rnnlm_config_sub['init_forget_gate_bias_with_one'],
                tie_weights=rnnlm_config_sub['tie_weights'])

            self.W_rnnlm_logits_1_fwd = LinearND(
                self.rnnlm_1_fwd.num_classes, decoder_num_units,
                dropout=dropout_decoder)
            self.W_rnnlm_gate_1_fwd = LinearND(
                decoder_num_units * 2, self.bottleneck_dim,
                dropout=dropout_decoder)

            # Fix RNNLM parameters
            if rnnlm_weight_sub == 0:
                for param in self.rnnlm_1_fwd.parameters():
                    param.requires_grad = False
        # TODO: backward RNNLM

        # Encoder
        # NOTE: overide encoder
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
                activation=activation,
                batch_norm=batch_norm)
            self.init_dec_state_0 = 'zero'
            self.init_dec_state_1 = 'zero'
        else:
            raise NotImplementedError

        self.is_bridge_sub = False
        if self.sub_loss_weight > 0:
            # Bridge layer between the encoder and decoder
            if encoder_type == 'cnn':
                self.bridge_1 = LinearND(
                    self.encoder.output_size, decoder_num_units_sub,
                    dropout=dropout_encoder)
                self.encoder_num_units_sub = decoder_num_units_sub
                self.is_bridge_sub = True
            elif bridge_layer:
                self.bridge_1 = LinearND(
                    self.encoder_num_units_sub, decoder_num_units_sub,
                    dropout=dropout_encoder)
                self.encoder_num_units_sub = decoder_num_units_sub
                self.is_bridge_sub = True
            else:
                self.is_bridge_sub = False

            # Initialization of the decoder
            if getattr(self, 'init_dec_state_1_' + dir_sub) != 'zero':
                setattr(self, 'W_dec_init_1_' + dir_sub, LinearND(
                    self.encoder_num_units_sub, decoder_num_units_sub))

            # Decoder (sub)
            decoder_input_size_sub = embedding_dim_sub
            if rnnlm_fusion_type in ['embedding_fusion', 'state_embedding_fusion']:
                decoder_input_size_sub += self.rnnlm_1_fwd.embedding_dim
            if decoding_order == 'conditional':
                setattr(self, 'decoder_first_1_' + dir_sub, RNNDecoder(
                    input_size=decoder_input_size_sub,
                    rnn_type=decoder_type,
                    num_units=decoder_num_units_sub,
                    num_layers=1,
                    dropout=dropout_decoder,
                    residual=False,
                    dense_residual=False))
                setattr(self, 'decoder_second_1_' + dir_sub, RNNDecoder(
                    input_size=self.encoder_num_units_sub,
                    rnn_type=decoder_type,
                    num_units=decoder_num_units_sub,
                    num_layers=1,
                    dropout=dropout_decoder,
                    residual=False,
                    dense_residual=False))
                # NOTE; the conditional decoder only supports the 1 layer
            else:
                setattr(self, 'decoder_1_' + dir_sub, RNNDecoder(
                    input_size=self.encoder_num_units_sub + decoder_input_size_sub,
                    rnn_type=decoder_type,
                    num_units=decoder_num_units_sub,
                    num_layers=decoder_num_layers_sub,
                    dropout=dropout_decoder,
                    residual=decoder_residual,
                    dense_residual=decoder_dense_residual))

            # Attention layer (sub)
            if num_heads_sub > 1:
                setattr(self, 'attend_1_' + dir_sub, MultiheadAttentionMechanism(
                    encoder_num_units=self.encoder_num_units_sub,
                    decoder_num_units=decoder_num_units_sub,
                    attention_type=attention_type,
                    attention_dim=attention_dim,
                    sharpening_factor=sharpening_factor,
                    sigmoid_smoothing=sigmoid_smoothing,
                    out_channels=attention_conv_num_channels,
                    kernel_size=attention_conv_width,
                    num_heads=num_heads_sub))
            else:
                setattr(self, 'attend_1_' + dir_sub, AttentionMechanism(
                    encoder_num_units=self.encoder_num_units_sub,
                    decoder_num_units=decoder_num_units_sub,
                    attention_type=attention_type,
                    attention_dim=attention_dim,
                    sharpening_factor=sharpening_factor,
                    sigmoid_smoothing=sigmoid_smoothing,
                    out_channels=attention_conv_num_channels,
                    kernel_size=attention_conv_width))

            # Output layer (sub)
            setattr(self, 'W_d_1_' + dir_sub, LinearND(
                decoder_num_units_sub, bottleneck_dim_sub,
                dropout=dropout_decoder))
            setattr(self, 'W_c_1_' + dir_sub, LinearND(
                self.encoder_num_units_sub, bottleneck_dim_sub,
                dropout=dropout_decoder))
            setattr(self, 'fc_1_' + dir_sub, LinearND(
                bottleneck_dim_sub, self.num_classes_sub))

            # Embedding (sub)
            self.embed_1 = Embedding(num_classes=self.num_classes_sub,
                                     embedding_dim=embedding_dim_sub,
                                     dropout=dropout_embedding,
                                     ignore_index=self.eos_1)

        # CTC (sub)
        if ctc_loss_weight_sub > 0:
            self.fc_ctc_1 = LinearND(
                self.encoder_num_units_sub, num_classes_sub + 1)

            # Set CTC decoders
            self._decode_ctc_greedy_np = GreedyDecoder(blank_index=0)
            self._decode_ctc_beam_np = BeamSearchDecoder(blank_index=0)
            # NOTE: index 0 is reserved for the blank class

        # Initialize weight matricess
        self.init_weights(parameter_init,
                          distribution=parameter_init_distribution,
                          ignore_keys=['bias'])

        # Initialize all biases with 0
        self.init_weights(0, distribution='constant', keys=['bias'])

        # Recurrent weights are orthogonalized
        if recurrent_weight_orthogonal:
            # encoder
            if encoder_type != 'cnn':
                self.init_weights(parameter_init,
                                  distribution='orthogonal',
                                  keys=[encoder_type, 'weight'],
                                  ignore_keys=['bias'])
            # decoder
            self.init_weights(parameter_init,
                              distribution='orthogonal',
                              keys=[decoder_type, 'weight'],
                              ignore_keys=['bias'])

        # Initialize bias in forget gate with 1
        if init_forget_gate_bias_with_one:
            self.init_forget_gate_bias_with_one()

    def forward(self, xs, ys, ys_sub, is_eval=False):
        """Forward computation.
        Args:
            xs (list): A list of length `[B]`, which contains arrays of size `[T, input_size]`
            ys (list): A list of lenght `[B]`, which contains arrays of size `[L]`
            ys_sub (list): A list of lenght `[B]`, which contains arrays of size `[L_sub]`
            is_eval (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (torch.autograd.Variable(float)): A tensor of size `[1]`
            loss_main (float):
            loss_sub (float):
            acc_main (float): Token-level accuracy in teacher-forcing in the main task
            acc_sub (float): Token-level accuracy in teacher-forcing in the sub task
        """
        if is_eval:
            self.eval()
        else:
            self.train()

            # Gaussian noise injection
            if self.weight_noise_injection:
                self.inject_weight_noise(mean=0, std=self.weight_noise_std)

        # Sort by lenghts in the descending order
        if is_eval and self.encoder_type != 'cnn' or self.input_type == 'text':
            perm_idx = sorted(list(range(0, len(xs), 1)),
                              key=lambda i: len(xs[i]), reverse=True)
            xs = [xs[i] for i in perm_idx]
            ys = [ys[i] for i in perm_idx]
            ys_sub = [ys_sub[i] for i in perm_idx]
            # NOTE: must be descending order for pack_padded_sequence
            # NOTE: assumed that xs is already sorted in the training stage

        # Encode input features
        xs, x_lens, xs_sub, x_lens_sub = self._encode(xs, is_multi_task=True)

        # Main task
        if self.main_loss_weight > 0:
            ys = [np2var(np.fromiter(y, dtype=np.int64), self.device_id).long()
                  for y in ys]
            loss, acc_main = self.compute_xe_loss(
                xs, ys, x_lens, task=0, dir='fwd')
            loss *= self.main_loss_weight
        else:
            loss = Variable(xs.data.new(1,).fill_(0.))
            acc_main = 0.
        loss_main = loss.data[0]

        # Sub task (attention)
        if self.sub_loss_weight > 0:
            if self.backward_1:
                ys_sub = [np2var(np.fromiter(y[::-1], dtype=np.int64), self.device_id).long()
                          for y in ys_sub]
            else:
                ys_sub = [np2var(np.fromiter(y, dtype=np.int64), self.device_id).long()
                          for y in ys_sub]
            loss_sub, acc_sub = self.compute_xe_loss(
                xs_sub, ys_sub, x_lens_sub,
                task=1, dir='bwd' if self.backward_1 else 'fwd')
            loss_sub *= self.sub_loss_weight
        else:
            loss_sub = Variable(xs.data.new(1,).fill_(0.))
            acc_sub = 0.

        # Sub task (CTC)
        if self.ctc_loss_weight_sub > 0:
            ys_sub_ctc = [np2var(np.fromiter(y, dtype=np.int64), self.device_id).long()
                          for y in ys_sub]
            ctc_loss_sub = self.compute_ctc_loss(
                xs_sub, ys_sub_ctc, x_lens_sub, task=1)
            loss_sub += ctc_loss_sub * self.ctc_loss_weight_sub
        loss += loss_sub

        if not is_eval:
            # Update the probability of scheduled sampling
            self._step += 1
            if self.ss_prob > 0:
                self._ss_prob = min(
                    self.ss_prob, self.ss_prob / self.ss_max_step * self._step)

        return loss, loss_main, loss_sub.data[0], acc_main, acc_sub

    def decode(self, xs, beam_width, max_decode_len, min_decode_len=0, min_decode_len_ratio=0,
               length_penalty=0, coverage_penalty=0, rnnlm_weight=0,
               task_index=0, joint_decoding=False, space_index=-1, oov_index=-1,
               word2char=None, score_sub_weight=0, entropy_threshold=1,
               idx2word=None, idx2char=None, rnnlm_weight_sub=0, exclude_eos=True):
        """Decoding in the inference stage.
        Args:
            xs (list): A list of length `[B]`, which contains arrays of size `[T, input_size]`
            beam_width (int): the size of beam
            max_decode_len (int): the maximum sequence length of tokens
            min_decode_len (int): the minimum sequence length of tokens
            min_decode_len_ratio (float):
            length_penalty (float): length penalty
            coverage_penalty (float): coverage penalty
            rnnlm_weight (float): the weight of RNNLM score
            task_index (int): the index of a task
            joint_decoding (bool): None or onepass or rescoring
            space_index (int):
            oov_index (int):
            word2char ():
            score_sub_weight (float):
            entropy_threshold (float):
            idx2word: for debug
            idx2char: for debug
            rnnlm_weight_sub (float): the weight of RNNLM score of the sub task
            exclude_eos (bool): if True, exclude <EOS> from best_hyps
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B]`
            aw (np.ndarray): A tensor of size `[B, L, T]`
            perm_idx (list): A list of length `[B]`
        """
        self.eval()

        if task_index > 0 and self.ctc_loss_weight_sub > self.sub_loss_weight:
            # Decode by CTC decoder
            best_hyps, perm_idx = self.decode_ctc(xs, beam_width, task_index)

            return best_hyps, None, perm_idx
            # NOTE: None corresponds to aw in attention-based models
        else:
            # Sort by lenghts in the descending order
            if self.encoder_type != 'cnn' or self.input_type == 'text':
                perm_idx = sorted(list(range(0, len(xs), 1)),
                                  key=lambda i: len(xs[i]), reverse=True)
                xs = [xs[i] for i in perm_idx]
                # NOTE: must be descending order for pack_padded_sequence
            else:
                perm_idx = list(range(0, len(xs), 1))

            dir = 'bwd' if task_index == 1 and self.backward_1 else 'fwd'

            # Encode input features
            if joint_decoding and task_index == 0 and dir == 'fwd':
                enc_out, x_lens, enc_out_sub, x_lens_sub = self._encode(
                    xs, is_multi_task=True)
            elif task_index == 0:
                enc_out, x_lens, _, _ = self._encode(xs, is_multi_task=True)
            elif task_index == 1:
                _, _, enc_out, x_lens = self._encode(xs, is_multi_task=True)
            else:
                raise NotImplementedError

            # Decode by attention decoder
            if joint_decoding and task_index == 0 and dir == 'fwd':
                best_hyps, aw, best_hyps_sub, aw_sub, = self._decode_infer_joint(
                    enc_out, x_lens,
                    enc_out_sub, x_lens_sub,
                    beam_width, max_decode_len, min_decode_len, min_decode_len_ratio,
                    length_penalty, coverage_penalty, rnnlm_weight, rnnlm_weight_sub,
                    space_index, oov_index, word2char, score_sub_weight, entropy_threshold,
                    idx2word, idx2char, exclude_eos)

                return best_hyps, aw, best_hyps_sub, aw_sub, perm_idx
            else:
                if beam_width == 1:
                    best_hyps, aw = self._decode_infer_greedy(
                        enc_out, x_lens, max_decode_len, task_index, dir, exclude_eos)
                else:
                    best_hyps, aw = self._decode_infer_beam(
                        enc_out, x_lens, beam_width, max_decode_len, min_decode_len,
                        min_decode_len_ratio, length_penalty, coverage_penalty,
                        rnnlm_weight, task_index, dir, exclude_eos)

            return best_hyps, aw, perm_idx

    def _decode_infer_joint(self, enc_out, x_lens, enc_out_sub, x_lens_sub,
                            beam_width, max_decode_len, min_decode_len, min_decode_len_ratio,
                            length_penalty, coverage_penalty, rnnlm_weight, rnnlm_weight_sub,
                            space_index, oov_index, word2char, score_sub_weight, entropy_threshold,
                            idx2word, idx2char, exclude_eos):
        """Joint decoding (one-pass).
        Args:
            enc_out (torch.FloatTensor): A tensor of size
                `[B, T, encoder_num_units]`
            x_lens (list): A list of length `[B]`
            enc_out_sub (torch.FloatTensor): A tensor of size
                `[B, T_in_sub, encoder_num_units]`
            x_lens_sub (list): A list of length `[B]`
            beam_width (int): the size of beam in the main task
            max_decode_len (int): the maximum sequence length of tokens
            min_decode_len (int): the minimum sequence length of tokens
            min_decode_len_ratio (float):
            length_penalty (float): length penalty
            coverage_penalty (float): coverage penalty
            rnnlm_weight (float): the weight of RNNLM score of the main task
            rnnlm_weight_sub (float): the weight of RNNLM score of the sub task
            space_index (int):
            oov_index (int):
            word2char ():
            score_sub_weight (float):
            entropy_threshold (flaot):
            idx2word (): for debug
            idx2char (): for debug
            exclude_eos (bool): if True, exclude <EOS> from best_hyps
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B, L]`
            aw (np.ndarray): A tensor of size `[B, L, T]`
            best_hyps_sub (np.ndarray): A tensor of size `[B, L_sub]`
            aw_sub (np.ndarray): A tensor of size `[B, L_sub, T]`
            aw_dec (np.ndarray): A tensor of size `[B, L, L_sub]`
        """
        debug = False

        batch_size, max_time = enc_out.size()[:2]

        if (rnnlm_weight > 0 or self.rnnlm_fusion_type) and self.rnnlm_0_fwd is not None:
            assert not self.rnnlm_0_fwd.training
        if (rnnlm_weight_sub > 0 and self.rnnlm_fusion_type)and self.rnnlm_1_fwd is not None:
            assert not self.rnnlm_1_fwd.training

        best_hyps, aw = [], []
        best_hyps_sub, aw_sub = [], []
        eos_flags = [False] * batch_size
        eos_flags_sub = [False] * batch_size
        for b in range(batch_size):
            # Initialization for the word model per utterance
            dec_out, hx_list, cx_list = self._init_dec_state(
                enc_out[b:b + 1], x_lens[b], task=0, dir='fwd')
            context_vec = Variable(enc_out.data.new(
                1, 1, enc_out.size(-1)).fill_(0.), volatile=True)
            self.attend_0_fwd.reset()

            dec_out_sub, hx_list_sub, cx_list_sub = self._init_dec_state(
                enc_out_sub[b:b + 1], x_lens_sub[b], task=1, dir='fwd')
            context_vec_sub = Variable(enc_out.data.new(
                1, 1, enc_out_sub.size(-1)).fill_(0.), volatile=True)
            self.attend_1_fwd.reset()

            complete = []
            beam = [{'hyp': [self.sos_0],
                     'hyp_sub': [self.sos_1],
                     'score': 0,  # log1
                     'score_sub': 0,  # log 1
                     'cx_list': cx_list,
                     'hx_list': hx_list,
                     'dec_out': dec_out,
                     'dec_out_sub': dec_out_sub,
                     'cx_list_sub': cx_list_sub,
                     'hx_list_sub': hx_list_sub,
                     'context_vec': context_vec,
                     'context_vec_sub': context_vec_sub,
                     'aw_steps': [None],
                     'aw_steps_sub':[None],
                     'rnnlm_state': None,
                     'rnnlm_state_sub': None,
                     'previous_coverage': 0}]
            for t in range(max_decode_len + 1):
                new_beam = []
                for i_beam in range(len(beam)):
                    # Update RNNLM states
                    if (rnnlm_weight > 0 or self.rnnlm_fusion_type) and self.rnnlm_0_fwd is not None:
                        y_rnnlm = Variable(enc_out.data.new(
                            1, 1).fill_(beam[i_beam]['hyp'][-1]).long(), volatile=True)
                        y_rnnlm = self.rnnlm_0_fwd.embed(y_rnnlm)
                        rnnlm_logits_step, rnnlm_out, rnnlm_state = self.rnnlm_0_fwd.predict(
                            y_rnnlm, h=beam[i_beam]['rnnlm_state'])

                    y = Variable(enc_out.data.new(
                        1, 1).fill_(beam[i_beam]['hyp'][-1]).long(), volatile=True)
                    y = self.embed_0(y)

                    if t == 0:
                        dec_out = beam[i_beam]['dec_out']
                    else:
                        # Recurrency
                        dec_in = torch.cat(
                            [y,  beam[i_beam]['context_vec']], dim=-1)
                        dec_out, hx_list, cx_list = self.decoder_0_fwd(
                            dec_in, beam[i_beam]['hx_list'], beam[i_beam]['cx_list'])

                    # Score
                    context_vec, aw_step = self.attend_0_fwd(
                        enc_out[b:b + 1, :x_lens[b]], x_lens[b:b + 1],
                        dec_out, beam[i_beam]['aw_steps'][-1])

                    # Generate
                    out = self.W_d_0_fwd(dec_out) + \
                        self.W_c_0_fwd(context_vec)
                    logits_step = self.fc_0_fwd(F.tanh(out))

                    # Path through the log-softmax layer
                    log_probs = F.log_softmax(logits_step.squeeze(1), dim=1)

                    ent = entropy(F.softmax(logits_step, dim=-
                                            1).data.cpu().numpy()[0, 0])

                    # Pick up the top-k scores
                    log_probs_topk, indices_topk = log_probs.topk(
                        beam_width, dim=1, largest=True, sorted=True)

                    for k in range(beam_width):
                        # Exclude short hypotheses
                        if indices_topk[0, k].data[0] == self.eos_0 and len(beam[i_beam]['hyp']) < min_decode_len:
                            continue
                        if indices_topk[0, k].data[0] == self.eos_0 and len(beam[i_beam]['hyp']) < x_lens[b] * min_decode_len_ratio:
                            continue

                        # Add length penalty
                        score = beam[i_beam]['score'] + \
                            log_probs_topk.data[0, k] + length_penalty

                        # Add coverage penalty
                        if coverage_penalty > 0:
                            # Recompute converage penalty in each step
                            score -= beam[i_beam]['previous_coverage'] * \
                                coverage_penalty

                            cov_threshold = 0.5
                            # TODO: make this parameter
                            aw_steps = torch.stack(
                                beam[i_beam]['aw_steps'][1:] + [aw_step], dim=1)

                            if self.num_heads_0 > 1:
                                cov_sum = aw_steps.data[0,
                                                        :, :, 0].cpu().numpy()
                                # TODO: fix for MHA
                            else:
                                cov_sum = aw_steps.data[0].cpu().numpy()
                            cov_sum = np.sum(cov_sum[np.where(
                                cov_sum > cov_threshold)[0]])
                            score += cov_sum * coverage_penalty
                        else:
                            cov_sum = 0

                        # Add RNNLM score
                        if rnnlm_weight > 0 and self.rnnlm_0_fwd is not None:
                            rnnlm_log_probs = F.log_softmax(
                                rnnlm_logits_step.squeeze(1), dim=1)
                            assert log_probs.size() == rnnlm_log_probs.size()
                            score += rnnlm_log_probs.data[0,
                                                          indices_topk.data[0, k]] * rnnlm_weight
                        else:
                            rnnlm_state = None

                        # NOTE: Resocre by the second decoder's score
                        oov_flag = indices_topk.data[0, k] == oov_index
                        eos_flag = indices_topk.data[0, k] == self.eos_0
                        score_c2w = beam[i_beam]['score_sub']
                        score_c2w_until_space = beam[i_beam]['score_sub']
                        word_idx = indices_topk.data[0, k]

                        if oov_flag:
                            # NOTE: Decode until outputting a space
                            t_sub = 0
                            dec_outs_sub = [beam[i_beam]['dec_out_sub']]
                            hx_list_sub_list = [beam[i_beam]['hx_list_sub']]
                            cx_list_sub_list = [beam[i_beam]['cx_list_sub']]
                            aw_steps_sub = [beam[i_beam]['aw_steps_sub'][-1]]
                            charseq = []
                            # TODO: add max OOV len
                            rnnlm_state_sub = beam[i_beam]['rnnlm_state_sub']
                            while True:
                                # Score
                                context_vec_sub, aw_step_sub = self.attend_1_fwd(
                                    enc_out_sub[b:b + 1, :x_lens_sub[b]],
                                    x_lens_sub[b:b + 1],
                                    dec_outs_sub[-1], aw_steps_sub[-1])

                                # Generate
                                logits_step_sub = self.fc_1_fwd(F.tanh(
                                    self.W_d_1_fwd(dec_outs_sub[-1]) +
                                    self.W_c_1_fwd(context_vec_sub)))

                                # Path through the log-softmax layer
                                log_probs_sub = F.log_softmax(
                                    logits_step_sub.squeeze(1), dim=-1)

                                if t_sub == 0 and t > 0:
                                    # space before the word
                                    score_c2w += log_probs_sub.data[0,
                                                                    space_index]
                                    score_c2w_until_space += log_probs_sub.data[0,
                                                                                space_index]

                                    # Add RNNLM score in the sub task
                                    if rnnlm_weight_sub > 0 and self.rnnlm_1_fwd is not None:
                                        y_rnnlm_sub = Variable(enc_out.data.new(
                                            1, 1).fill_(beam[i_beam]['hyp_sub'][-1]).long(), volatile=True)
                                        y_rnnlm_sub = self.rnnlm_1_fwd.embed(
                                            y_rnnlm_sub)
                                        rnnlm_logits_step_sub, rnnlm_out_sub, rnnlm_state_sub = self.rnnlm_1_fwd.predict(
                                            y_rnnlm_sub, h=rnnlm_state_sub)
                                        rnnlm_log_probs_sub = F.log_softmax(
                                            rnnlm_logits_step_sub.squeeze(1), dim=1)
                                        assert log_probs_sub.size() == rnnlm_log_probs_sub.size()
                                        score_c2w += rnnlm_log_probs_sub.data[0,
                                                                              space_index] * rnnlm_weight_sub / score_sub_weight
                                        score_c2w_until_space += rnnlm_log_probs_sub.data[0,
                                                                                          space_index] * rnnlm_weight_sub / score_sub_weight
                                    else:
                                        rnnlm_state_sub = None

                                    charseq += [space_index]
                                    y_sub = Variable(enc_out.data.new(
                                        1, 1).fill_(space_index).long(), volatile=True)
                                else:
                                    y_sub = torch.max(log_probs_sub, dim=1)[
                                        1].data[0]

                                    if y_sub == space_index:
                                        break
                                    if y_sub == self.eos_1:
                                        break
                                    if t_sub > 20:
                                        break

                                    # Add RNNLM score in the sub task
                                    if rnnlm_weight_sub > 0 and self.rnnlm_1_fwd is not None:
                                        if t == 0:
                                            y_rnnlm_sub = Variable(enc_out.data.new(
                                                1, 1).fill_(self.sos_1).long(), volatile=True)
                                        else:
                                            y_rnnlm_sub = Variable(enc_out.data.new(
                                                1, 1).fill_(charseq[t_sub - 1]).long(), volatile=True)
                                        y_rnnlm_sub = self.rnnlm_1_fwd.embed(
                                            y_rnnlm_sub)
                                        rnnlm_logits_step_sub, rnnlm_out_sub, rnnlm_state_sub = self.rnnlm_1_fwd.predict(
                                            y_rnnlm_sub, h=rnnlm_state_sub)
                                        rnnlm_log_probs_sub = F.log_softmax(
                                            rnnlm_logits_step_sub.squeeze(1), dim=1)
                                        assert log_probs_sub.size() == rnnlm_log_probs_sub.size()
                                        score_c2w += rnnlm_log_probs_sub.data[0,
                                                                              y_sub] * rnnlm_weight_sub / score_sub_weight
                                    else:
                                        rnnlm_state_sub = None

                                    score_c2w += log_probs_sub.data[0, y_sub]
                                    charseq += [y_sub]
                                    y_sub = Variable(enc_out.data.new(
                                        1, 1).fill_(y_sub).long(), volatile=True)

                                y_sub = self.embed_1(y_sub)

                                if ent > entropy_threshold and debug:
                                    print(idx2word([word_idx]))
                                    print(idx2char(charseq))

                                # Recurrency
                                dec_in_sub = torch.cat(
                                    [y_sub, context_vec_sub], dim=-1)
                                dec_out_sub, hx_list_sub, cx_list_sub = self.decoder_1_fwd(
                                    dec_in_sub, hx_list_sub_list[-1], cx_list_sub_list[-1])

                                dec_outs_sub += [dec_out_sub]
                                hx_list_sub_list += [hx_list_sub]
                                cx_list_sub_list += [cx_list_sub]
                                aw_steps_sub += [aw_step_sub]
                                t_sub += 1

                            aw_steps_sub = aw_steps_sub[1:]  # remove start aw
                            dec_out_sub = dec_outs_sub[-1]
                            hx_list_sub = hx_list_sub_list[-1]
                            cx_list_sub = cx_list_sub_list[-1]

                        elif eos_flag:
                            # Score
                            context_vec_sub, aw_step_sub = self.attend_1_fwd(
                                enc_out_sub[b:b + 1, :x_lens_sub[b]],
                                x_lens_sub[b:b + 1],
                                beam[i_beam]['dec_out_sub'],
                                beam[i_beam]['aw_steps_sub'][-1])

                            # Generate
                            logits_step_sub = self.fc_1_fwd(F.tanh(
                                self.W_d_1_fwd(dec_out_sub) +
                                self.W_c_1_fwd(context_vec_sub)))

                            # Path through the log-softmax layer
                            log_probs_sub = F.log_softmax(
                                logits_step_sub.squeeze(1), dim=-1)

                            # Add RNNLM score in the sub task
                            if rnnlm_weight_sub > 0 and self.rnnlm_1_fwd is not None:
                                y_rnnlm_sub = Variable(enc_out.data.new(
                                    1, 1).fill_(beam[i_beam]['hyp_sub'][-1]).long(), volatile=True)
                                y_rnnlm_sub = self.rnnlm_1_fwd.embed(
                                    y_rnnlm_sub)
                                rnnlm_logits_step_sub, rnnlm_out_sub, rnnlm_state_sub = self.rnnlm_1_fwd.predict(
                                    y_rnnlm_sub, h=beam[i_beam]['rnnlm_state_sub'])
                                rnnlm_log_probs_sub = F.log_softmax(
                                    rnnlm_logits_step_sub.squeeze(1), dim=1)
                                assert log_probs_sub.size() == rnnlm_log_probs_sub.size()
                                score_c2w += rnnlm_log_probs_sub.data[0,
                                                                      self.eos_1] * rnnlm_weight_sub / score_sub_weight
                            else:
                                rnnlm_state_sub = None

                            charseq = [self.eos_1]
                            score_c2w += log_probs_sub.data[0, self.eos_1]
                            aw_steps_sub = [aw_step_sub]

                            if ent > entropy_threshold and debug:
                                print(idx2word([word_idx]))
                                print(idx2char([self.eos_1]))

                        else:
                            # Decompose a word to characters
                            charseq = word2char(word_idx)
                            # charseq: `[num_chars,]` (list)

                            if t > 0:
                                charseq = [space_index] + charseq

                            if ent > entropy_threshold and debug:
                                print(idx2word([word_idx]))
                                print(idx2char(charseq))

                            dec_out_sub = beam[i_beam]['dec_out_sub']
                            hx_list_sub = beam[i_beam]['hx_list_sub']
                            cx_list_sub = beam[i_beam]['cx_list_sub']
                            aw_step_sub = beam[i_beam]['aw_steps_sub'][-1]
                            aw_steps_sub = []
                            rnnlm_state_sub = beam[i_beam]['rnnlm_state_sub']
                            for t_sub in range(len(charseq)):
                                # Score
                                context_vec_sub, aw_step_sub = self.attend_1_fwd(
                                    enc_out_sub[b:b + 1, :x_lens_sub[b]],
                                    x_lens_sub[b:b + 1],
                                    dec_out_sub, aw_step_sub)

                                # Generate
                                logits_step_sub = self.fc_1_fwd(F.tanh(
                                    self.W_d_1_fwd(dec_out_sub) +
                                    self.W_c_1_fwd(context_vec_sub)))

                                # Path through the log-softmax layer
                                log_probs_sub = F.log_softmax(
                                    logits_step_sub.squeeze(1), dim=-1)

                                score_c2w += log_probs_sub.data[0,
                                                                charseq[t_sub]]
                                if t_sub == 0 and t > 0:
                                    score_c2w_until_space += log_probs_sub.data[0,
                                                                                charseq[t_sub]]

                                # Add RNNLM score in the sub task
                                if rnnlm_weight_sub > 0 and self.rnnlm_1_fwd is not None:
                                    if t_sub == 0:
                                        y_rnnlm_sub = Variable(enc_out.data.new(
                                            1, 1).fill_(beam[i_beam]['hyp_sub'][-1]).long(), volatile=True)
                                    else:
                                        y_rnnlm_sub = Variable(enc_out.data.new(
                                            1, 1).fill_(charseq[t_sub - 1]).long(), volatile=True)
                                    y_rnnlm_sub = self.rnnlm_1_fwd.embed(
                                        y_rnnlm_sub)
                                    rnnlm_logits_step_sub, rnnlm_out_sub, rnnlm_state_sub = self.rnnlm_1_fwd.predict(
                                        y_rnnlm_sub, h=rnnlm_state_sub)
                                    rnnlm_log_probs_sub = F.log_softmax(
                                        rnnlm_logits_step_sub.squeeze(1), dim=1)
                                    assert log_probs_sub.size() == rnnlm_log_probs_sub.size()
                                    score_c2w += rnnlm_log_probs_sub.data[0,
                                                                          charseq[t_sub]] * rnnlm_weight_sub / score_sub_weight
                                    if t_sub == 0 and t > 0:
                                        score_c2w_until_space += rnnlm_log_probs_sub.data[0,
                                                                                          charseq[t_sub]] * rnnlm_weight_sub / score_sub_weight
                                else:
                                    rnnlm_state_sub = None

                                aw_steps_sub += [aw_step_sub]

                                # teacher-forcing
                                y_sub = Variable(enc_out.data.new(
                                    1, 1).fill_(charseq[t_sub]).long(), volatile=True)
                                y_sub = self.embed_1(y_sub)

                                # Recurrency
                                dec_in_sub = torch.cat(
                                    [y_sub, context_vec_sub], dim=-1)
                                dec_out_sub, hx_list_sub, cx_list_sub = self.decoder_1_fwd(
                                    dec_in_sub, hx_list_sub, cx_list_sub)

                        # Rescoreing
                        if ent > entropy_threshold:
                            score += (score_c2w - score_c2w_until_space +
                                      math.log(len(charseq)) * 0.1) * score_sub_weight
                            # NOTE: consider length of characters

                        new_beam.append(
                            {'hyp': beam[i_beam]['hyp'] + [indices_topk.data[0, k]],
                             'hyp_sub': beam[i_beam]['hyp_sub'] + charseq,
                             'score': score,
                             'score_sub': score_c2w,
                             'dec_out': dec_out,
                             'hx_list': copy.deepcopy(hx_list),
                             'cx_list': copy.deepcopy(cx_list),
                             'dec_out_sub': dec_out_sub,
                             'hx_list_sub': hx_list_sub,
                             'cx_list_sub': cx_list_sub,
                             'context_vec': context_vec,
                             'context_vec_sub': context_vec_sub,
                             'aw_steps': beam[i_beam]['aw_steps'] + [aw_step],
                             'aw_steps_sub': beam[i_beam]['aw_steps_sub'] + aw_steps_sub,
                             'rnnlm_state': rnnlm_state,
                             'rnnlm_state_sub': rnnlm_state_sub,
                             'previous_coverage': cov_sum})

                new_beam = sorted(
                    new_beam, key=lambda x: x['score'], reverse=True)

                # Remove complete hypotheses
                not_complete = []
                for cand in new_beam[:beam_width]:
                    if cand['hyp'][-1] == self.eos_0:
                        complete += [cand]
                    else:
                        not_complete += [cand]
                if len(complete) >= beam_width:
                    complete = complete[:beam_width]
                    break
                beam = not_complete[:beam_width]

            if len(complete) == 0:
                complete = beam

            complete = sorted(
                complete, key=lambda x: x['score'], reverse=True)
            best_hyps += [np.array(complete[0]['hyp'][1:])]
            aw += [complete[0]['aw_steps'][1:]]
            best_hyps_sub += [np.array(complete[0]['hyp_sub'][1:])]
            aw_sub += [complete[0]['aw_steps_sub'][1:]]
            if complete[0]['hyp'][-1] == self.eos_0:
                eos_flags[b] = True
            if complete[0]['hyp_sub'][-1] == self.eos_1:
                eos_flags_sub[b] = True

            if debug:
                print(complete[0]['score'])
                print(complete[0]['score_sub'] * score_sub_weight)

        # Concatenate in L dimension
        for b in range(len(aw)):
            aw_sub[b] = var2np(torch.stack(aw_sub[b], dim=1).squeeze(0))
            if self.num_heads_1 > 1:
                aw_sub[b] = aw_sub[b][:, :, 0]
                # TODO: fix for MHA

            aw[b] = var2np(torch.stack(aw[b], dim=1).squeeze(0))
            if self.num_heads_0 > 1:
                aw[b] = aw[b][:, :, 0]
                # TODO: fix for MHA

        # Exclude <EOS>
        if exclude_eos:
            best_hyps = [best_hyps[b][:-1] if eos_flags[b]
                         else best_hyps[b] for b in range(batch_size)]
            best_hyps_sub = [best_hyps_sub[b][:-1] if eos_flags_sub[b]
                             else best_hyps_sub[b] for b in range(batch_size)]

        return best_hyps, aw, best_hyps_sub, aw_sub
