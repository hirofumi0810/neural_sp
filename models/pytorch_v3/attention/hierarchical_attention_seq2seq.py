#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Hierarchical attention-based sequence-to-sequence model (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
import torch
import torch.nn.functional as F

from models.pytorch_v3.attention.attention_seq2seq import AttentionSeq2seq
from models.pytorch_v3.linear import LinearND, Embedding, Embedding_LS
from models.pytorch_v3.encoders.load_encoder import load
from models.pytorch_v3.attention.rnn_decoder import RNNDecoder
from models.pytorch_v3.attention.attention_layer import AttentionMechanism
from models.pytorch_v3.ctc.decoders.greedy_decoder import GreedyDecoder
from models.pytorch_v3.ctc.decoders.beam_search_decoder import BeamSearchDecoder


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
                 init_dec_state='first',
                 sharpening_factor=1,
                 logits_temperature=1,
                 sigmoid_smoothing=False,
                 coverage_weight=0,
                 ctc_loss_weight_sub=0,  # ***
                 attention_conv_num_channels=10,
                 attention_conv_width=201,
                 num_stack=1,
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
                 num_heads_sub=1):  # ***

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
            splice=splice,
            input_channel=input_channel,
            conv_channels=conv_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            conv_strides=conv_strides,
            poolings=poolings,
            scheduled_sampling_prob=scheduled_sampling_prob,
            scheduled_sampling_max_step=scheduled_sampling_max_step,
            label_smoothing_prob=label_smoothing_prob,
            weight_noise_std=weight_noise_std,
            encoder_residual=encoder_residual,
            encoder_dense_residual=encoder_dense_residual,
            decoder_residual=decoder_residual,
            decoder_dense_residual=decoder_dense_residual,
            decoding_order=decoding_order,
            bottleneck_dim=bottleneck_dim,
            backward_loss_weight=0,
            num_heads=num_heads)
        self.model_type = 'hierarchical_attention'

        # Setting for the encoder
        self.encoder_num_units_sub = encoder_num_units
        if encoder_bidirectional:
            self.encoder_num_units_sub *= 2

        # Setting for the decoder in the sub task
        self.decoder_num_units_1 = decoder_num_units_sub
        self.decoder_num_layers_1 = decoder_num_layers_sub
        self.num_classes_sub = num_classes_sub + 1  # Add <EOS> class
        self.sos_1 = num_classes_sub
        self.eos_1 = num_classes_sub
        # NOTE: <SOS> and <EOS> have the same index
        self.backward_1 = backward_sub

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

        dir = 'bwd' if backward_sub else 'fwd'
        self.is_bridge_sub = False
        if self.sub_loss_weight > 0:
            ##################################################
            # Bridge layer between the encoder and decoder
            ##################################################
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

            ##################################################
            # Initialization of the decoder
            ##################################################
            if getattr(self, 'init_dec_state_1_' + dir) != 'zero':
                setattr(self, 'W_dec_init_1_' + dir, LinearND(
                    self.encoder_num_units_sub, decoder_num_units_sub))

            ##############################
            # Decoder (sub)
            ##############################
            if decoding_order == 'conditional':
                setattr(self, 'decoder_first_1_' + dir, RNNDecoder(
                    input_size=embedding_dim_sub,
                    rnn_type=decoder_type,
                    num_units=decoder_num_units_sub,
                    num_layers=1,
                    dropout=dropout_decoder,
                    residual=False,
                    dense_residual=False))
                setattr(self, 'decoder_second_1_' + dir, RNNDecoder(
                    input_size=self.encoder_num_units_sub,
                    rnn_type=decoder_type,
                    num_units=decoder_num_units_sub,
                    num_layers=1,
                    dropout=dropout_decoder,
                    residual=False,
                    dense_residual=False))
                # NOTE; the conditional decoder only supports the 1 layer
            else:
                setattr(self, 'decoder_1_' + dir, RNNDecoder(
                    input_size=self.encoder_num_units_sub + embedding_dim_sub,
                    rnn_type=decoder_type,
                    num_units=decoder_num_units_sub,
                    num_layers=decoder_num_layers_sub,
                    dropout=dropout_decoder,
                    residual=decoder_residual,
                    dense_residual=decoder_dense_residual))

            ###################################
            # Attention layer (sub)
            ###################################
            setattr(self, 'attend_1_' + dir, AttentionMechanism(
                encoder_num_units=self.encoder_num_units_sub,
                decoder_num_units=decoder_num_units_sub,
                attention_type=attention_type,
                attention_dim=attention_dim,
                sharpening_factor=sharpening_factor,
                sigmoid_smoothing=sigmoid_smoothing,
                out_channels=attention_conv_num_channels,
                kernel_size=attention_conv_width,
                num_heads=num_heads_sub))

            ##############################
            # Output layer (sub)
            ##############################
            setattr(self, 'W_d_1_' + dir, LinearND(
                decoder_num_units_sub, bottleneck_dim_sub,
                dropout=dropout_decoder))
            setattr(self, 'W_c_1_' + dir, LinearND(
                self.encoder_num_units_sub, bottleneck_dim_sub,
                dropout=dropout_decoder))
            setattr(self, 'fc_1_' + dir, LinearND(
                bottleneck_dim_sub, self.num_classes_sub))

            ##############################
            # Embedding (sub)
            ##############################
            if label_smoothing_prob > 0:
                self.embed_1 = Embedding_LS(
                    num_classes=self.num_classes_sub,
                    embedding_dim=embedding_dim_sub,
                    dropout=dropout_embedding,
                    label_smoothing_prob=label_smoothing_prob)
            else:
                self.embed_1 = Embedding(
                    num_classes=self.num_classes_sub,
                    embedding_dim=embedding_dim_sub,
                    dropout=dropout_embedding,
                    ignore_index=-1)

        ##############################
        # CTC (sub)
        ##############################
        if ctc_loss_weight_sub > 0:
            self.fc_ctc_1 = LinearND(
                self.encoder_num_units_sub, num_classes_sub + 1)

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

    def forward(self, xs, ys, x_lens, y_lens, ys_sub, y_lens_sub, is_eval=False):
        """Forward computation.
        Args:
            xs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            ys (np.ndarray): A tensor of size `[B, T_out]`
            x_lens (np.ndarray): A tensor of size `[B]`
            y_lens (np.ndarray): A tensor of size `[B]`
            ys_sub (np.ndarray): A tensor of size `[B, T_out_sub]`
            y_lens_sub (np.ndarray): A tensor of size `[B]`
            is_eval (bool): if True, the history will not be saved.
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

        # Reverse the order
        if self.backward_1:
            ys_sub_tmp = copy.deepcopy(ys_sub)
            for b in range(len(xs)):
                ys_sub_tmp[b, :y_lens_sub[b]] = ys_sub[b, :y_lens_sub[b]][::-1]
        else:
            ys_sub_tmp = ys_sub

        # NOTE: ys and ys_sub are padded with -1 here
        # ys_in and ys_in_sub areb padded with <EOS> in order to convert to
        # one-hot vector, and added <SOS> before the first token
        # ys_out and ys_out_sub are padded with -1, and added <EOS>
        # after the last token
        ys_in = self._create_var((ys.shape[0], ys.shape[1] + 1),
                                 fill_value=self.eos_0, dtype='long')
        ys_in_sub = self._create_var((ys_sub.shape[0], ys_sub.shape[1] + 1),
                                     fill_value=self.eos_1, dtype='long')
        ys_out = self._create_var((ys.shape[0], ys.shape[1] + 1),
                                  fill_value=-1, dtype='long')
        ys_out_sub = self._create_var((ys_sub.shape[0], ys_sub.shape[1] + 1),
                                      fill_value=-1, dtype='long')

        ys_in.data[:, 0] = self.sos_0
        ys_in_sub.data[:, 0] = self.sos_1
        for b in range(len(xs)):
            ys_in.data[b, 1:y_lens[b] + 1] = torch.from_numpy(
                ys[b, :y_lens[b]])
            ys_in_sub.data[b, 1:y_lens_sub[b] + 1] = torch.from_numpy(
                ys_sub_tmp[b, :y_lens_sub[b]])

            ys_out.data[b, :y_lens[b]] = torch.from_numpy(ys[b, :y_lens[b]])
            ys_out.data[b, y_lens[b]] = self.eos_0
            ys_out_sub.data[b, :y_lens_sub[b]] = torch.from_numpy(
                ys_sub_tmp[b, :y_lens_sub[b]])
            ys_out_sub.data[b, y_lens_sub[b]] = self.eos_1

        if self.use_cuda:
            ys_in = ys_in.cuda()
            ys_out = ys_out.cuda()
            ys_in_sub = ys_in_sub.cuda()
            ys_out_sub = ys_out_sub.cuda()

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

            ys_in_sub = ys_in_sub[perm_idx]
            ys_out_sub = ys_out_sub[perm_idx]
            y_lens_sub = y_lens_sub[perm_idx]

        ##################################################
        # Main task
        ##################################################
        if self.main_loss_weight > 0:
            # Compute XE loss
            loss_main = self.compute_xe_loss(
                xs, ys_in, ys_out, x_lens, y_lens,
                task=0, dir='fwd') * self.main_loss_weight
        else:
            loss_main = self._create_var((1,), fill_value=0.)
        loss = loss_main.clone()

        ##################################################
        # Sub task (attention)
        ##################################################
        if self.sub_loss_weight > 0:
            # Compute XE loss
            loss_sub = self.compute_xe_loss(
                xs_sub, ys_in_sub, ys_out_sub, x_lens_sub, y_lens_sub,
                task=1, dir='bwd' if self.backward_1 else 'fwd') * self.sub_loss_weight
            loss += loss_sub

        ##################################################
        # Sub task (CTC)
        ##################################################
        if self.ctc_loss_weight_sub > 0:
            # Wrap by Variable
            ys_ctc_sub = self.np2var(ys_sub, dtype='long')

            if self.use_cuda:
                ys_ctc_sub = ys_ctc_sub.cuda()

            # Permutate indices
            if perm_idx is not None:
                ys_ctc_sub = ys_ctc_sub[perm_idx]

            ctc_loss_sub = self.compute_ctc_loss(
                xs_sub, ys_ctc_sub + 1,
                x_lens_sub, y_lens_sub, task=1) * self.ctc_loss_weight_sub
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
            if self.ss_prob > 0:
                self._ss_prob = min(
                    self.ss_prob, self.ss_prob / self.ss_max_step * self._step)

        if self.sub_loss_weight > self.ctc_loss_weight_sub:
            return loss, loss_main, loss_sub
        else:
            return loss, loss_main, ctc_loss_sub

    def decode(self, xs, x_lens, beam_width, max_decode_len, min_decode_len=0,
               length_penalty=0, coverage_penalty=0, task_index=0,
               joint_decoding=None, space_index=-1, oov_index=-1,
               word2char=None, score_sub_weight=0, idx2word=None, idx2char=None):
        """Decoding in the inference stage.
        Args:
            xs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            x_lens (np.ndarray): A tensor of size `[B]`
            beam_width (int): the size of beam
            max_decode_len (int): the maximum sequence length of tokens
            min_decode_len (int): the minimum sequence length of tokens
            length_penalty (float): length penalty in beam search decoding
            coverage_penalty (float): coverage penalty in beam search decoding
            task_index (int): the index of a task
            joint_decoding (): None or onepass or rescoring
            space_index (int):
            oov_index (int):
            word2char ():
            score_sub_weight (float):
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B]`
            aw ():
            perm_idx (np.ndarray): A tensor of size `[B]`
        """
        self.eval()

        if task_index > 0 and self.ctc_loss_weight_sub > self.sub_loss_weight:
            # Decode by CTC decoder
            best_hyps, perm_idx = self.decode_ctc(
                xs, x_lens, beam_width, task_index)

            return best_hyps, None, perm_idx
            # NOTE: None corresponds to aw in attention-based models
        else:
            # Wrap by Variable
            xs = self.np2var(xs)
            x_lens = self.np2var(x_lens, dtype='int')

            dir = 'bwd' if task_index == 1 and self.backward_1 else 'fwd'

            # Encode acoustic features
            if joint_decoding is not None and task_index == 0 and dir == 'fwd':
                enc_out, x_lens, enc_out_sub, x_lens_sub, perm_idx = self._encode(
                    xs, x_lens, is_multi_task=True)
            elif task_index == 0:
                enc_out, x_lens, _, _, perm_idx = self._encode(
                    xs, x_lens, is_multi_task=True)
            elif task_index == 1:
                _, _, enc_out, x_lens, perm_idx = self._encode(
                    xs, x_lens, is_multi_task=True)
            else:
                raise NotImplementedError

            # Decode by attention decoder
            if joint_decoding is not None and task_index == 0 and dir == 'fwd':
                if joint_decoding == 'onepass':
                    best_hyps, aw, best_hyps_sub, aw_sub, = self._decode_infer_joint_onepass(
                        enc_out, x_lens,
                        enc_out_sub, x_lens_sub,
                        beam_width, max_decode_len, min_decode_len,
                        length_penalty, coverage_penalty,
                        space_index, oov_index, word2char, score_sub_weight,
                        idx2word, idx2char)
                elif joint_decoding == 'rescoring':
                    best_hyps, aw, best_hyps_sub, aw_sub, = self._decode_infer_joint_rescoring(
                        enc_out, x_lens,
                        enc_out_sub, x_lens_sub,
                        beam_width, max_decode_len, min_decode_len,
                        length_penalty, coverage_penalty,
                        space_index, oov_index, word2char, score_sub_weight,
                        idx2word, idx2char)
                else:
                    raise ValueError(joint_decoding)

                # Permutate indices to the original order
                if perm_idx is None:
                    perm_idx = np.arange(0, len(xs), 1)
                else:
                    perm_idx = self.var2np(perm_idx)

                return best_hyps, aw, best_hyps_sub, aw_sub, perm_idx
            else:
                if beam_width == 1:
                    best_hyps, aw = self._decode_infer_greedy(
                        enc_out, x_lens, max_decode_len, task_index, dir)
                else:
                    best_hyps, aw = self._decode_infer_beam(
                        enc_out, x_lens, beam_width, max_decode_len, min_decode_len,
                        length_penalty, coverage_penalty, task_index, dir)

            # TODO: fix this
            if beam_width == 1:
                aw = aw[:, :, :, 0]

            # Permutate indices to the original order
            if perm_idx is None:
                perm_idx = np.arange(0, len(xs), 1)
            else:
                perm_idx = self.var2np(perm_idx)

            return best_hyps, aw, perm_idx

    def _decode_infer_joint_onepass(self, enc_out, x_lens, enc_out_sub, x_lens_sub,
                                    beam_width, max_decode_len, min_decode_len,
                                    length_penalty, coverage_penalty,
                                    space_index, oov_index, word2char, score_sub_weight,
                                    idx2word, idx2char):
        """Joint decoding (one-pass).
        Args:
            enc_out (torch.FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            x_lens (torch.IntTensor): A tensor of size `[B]`
            enc_out_sub (torch.FloatTensor): A tensor of size
                `[B, T_in_sub, encoder_num_units]`
            x_lens_sub (torch.IntTensor): A tensor of size `[B]`
            beam_width (int): the size of beam in the main task
            max_decode_len (int): the maximum sequence length of tokens
            min_decode_len (int): the minimum sequence length of tokens
            length_penalty (float): length penalty in beam search decoding
            coverage_penalty (float): coverage penalty in beam search decoding
            space_index (int):
            oov_index (int):
            word2char ():
            score_sub_weight (float):
            idx2word (): for debug
            idx2char (): for debug
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B, T_out]`
            aw (np.ndarray): A tensor of size `[B, T_out, T_in]`
            best_hyps_sub (np.ndarray): A tensor of size `[B, T_out_sub]`
            aw_sub (np.ndarray): A tensor of size `[B, T_out_sub, T_in]`
            aw_dec (np.ndarray): A tensor of size `[B, T_out, T_out_sub]`
        """
        batch_size, max_time = enc_out.size()[:2]

        # Pre-computation of encoder-side features computing scores
        enc_out_a, enc_out_sub_a = [], []
        for h in range(self.num_heads_0):
            enc_out_a += [getattr(self.attend_0_fwd,
                                  'W_enc_head' + str(h))(enc_out)]
        for h in range(self.num_heads_1):
            enc_out_sub_a += [getattr(self.attend_1_fwd,
                                      'W_enc_head' + str(h))(enc_out_sub)]
        enc_out_a = torch.stack(enc_out_a, dim=-1)
        enc_out_sub_a = torch.stack(enc_out_sub_a, dim=-1)

        best_hyps, aw = [], []
        best_hyps_sub, aw_sub = [], []
        for b in range(batch_size):
            # Initialization for the word model per utterance
            dec_state, dec_out = self._init_dec_state(
                enc_out[b:b + 1], x_lens[b:b + 1], task=0, dir='fwd')
            aw_step = self._create_var(
                (1, x_lens[b].data[0], self.num_heads_0), fill_value=0., volatile=True)
            context_vec = self._create_var(
                (1,  1, enc_out.size(-1)), fill_value=0., volatile=True)

            # Initialization for the character model per utterance
            dec_state_sub, dec_out_sub = self._init_dec_state(
                enc_out_sub[b: b + 1], x_lens_sub[b:b + 1], task=1, dir='fwd')
            aw_step_sub = self._create_var(
                (1, x_lens_sub[b].data[0], self.num_heads_1), fill_value=0., volatile=True)
            context_vec_sub = self._create_var(
                (1,  1, enc_out_sub.size(-1)), fill_value=0., volatile=True)

            complete = []
            beam = [{'hyp': [self.sos_0],
                     'hyp_sub': [self.sos_1],
                     'score': 0,  # log1
                     'score_sub': 0,  # log 1
                     'dec_state': dec_state,
                     'dec_state_sub': dec_state_sub,
                     'dec_out': dec_out,
                     'dec_out_sub': dec_out_sub,
                     'context_vec': context_vec,
                     'context_vec_sub': context_vec_sub,
                     'aw_steps': [aw_step],
                     'aw_steps_sub':[aw_step_sub]}]
            for t in range(max_decode_len):
                new_beam = []
                for i_beam in range(len(beam)):
                    y = self._create_var(
                        (1, 1), fill_value=beam[i_beam]['hyp'][-1], dtype='long')
                    y = self.embed_0(y)

                    if self.decoding_order == 'bahdanau':
                        if t == 0:
                            dec_out = beam[i_beam]['dec_out']
                        else:
                            # Recurrency
                            dec_in = torch.cat([y, context_vec], dim=-1)
                            dec_out, dec_state = self.decoder_0_fwd(
                                dec_in, beam[i_beam]['dec_state'])

                        # Score
                        context_vec, aw_step = self.attend_0_fwd(
                            enc_out[b:b + 1, :x_lens.data[b]],
                            enc_out_a[b:b + 1, :x_lens.data[b]],
                            x_lens[b:b + 1],
                            dec_out, beam[i_beam]['aw_steps'][-1])

                    elif self.decoding_order == 'luong':
                        # Recurrency
                        dec_in = torch.cat(
                            [y, beam[i_beam]['context_vec']], dim=-1)
                        dec_out, dec_state = self.decoder_0_fwd(
                            dec_in, beam[i_beam]['dec_state'])

                        # Score
                        context_vec, aw_step = self.attend_0_fwd(
                            enc_out[b:b + 1, :x_lens.data[b]],
                            enc_out_a[b:b + 1, :x_lens.data[b]],
                            x_lens[b:b + 1],
                            dec_out, beam[i_beam]['aw_steps'][-1])

                    elif self.decoding_order == 'conditional':
                        # Recurrency of the first decoder
                        _dec_out, _dec_state = self.decoder_first_0_fwd(
                            y, beam[i_beam]['dec_state'])

                        # Score
                        context_vec, aw_step = self.attend_0_fwd(
                            enc_out[b:b + 1, :x_lens.data[b]],
                            enc_out_a[b:b + 1, :x_lens.data[b]],
                            x_lens[b:b + 1],
                            _dec_out, beam[i_beam]['aw_steps'][-1])

                        # Recurrency of the second decoder
                        dec_out, dec_state = self.decoder_second_0_fwd(
                            context_vec, _dec_state)

                    else:
                        raise ValueError(self.decoding_order)

                    # Generate
                    out = self.W_d_0_fwd(dec_out) + \
                        self.W_c_0_fwd(context_vec)
                    logits_step = self.fc_0_fwd(F.tanh(out))

                    # Path through the log-softmax layer
                    log_probs = F.log_softmax(logits_step.squeeze(1), dim=1)

                    # Pick up the top-k scores
                    log_probs_topk, indices_topk = log_probs.topk(
                        beam_width, dim=1, largest=True, sorted=True)

                    for k in range(beam_width):
                        # Exclude short hypotheses
                        if indices_topk[0, k].data[0] == self.eos_0 and len(beam[i_beam]['hyp']) < min_decode_len:
                            continue
                        # if indices_topk[0, k].data[0] == self.eos_0 and len(beam[i_beam]['hyp']) < x_lens[b].data[0] * min_decode_len_ratio:
                        #     continue

                        # Add length penalty
                        score = beam[i_beam]['score'] + \
                            log_probs_topk.data[0, k] + length_penalty

                        # Add coverage penalty
                        if coverage_penalty > 0:
                            threshold = 0.5
                            aw_steps = torch.cat(
                                beam[i_beam]['aw_steps'], dim=0).sum(0).squeeze(1)

                            # Google NMT
                            # cov_sum = torch.where(
                            #     aw_steps < threshold, aw_steps, torch.ones_like(aw_steps) * threshold).sum(0)
                            # score += torch.log(cov_sum) * coverage_penalty

                            # Toward better decoding
                            cov_sum = torch.where(
                                aw_steps > threshold, aw_steps, torch.zeros_like(aw_steps)).sum(0)
                            score += cov_sum * coverage_penalty

                        #######################################################
                        # NOTE: Resocre by the second decoder's score
                        #######################################################
                        oov_flag = indices_topk.data[0, k] == oov_index
                        eos_flag = indices_topk.data[0, k] == self.eos_0
                        score_c2w = beam[i_beam]['score_sub']
                        score_c2w_until_space = beam[i_beam]['score_sub']
                        word_idx = indices_topk.data[0, k]

                        if self.decoding_order == 'bahdanau':
                            if oov_flag:
                                # NOTE: Decode until outputting a space
                                t_sub = 0
                                dec_outs_sub = [beam[i_beam]['dec_out_sub']]
                                dec_states_sub = [
                                    beam[i_beam]['dec_state_sub']]
                                aw_steps_sub = [
                                    beam[i_beam]['aw_steps_sub'][-1]]
                                charseq = []
                                # TODO: add max OOV len
                                while True:
                                    # Score
                                    context_vec_sub, aw_step_sub = self.attend_1_fwd(
                                        enc_out_sub[b:b + 1,
                                                    :x_lens_sub.data[b]],
                                        enc_out_sub_a[b:b + 1,
                                                      :x_lens_sub.data[b]],
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
                                        charseq.append(space_index)
                                        y_sub = self._create_var(
                                            (1, 1), fill_value=space_index, dtype='long')
                                    else:
                                        y_sub = torch.max(log_probs_sub, dim=1)[
                                            1].data[0]

                                        if y_sub == space_index:
                                            break
                                        if y_sub == self.eos_1:
                                            break
                                        if t_sub > 20:
                                            break

                                        score_c2w += log_probs_sub.data[0, y_sub]
                                        charseq.append(y_sub)
                                        y_sub = self._create_var(
                                            (1, 1), fill_value=y_sub, dtype='long')
                                    y_sub = self.embed_1(y_sub)

                                    # print(idx2word([word_idx]))
                                    # print(idx2char(charseq))

                                    # Recurrency
                                    dec_in_sub = torch.cat(
                                        [y_sub, context_vec_sub], dim=-1)
                                    dec_out_sub, dec_state_sub = self.decoder_1_fwd(
                                        dec_in_sub, dec_states_sub[-1])

                                    dec_outs_sub += [dec_out_sub]
                                    dec_states_sub += [dec_state_sub]
                                    aw_steps_sub += [aw_step_sub]
                                    t_sub += 1

                                dec_out_sub = dec_outs_sub[-1]
                                dec_state_sub = copy.deepcopy(
                                    dec_states_sub[-1])

                            elif eos_flag:
                                # Score
                                context_vec_sub, aw_step_sub = self.attend_1_fwd(
                                    enc_out_sub[b:b + 1,
                                                :x_lens_sub.data[b]],
                                    enc_out_sub_a[b:b + 1,
                                                  :x_lens_sub.data[b]],
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

                                charseq = [self.eos_1]
                                score_c2w += log_probs_sub.data[0, self.eos_1]
                                aw_steps_sub.append(aw_step_sub)

                                # print(idx2word([word_idx]))
                                # print(idx2char([self.eos_1]))

                            else:
                                # Decompose a word to characters
                                charseq = word2char(word_idx)
                                # charseq: `[num_chars,]` (list)

                                if t > 0:
                                    charseq = [space_index] + charseq

                                # print(idx2word([word_idx]))
                                # print(idx2char(charseq))

                                dec_out_sub = beam[i_beam]['dec_out_sub']
                                dec_state_sub = beam[i_beam]['dec_state_sub']
                                aw_step_sub = beam[i_beam]['aw_steps_sub'][-1]
                                aw_steps_sub = []
                                for t_sub in range(len(charseq)):
                                    # Score
                                    context_vec_sub, aw_step_sub = self.attend_1_fwd(
                                        enc_out_sub[b:b + 1,
                                                    :x_lens_sub.data[b]],
                                        enc_out_sub_a[b:b + 1,
                                                      :x_lens_sub.data[b]],
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
                                    aw_steps_sub.append(aw_step_sub)

                                    # teacher-forcing
                                    y_sub = self._create_var(
                                        (1, 1), fill_value=charseq[t_sub], dtype='long')
                                    y_sub = self.embed_1(y_sub)

                                    # Recurrency
                                    dec_in_sub = torch.cat(
                                        [y_sub, context_vec_sub], dim=-1)
                                    dec_out_sub, dec_state_sub = self.decoder_1_fwd(
                                        dec_in_sub, dec_state_sub)

                        elif self.decoding_order in ['luong', 'conditional']:
                            if oov_flag:
                                # decoder until outputting a space
                                t_sub = 0
                                dec_outs_sub_tmp = [
                                    beam[i_beam]['dec_out_sub']]
                                dec_states_sub_tmp = [
                                    beam[i_beam]['dec_state_sub']]
                                aw_steps_sub = [
                                    beam[i_beam]['aw_steps_sub'][-1]]
                                context_vecs_sub_tmp = [
                                    beam[i_beam]['coontext_vec_sub']]
                                charseq = []
                                # TODO: add max OOV len
                                while True:
                                    if t_sub == 0:
                                        if t == 0:
                                            # <SOS>
                                            y_sub = self._create_var(
                                                (1, 1), fill_value=self.sos_1, dtype='long')
                                        else:
                                            # the last character of the previous word
                                            last_char = beam[i_beam]['hyp_sub'][-1]
                                            y_sub = self._create_var(
                                                (1, 1), fill_value=last_char, dtype='long')
                                    elif t_sub == 1 and t > 0:
                                        # space before the word
                                        y_sub = self._create_var(
                                            (1, 1), fill_value=space_index, dtype='long')
                                    else:
                                        y_sub = torch.max(log_probs_sub.squeeze(1), dim=1)[
                                            1].unsqueeze(1)
                                    y_sub = self.embed_1(y_sub)

                                    if self.decoding_order == 'luong':
                                        # Recurrency
                                        dec_in_sub = torch.cat(
                                            [y_sub, context_vecs_sub_tmp[-1]], dim=-1)
                                        dec_out_sub, dec_state_sub = self.decoder_1_fwd(
                                            dec_in_sub,  dec_states_sub_tmp[-1])

                                        # Score
                                        context_vec_sub, aw_step_sub = self.attend_1_fwd(
                                            enc_out_sub[b:b + 1,
                                                        :x_lens_sub.data[b]],
                                            enc_out_sub_a[b:b + 1,
                                                          :x_lens_sub.data[b]],
                                            x_lens_sub[b:b + 1],
                                            dec_out_sub, aw_steps_sub[-1])

                                    elif self.decoding_order == 'conditional':
                                        # Recurrency of the first decoder
                                        _dec_out_sub, _dec_state_sub = self.decoder_first_1_fwd(
                                            y_sub, dec_states_sub_tmp[-1])

                                        # Score
                                        context_vec_sub, aw_step_sub = self.attend_1_fwd(
                                            enc_out_sub[b:b + 1,
                                                        :x_lens_sub.data[b]],
                                            enc_out_sub_a[b:b + 1,
                                                          :x_lens_sub.data[b]],
                                            x_lens_sub[b:b + 1],
                                            _dec_out_sub, aw_steps_sub[-1])

                                        # Recurrency of the second decoder
                                        dec_out_sub, dec_state_sub = self.decoder_second_1_fwd(
                                            context_vec_sub, _dec_state_sub)

                                    # Generate
                                    logits_step_sub = self.fc_1_fwd(F.tanh(
                                        self.W_d_1_fwd(dec_out_sub) +
                                        self.W_c_1_fwd(context_vec_sub)))

                                    # Path through the log-softmax layer
                                    log_probs_sub = F.log_softmax(
                                        logits_step_sub.squeeze(1), dim=-1)

                                    if t_sub == 0 and t > 0:
                                        # space before the word
                                        charseq.append(space_index)
                                        score_c2w += log_probs_sub.data[0,
                                                                        space_index]
                                        score_c2w_until_space += log_probs_sub.data[0,
                                                                                    space_index]
                                        y_sub = self._create_var(
                                            (1, 1), fill_value=space_index, dtype='long')
                                    else:
                                        y_sub = torch.max(log_probs_sub, dim=1)[
                                            1].data[0]

                                        if y_sub == space_index:
                                            break
                                        if y_sub == self.eos_1:
                                            break
                                        if t_sub > 20:
                                            break

                                    # print(idx2word([word_idx]))
                                    # print(idx2char(charseq))

                                    dec_outs_sub_tmp += [dec_out_sub]
                                    dec_states_sub_tmp += [dec_state_sub]
                                    aw_steps_sub += [aw_step_sub]
                                    context_vecs_sub_tmp += [context_vec_sub]
                                    t_sub += 1

                                dec_out_sub = dec_outs_sub_tmp[-1]
                                dec_state_sub = copy.deepcopy(
                                    dec_states_sub_tmp[-1])
                                context_vec_sub = context_vecs_sub_tmp[-1]

                            elif eos_flag:
                                # teacher-forcing
                                last_char = beam[i_beam]['hyp_sub'][-1]
                                y_sub = self._create_var(
                                    (1, 1), fill_value=last_char, dtype='long')
                                y_sub = self.embed_1(y_sub)

                                if self.decoding_order == 'luong':
                                    # Recurrency
                                    dec_in_sub = torch.cat(
                                        [y_sub, context_vec_sub], dim=-1)
                                    dec_out_sub, dec_state_sub = self.decoder_1_fwd(
                                        dec_in_sub, beam[i_beam]['dec_state_sub'])

                                    # Score
                                    context_vec_sub, aw_step_sub = self.attend_1_fwd(
                                        enc_out_sub[b:b + 1,
                                                    :x_lens_sub.data[b]],
                                        enc_out_sub_a[b:b + 1,
                                                      :x_lens_sub.data[b]],
                                        x_lens_sub[b:b + 1],
                                        dec_out_sub, beam[i_beam]['aw_steps_sub'][-1])

                                elif self.decoding_order == 'conditional':
                                    # Recurrency of the first decoder
                                    _dec_out_sub, _dec_state_sub = self.decoder_first_1_fwd(
                                        y_sub, beam[i_beam]['dec_state_sub'])

                                    # Score
                                    context_vec_sub, aw_step_sub = self.attend_1_fwd(
                                        enc_out_sub[b:b + 1,
                                                    :x_lens_sub.data[b]],
                                        enc_out_sub_a[b:b + 1,
                                                      :x_lens_sub.data[b]],
                                        x_lens_sub[b:b + 1],
                                        _dec_out_sub, beam[i_beam]['aw_steps_sub'][-1])

                                    # Recurrency of the second decoder
                                    dec_out_sub, dec_state_sub = self.decoder_second_1_fwd(
                                        context_vec_sub, _dec_state_sub)

                                # Generate
                                logits_step_sub = self.fc_1_fwd(F.tanh(
                                    self.W_d_1_fwd(dec_out_sub) +
                                    self.W_c_1_fwd(context_vec_sub)))

                                # Path through the log-softmax layer
                                log_probs_sub = F.log_softmax(
                                    logits_step_sub.squeeze(1), dim=-1)

                                charseq = [self.eos_1]
                                score_c2w += log_probs_sub.data[0, self.eos_1]
                                aw_steps_sub.append(aw_step_sub)

                                # print(idx2word([word_idx]))
                                # print(idx2char([self.eos_1]))

                            else:
                                # Decompose a word to characters
                                charseq = word2char(word_idx)
                                # charseq: `[num_chars,]` (list)

                                if t == 0:
                                    charseq = [self.sos_1] + charseq
                                elif t > 0:
                                    last_char = beam[i_beam]['hyp_sub'][-1]
                                    charseq = [last_char,
                                               space_index] + charseq

                                # print(idx2word([word_idx]))
                                # print(idx2char(charseq))

                                dec_out_sub = beam[i_beam]['dec_out_sub']
                                dec_state_sub = beam[i_beam]['dec_state_sub']
                                context_vec_sub = beam[i_beam]['context_vec_sub']
                                aw_step_sub = beam[i_beam]['aw_steps_sub'][-1]
                                aw_steps_sub = []
                                for t_sub in range(len(charseq) - 1):
                                    # teacher-forcing
                                    y_sub = self._create_var(
                                        (1, 1), fill_value=charseq[t_sub], dtype='long')
                                    y_sub = self.embed_1(y_sub)

                                    if self.decoding_order == 'luong':
                                        # Recurrency
                                        dec_in_sub = torch.cat(
                                            [y_sub, context_vec_sub], dim=-1)
                                        dec_out_sub, dec_state_sub = self.decoder_1_fwd(
                                            dec_in_sub, dec_state_sub)

                                        # Score
                                        context_vec_sub, aw_step_sub = self.attend_1_fwd(
                                            enc_out_sub[b:b + 1,
                                                        :x_lens_sub.data[b]],
                                            enc_out_sub_a[b:b + 1,
                                                          :x_lens_sub.data[b]],
                                            x_lens_sub[b:b + 1],
                                            dec_out_sub, aw_step_sub)

                                    elif self.decoding_order == 'conditional':
                                        # Recurrency of the first decoder
                                        _dec_out_sub, _dec_state_sub = self.decoder_first_1_fwd(
                                            y_sub, dec_state_sub)

                                        # Score
                                        context_vec_sub, aw_step_sub = self.attend_1_fwd(
                                            enc_out_sub[b:b + 1,
                                                        :x_lens_sub.data[b]],
                                            enc_out_sub_a[b:b + 1,
                                                          :x_lens_sub.data[b]],
                                            x_lens_sub[b:b + 1],
                                            _dec_out_sub, aw_step_sub)

                                        # Recurrency of the second decoder
                                        dec_out_sub, dec_state_sub = self.decoder_second_1_fwd(
                                            context_vec_sub, _dec_state_sub)

                                    # Generate
                                    logits_step_sub = self.fc_1_fwd(F.tanh(
                                        self.W_d_1_fwd(dec_out_sub) +
                                        self.W_c_1_fwd(context_vec_sub)))

                                    # Path through the log-softmax layer
                                    log_probs_sub = F.log_softmax(
                                        logits_step_sub.squeeze(1), dim=-1)

                                    score_c2w += log_probs_sub[0,
                                                               charseq[t_sub]].data[0]
                                    if t_sub == 0:
                                        score_c2w_until_space += log_probs_sub.data[0,
                                                                                    charseq[t_sub]]
                                        # NOTE: if t == 0, <SOS>
                                    aw_steps_sub.append(aw_step_sub)

                        # Rescoreing
                        score += (score_c2w - score_c2w_until_space) * \
                            score_sub_weight

                        new_beam.append(
                            {'hyp': beam[i_beam]['hyp'] + [indices_topk.data[0, k]],
                             'hyp_sub': beam[i_beam]['hyp_sub'] + charseq,
                             'score': score,
                             'score_sub': score_c2w,
                             'dec_state': copy.deepcopy(dec_state),
                             'dec_state_sub': copy.deepcopy(dec_state_sub),
                             'dec_out': dec_out,
                             'dec_out_sub': dec_out_sub,
                             'context_vec': context_vec,
                             'context_vec_sub': context_vec_sub,
                             'aw_steps': beam[i_beam]['aw_steps'] + [aw_step],
                             'aw_steps_sub': beam[i_beam]['aw_steps_sub'] + aw_steps_sub})

                new_beam = sorted(
                    new_beam, key=lambda x: x['score'], reverse=True)

                # Remove complete hypotheses
                not_complete = []
                for cand in new_beam[:beam_width]:
                    if cand['hyp'][-1] == self.eos_0:
                        complete.append(cand)
                    else:
                        not_complete.append(cand)
                if len(complete) >= beam_width:
                    complete = complete[:beam_width]
                    break
                beam = not_complete[:beam_width]

            if len(complete) == 0:
                complete = beam

            complete = sorted(
                complete, key=lambda x: x['score'], reverse=True)
            best_hyps.append(np.array(complete[0]['hyp'][1:]))
            aw.append(complete[0]['aw_steps'][1:])

            best_hyps_sub.append(
                np.array(complete[0]['hyp_sub'][1:]))
            aw_sub.append(complete[0]['aw_steps_sub'][1:])

        # Concatenate in T_out dimension
        for t_sub in range(len(aw)):
            for k in range(len(aw_sub[t_sub])):
                # TODO: fix for MHA
                aw_sub[t_sub][k] = aw_sub[t_sub][k][:, :, 0]
            aw_sub[t_sub] = self.var2np(
                torch.stack(aw_sub[t_sub], dim=1).squeeze(0))

            for k in range(len(aw[t_sub])):
                aw[t_sub][k] = aw[t_sub][k][:, :, 0]  # TODO: fix for MHA
            aw[t_sub] = self.var2np(
                torch.stack(aw[t_sub], dim=1).squeeze(0))

        return best_hyps, aw, best_hyps_sub, aw_sub

    def _decode_infer_joint_rescoring(self, enc_out, x_lens, enc_out_sub, x_lens_sub,
                                      beam_width, max_decode_len, min_decode_len,
                                      length_penalty, coverage_penalty,
                                      space_index, oov_index, word2char, score_sub_weight,
                                      idx2word, idx2char):
        """Joint decoding (rescoring).
        Args:
            enc_out (torch.FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            x_lens (torch.IntTensor): A tensor of size `[B]`
            enc_out_sub (torch.FloatTensor): A tensor of size
                `[B, T_in_sub, encoder_num_units]`
            x_lens_sub (torch.IntTensor): A tensor of size `[B]`
            beam_width (int): the size of beam in the main task
            max_decode_len (int): the maximum sequence length of tokens
            min_decode_len (int): the minimum sequence length of tokens
            length_penalty (float): length penalty in beam search decoding
            coverage_penalty (float): coverage penalty in beam search decoding
            space_index (int):
            oov_index (int):
            word2char ():
            score_sub_weight (float):
            idx2word (): for debug
            idx2char (): for debug
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B, T_out]`
            aw (np.ndarray): A tensor of size `[B, T_out, T_in]`
            best_hyps_sub (np.ndarray): A tensor of size `[B, T_out_sub]`
            aw_sub (np.ndarray): A tensor of size `[B, T_out_sub, T_in]`
            aw_dec (np.ndarray): A tensor of size `[B, T_out, T_out_sub]`
        """
        batch_size, max_time = enc_out.size()[:2]

        # Pre-computation of encoder-side features computing scores
        enc_out_a, enc_out_sub_a = [], []
        for h in range(self.num_heads_0):
            enc_out_a += [getattr(self.attend_0_fwd,
                                  'W_enc_head' + str(h))(enc_out)]
        for h in range(self.num_heads_1):
            enc_out_sub_a += [getattr(self.attend_1_fwd,
                                      'W_enc_head' + str(h))(enc_out_sub)]
        enc_out_a = torch.stack(enc_out_a, dim=-1)
        enc_out_sub_a = torch.stack(enc_out_sub_a, dim=-1)

        best_hyps, aw = [], []
        best_hyps_sub, aw_sub = [], []
        for b in range(batch_size):
            # Initialization for the word model per utterance
            dec_state, dec_out = self._init_dec_state(
                enc_out[b: b + 1], x_lens[b: b + 1], task=0, dir='fwd')
            aw_step = self._create_var(
                (1, x_lens[b].data[0], self.num_heads_0), fill_value=0., volatile=True)
            context_vec = self._create_var(
                (1,  1, enc_out.size(-1)), fill_value=0., volatile=True)

            complete = []
            beam = [{'hyp': [self.sos_0],
                     'score': 0,  # log1
                     'dec_state': dec_state,
                     'dec_out': dec_out,
                     'context_vec': context_vec,
                     'aw_steps': [aw_step]}]

            for t in range(max_decode_len):
                new_beam = []
                for i_beam in range(len(beam)):
                    y = self._create_var(
                        (1, 1), fill_value=beam[i_beam]['hyp'][-1], dtype='long')
                    y = self.embed_0(y)

                    if self.decoding_order == 'bahdanau':
                        if t == 0:
                            dec_out = beam[i_beam]['dec_out']
                        else:
                            # Recurrency
                            dec_in = torch.cat(
                                [y, beam[i_beam]['context_vec']], dim=-1)
                            dec_out, dec_state = self.decoder_0_fwd(
                                dec_in, beam[i_beam]['dec_state'])

                        # Score
                        context_vec, aw_step = self.attend_0_fwd(
                            enc_out[b:b + 1, :x_lens.data[b]],
                            enc_out_a[b:b + 1, :x_lens.data[b]],
                            x_lens[b:b + 1],
                            dec_out, beam[i_beam]['aw_steps'][-1])

                    elif self.decoding_order == 'luong':
                        # Recurrency
                        dec_in = torch.cat(
                            [y, beam[i_beam]]['context_vec'], dim=-1)
                        dec_out, dec_state = self.decoder_0_fwd(
                            dec_in, beam[i_beam]['dec_state'])

                        # Score
                        context_vec, aw_step = self.attend_0_fwd(
                            enc_out[b:b + 1, :x_lens.data[b]],
                            enc_out_a[b:b + 1, :x_lens.data[b]],
                            x_lens[b:b + 1],
                            dec_out, beam[i_beam]['aw_steps'][-1])

                    elif self.decoding_order == 'conditional':
                        # Recurrency of the first decoder
                        _dec_out, _dec_state = self.decoder_first_0_fwd(
                            y, beam[i_beam]['dec_state'])

                        # Score
                        context_vec, aw_step = self.attend_0_fwd(
                            enc_out[b:b + 1, :x_lens.data[b]],
                            enc_out_a[b:b + 1, :x_lens.data[b]],
                            x_lens[b:b + 1],
                            _dec_out, beam[i_beam]['aw_steps'][-1])

                        # Recurrency of the second decoder
                        dec_out, dec_state = self.decoder_second_0_fwd(
                            context_vec, _dec_state)

                    else:
                        raise ValueError(self.decoding_order)

                    # Generate
                    out = self.W_d_0_fwd(dec_out) + \
                        self.W_c_0_fwd(context_vec)
                    logits_step = self.fc_0_fwd(F.tanh(out))

                    # Path through the log-softmax layer
                    log_probs = F.log_softmax(logits_step.squeeze(1), dim=1)

                    # Pick up the top-k scores
                    log_probs_topk, indices_topk = log_probs.topk(
                        beam_width, dim=1, largest=True, sorted=True)

                    for k in range(beam_width):
                        # Exclude short hypotheses
                        if indices_topk[0, k].data[0] == self.eos_0 and len(beam[i_beam]['hyp']) < min_decode_len:
                            continue
                        # if indices_topk[0, k].data[0] == self.eos_0 and len(beam[i_beam]['hyp']) < x_lens[b].data[0] * min_decode_len_ratio:
                        #     continue

                        # Add length penalty
                        score = beam[i_beam]['score'] + \
                            log_probs_topk.data[0, k] + length_penalty

                        # Add coverage penalty
                        if coverage_penalty > 0:
                            threshold = 0.5
                            aw_steps = torch.cat(
                                beam[i_beam]['aw_steps'], dim=0).sum(0).squeeze(1)

                            # Google NMT
                            # cov_sum = torch.where(
                            #     aw_steps < threshold, aw_steps, torch.ones_like(aw_steps) * threshold).sum(0)
                            # score += torch.log(cov_sum) * coverage_penalty

                            # Toward better decoding
                            cov_sum = torch.where(
                                aw_steps > threshold, aw_steps, torch.zeros_like(aw_steps)).sum(0)
                            score += cov_sum * coverage_penalty

                        new_beam.append(
                            {'hyp': beam[i_beam]['hyp'] + [indices_topk.data[0, k]],
                             'score': score,
                             'dec_state': copy.deepcopy(dec_state),
                             'dec_out': dec_out,
                             'context_vec': context_vec,
                             'aw_steps': beam[i_beam]['aw_steps'] + [aw_step]})

                new_beam = sorted(
                    new_beam, key=lambda x: x['score'], reverse=True)

                # Remove complete hypotheses
                not_complete = []
                for cand in new_beam[:beam_width]:
                    if cand['hyp'][-1] == self.eos_0:
                        complete.append(cand)
                    else:
                        not_complete.append(cand)
                if len(complete) >= beam_width:
                    complete = complete[:beam_width]
                    break
                beam = not_complete[:beam_width]

            if len(complete) == 0:
                complete = beam

            #######################################################
            # NOTE: Resocre by the second decoder's score
            #######################################################
            for i_beam in range(len(complete)):
                # Initialization for the character model per utterance
                dec_state_sub, dec_out_sub = self._init_dec_state(
                    enc_out_sub[b: b + 1], x_lens_sub[b:b + 1], task=1, dir='fwd')
                aw_step_sub = self._create_var(
                    (1, x_lens_sub[b].data[0], self.num_heads_1), fill_value=0., volatile=True)
                aw_steps_sub = [aw_step_sub]
                context_vec_sub = self._create_var(
                    (1, 1, dec_out_sub.size(1)), fill_value=0., volatile=True)

                score_c2w = 0  # log 1
                score_c2w_until_space = 0  # log 1
                charseq = [self.sos_1]

                for t in range(len(complete[i_beam]['hyp']) - 1):
                    oov_flag = complete[i_beam]['hyp'][t + 1] == oov_index
                    eos_flag = complete[i_beam]['hyp'][t + 1] == self.eos_0
                    word_idx = complete[i_beam]['hyp'][t + 1]

                    if self.decoding_order == 'bahdanau':
                        if oov_flag:
                            # decoder until outputting a space
                            t_sub = 0
                            dec_outs_sub = [dec_out_sub]
                            dec_states_sub = [dec_state_sub]
                            aw_steps_sub_tmp = [aw_steps_sub[-1]]
                            charseq_tmp = []
                            # TODO: add max OOV len
                            while True:
                                # Score
                                context_vec_sub, aw_step_sub = self.attend_1_fwd(
                                    enc_out_sub[b:b + 1,
                                                :x_lens_sub.data[b]],
                                    enc_out_sub_a[b:b + 1,
                                                  :x_lens_sub.data[b]],
                                    x_lens_sub[b:b + 1],
                                    dec_outs_sub[-1], aw_steps_sub_tmp[-1])

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
                                    charseq_tmp.append(space_index)
                                    y_sub = self._create_var(
                                        (1, 1), fill_value=space_index, dtype='long')
                                else:
                                    y_sub = torch.max(log_probs_sub, dim=1)[
                                        1].data[0]

                                    if y_sub == space_index:
                                        break
                                    if y_sub == self.eos_1:
                                        break
                                    if t_sub > 20:
                                        break

                                    score_c2w += log_probs_sub.data[0, y_sub]
                                    charseq_tmp.append(y_sub)
                                    y_sub = self._create_var(
                                        (1, 1), fill_value=y_sub, dtype='long')

                                # print(idx2word([word_idx]))
                                # print(idx2char(charseq_tmp))

                                y_sub = self.embed_1(y_sub)

                                # Recurrency
                                dec_in_sub = torch.cat(
                                    [y_sub, context_vec_sub], dim=-1)
                                dec_out_sub, dec_state_sub = self.decoder_1_fwd(
                                    dec_in_sub, dec_states_sub[-1])

                                dec_outs_sub += [dec_out_sub]
                                dec_states_sub += [dec_state_sub]
                                aw_steps_sub_tmp += [aw_step_sub]
                                t_sub += 1

                            dec_out_sub = dec_outs_sub[-1]
                            dec_state_sub = copy.deepcopy(
                                dec_states_sub[-1])
                            aw_steps_sub += aw_steps_sub_tmp[:-1]
                            charseq += charseq_tmp

                        elif eos_flag:
                            # Score
                            context_vec_sub, aw_step_sub = self.attend_1_fwd(
                                enc_out_sub[b:b + 1,
                                            :x_lens_sub.data[b]],
                                enc_out_sub_a[b:b + 1,
                                              :x_lens_sub.data[b]],
                                x_lens_sub[b:b + 1],
                                dec_out_sub, aw_steps_sub[-1])

                            # Generate
                            logits_step_sub = self.fc_1_fwd(F.tanh(
                                self.W_d_1_fwd(dec_out_sub) +
                                self.W_c_1_fwd(context_vec_sub)))

                            # Path through the log-softmax layer
                            log_probs_sub = F.log_softmax(
                                logits_step_sub.squeeze(1), dim=-1)

                            charseq.append(self.eos_1)
                            score_c2w += log_probs_sub.data[0, self.eos_1]
                            aw_steps_sub.append(aw_step_sub)

                            # print(idx2word([word_idx]))
                            # print(idx2char([self.eos_1]))

                        else:
                            # Decompose a word to characters
                            charseq_tmp = word2char(word_idx)
                            # charseq_tmp: `[num_chars,]` (list)

                            if t > 0:
                                charseq_tmp = [space_index] + charseq_tmp

                            # print(idx2word([word_idx]))
                            # print(idx2char(charseq_tmp))

                            for t_sub in range(len(charseq_tmp)):
                                # Score
                                context_vec_sub, aw_step_sub = self.attend_1_fwd(
                                    enc_out_sub[b:b + 1,
                                                :x_lens_sub.data[b]],
                                    enc_out_sub_a[b:b + 1,
                                                  :x_lens_sub.data[b]],
                                    x_lens_sub[b:b + 1],
                                    dec_out_sub, aw_steps_sub[-1])

                                # Generate
                                logits_step_sub = self.fc_1_fwd(F.tanh(
                                    self.W_d_1_fwd(dec_out_sub) +
                                    self.W_c_1_fwd(context_vec_sub)))

                                # Path through the log-softmax layer
                                log_probs_sub = F.log_softmax(
                                    logits_step_sub.squeeze(1), dim=-1)

                                score_c2w += log_probs_sub.data[0,
                                                                charseq_tmp[t_sub]]
                                if t_sub == 0 and t > 0:
                                    score_c2w_until_space += log_probs_sub.data[0,
                                                                                charseq_tmp[t_sub]]
                                aw_steps_sub.append(aw_step_sub)

                                # teacher-forcing
                                y_sub = self._create_var(
                                    (1, 1), fill_value=charseq_tmp[t_sub], dtype='long')
                                y_sub = self.embed_1(y_sub)

                                # Recurrency
                                dec_in_sub = torch.cat(
                                    [y_sub, context_vec_sub], dim=-1)
                                dec_out_sub, dec_state_sub = self.decoder_1_fwd(
                                    dec_in_sub, dec_state_sub)

                            charseq += charseq_tmp

                    elif self.decoding_order in ['luong', 'conditional']:
                        if oov_flag:
                            # decoder until outputting a space
                            t_sub = 0
                            dec_outs_sub_tmp = [dec_out_sub]
                            dec_states_sub_tmp = [dec_state_sub]
                            context_vecs_sub_tmp = [context_vec_sub]
                            charseq_tmp = []
                            # TODO: add max OOV len
                            while True:
                                if t_sub == 0:
                                    if t == 0:
                                        # <SOS>
                                        y_sub = self._create_var(
                                            (1, 1), fill_value=self.sos_1, dtype='long')
                                    else:
                                        # the last character of the previous word
                                        last_char = beam[i_beam]['hyp_sub'][-1]
                                        y_sub = self._create_var(
                                            (1, 1), fill_value=last_char, dtype='long')
                                elif t_sub == 1 and t > 0:
                                    # space before the word
                                    y_sub = self._create_var(
                                        (1, 1), fill_value=space_index, dtype='long')
                                else:
                                    y_sub = torch.max(log_probs_sub.squeeze(1), dim=1)[
                                        1].unsqueeze(1)
                                y_sub = self.embed_1(y_sub)

                                if self.decoding_order == 'luong':
                                    # Recurrency
                                    dec_in_sub = torch.cat(
                                        [y_sub, context_vecs_sub_tmp[-1]], dim=-1)
                                    dec_out_sub, dec_state_sub = self.decoder_1_fwd(
                                        dec_in_sub, dec_states_sub_tmp[-1])

                                    # Score
                                    context_vec_sub, aw_step_sub = self.attend_1_fwd(
                                        enc_out_sub[b:b + 1,
                                                    :x_lens_sub.data[b]],
                                        enc_out_sub_a[b:b + 1,
                                                      :x_lens_sub.data[b]],
                                        x_lens_sub[b:b + 1],
                                        dec_out_sub, aw_steps_sub[-1])

                                elif self.decoding_order == 'conditional':
                                    # Recurrency of the first decoder
                                    _dec_out_sub, _dec_state_sub = self.decoder_first_1_fwd(
                                        y_sub, dec_states_sub_tmp[-1])

                                    # Score
                                    context_vec_sub, aw_step_sub = self.attend_1_fwd(
                                        enc_out_sub[b:b + 1,
                                                    :x_lens_sub.data[b]],
                                        enc_out_sub_a[b:b + 1,
                                                      :x_lens_sub.data[b]],
                                        x_lens_sub[b:b + 1],
                                        _dec_out_sub, aw_steps_sub[-1])

                                    # Recurrency of the second decoder
                                    dec_out_sub, dec_state_sub = self.decoder_second_1_fwd(
                                        context_vec_sub, _dec_state_sub)

                                # Generate
                                logits_step_sub = self.fc_1_fwd(F.tanh(
                                    self.W_d_1_fwd(dec_out_sub) +
                                    self.W_c_1_fwd(context_vec_sub)))

                                # Path through the log-softmax layer
                                log_probs_sub = F.log_softmax(
                                    logits_step_sub.squeeze(1), dim=-1)

                                if t_sub == 0 and t > 0:
                                    # space before the word
                                    charseq_tmp.append(space_index)
                                    score_c2w += log_probs_sub.data[0,
                                                                    space_index]
                                    score_c2w_until_space += log_probs_sub.data[0,
                                                                                space_index]
                                    y_sub = self._create_var(
                                        (1, 1), fill_value=space_index, dtype='long')
                                else:
                                    y_sub = torch.max(log_probs_sub, dim=1)[
                                        1].data[0]

                                    if y_sub == space_index:
                                        break
                                    if y_sub == self.eos_1:
                                        break
                                    if t_sub > 20:
                                        break

                                # print(idx2word([word_idx]))
                                # print(idx2char(charseq_tmp))

                                dec_outs_sub_tmp += [dec_out_sub]
                                dec_states_sub_tmp += [dec_state_sub]
                                aw_steps_sub += [aw_step_sub]
                                context_vecs_sub_tmp += [context_vec_sub]
                                t_sub += 1

                            dec_out_sub = dec_outs_sub_tmp[-1]
                            dec_state_sub = copy.deepcopy(
                                dec_states_sub_tmp[-1])
                            context_vec_sub = context_vecs_sub_tmp[-1]
                            charseq += charseq_tmp

                        elif eos_flag:
                            # teacher-forcing
                            last_char = beam[i_beam]['hyp_sub'][-1]
                            y_sub = self._create_var(
                                (1, 1), fill_value=last_char, dtype='long')
                            y_sub = self.embed_1(y_sub)

                            if self.decoding_order == 'luong':
                                # Recurrency
                                dec_in_sub = torch.cat(
                                    [y_sub, context_vec_sub], dim=-1)
                                dec_out_sub, dec_state_sub = self.decoder_1_fwd(
                                    dec_in_sub, dec_state_sub)

                                # Score
                                context_vec_sub, aw_step_sub = self.attend_1_fwd(
                                    enc_out_sub[b:b + 1,
                                                :x_lens_sub.data[b]],
                                    enc_out_sub_a[b:b + 1,
                                                  :x_lens_sub.data[b]],
                                    x_lens_sub[b:b + 1],
                                    dec_out_sub, aw_steps_sub[-1])

                            elif self.decoding_order == 'conditional':
                                # Recurrency of the first decoder
                                _dec_out_sub, _dec_state_sub = self.decoder_first_1_fwd(
                                    y_sub, dec_state_sub)

                                # Score
                                context_vec_sub, aw_step_sub = self.attend_1_fwd(
                                    enc_out_sub[b:b + 1,
                                                :x_lens_sub.data[b]],
                                    enc_out_sub_a[b:b + 1,
                                                  :x_lens_sub.data[b]],
                                    x_lens_sub[b:b + 1],
                                    _dec_out_sub, aw_steps_sub[-1])

                                # Recurrency of the second decoder
                                dec_out_sub, dec_state_sub = self.decoder_second_1_fwd(
                                    context_vec_sub, _dec_state_sub)

                            # Generate
                            logits_step_sub = self.fc_1_fwd(F.tanh(
                                self.W_d_1_fwd(dec_out_sub) +
                                self.W_c_1_fwd(context_vec_sub)))

                            # Path through the log-softmax layer
                            log_probs_sub = F.log_softmax(
                                logits_step_sub.squeeze(1), dim=-1)

                            charseq.append(self.eos_1)
                            score_c2w += log_probs_sub.data[0, self.eos_1]
                            aw_steps_sub.append(aw_step_sub)

                            # print(idx2word([word_idx]))
                            # print(idx2char([self.eos_1]))

                        else:
                            # Decompose a word to characters
                            charseq_tmp = word2char(word_idx)
                            # charseq_tmp: `[num_chars,]` (list)

                            if t == 0:
                                charseq_tmp = [self.sos_1] + charseq_tmp
                            elif t > 0:
                                last_char = beam[i_beam]['hyp_sub'][-1]
                                charseq_tmp = [last_char,
                                               space_index] + charseq_tmp

                            # print(idx2word([word_idx]))
                            # print(idx2char(charseq_tmp))

                            for t_sub in range(len(charseq_tmp) - 1):
                                # teacher-forcing
                                y_sub = self._create_var(
                                    (1, 1), fill_value=charseq_tmp[t_sub], dtype='long')
                                y_sub = self.embed_1(y_sub)

                                if self.decoding_order == 'luong':
                                    # Recurrency
                                    dec_in_sub = torch.cat(
                                        [y_sub, context_vec_sub], dim=-1)
                                    dec_out_sub, dec_state_sub = self.decoder_1_fwd(
                                        dec_in_sub, dec_state_sub)

                                    # Score
                                    context_vec_sub, aw_step_sub = self.attend_1_fwd(
                                        enc_out_sub[b:b + 1,
                                                    :x_lens_sub.data[b]],
                                        enc_out_sub_a[b:b + 1,
                                                      :x_lens_sub.data[b]],
                                        x_lens_sub[b:b + 1],
                                        dec_out_sub, aw_steps_sub[-1])

                                elif self.decoding_order == 'conditional':
                                    # Recurrency of the first decoder
                                    _dec_out_sub, _dec_state_sub = self.decoder_first_1_fwd(
                                        y_sub, dec_state_sub)

                                    # Score
                                    context_vec_sub, aw_step_sub = self.attend_1_fwd(
                                        enc_out_sub[b:b + 1,
                                                    :x_lens_sub.data[b]],
                                        enc_out_sub_a[b:b + 1,
                                                      :x_lens_sub.data[b]],
                                        x_lens_sub[b:b + 1],
                                        _dec_out_sub, aw_steps_sub[-1])

                                    # Recurrency of the second decoder
                                    dec_out_sub, dec_state_sub = self.decoder_second_1_fwd(
                                        context_vec_sub, _dec_state_sub)

                                # Generate
                                logits_step_sub = self.fc_1_fwd(F.tanh(
                                    self.W_d_1_fwd(dec_out_sub) +
                                    self.W_c_1_fwd(context_vec_sub)))

                                # Path through the log-softmax layer
                                log_probs_sub = F.log_softmax(
                                    logits_step_sub.squeeze(1), dim=-1)

                                charseq += charseq_tmp
                                score_c2w += log_probs_sub[0,
                                                           charseq_tmp[t_sub]].data[0]
                                if t_sub == 0:
                                    score_c2w_until_space += log_probs_sub.data[0,
                                                                                charseq_tmp[t_sub]]
                                    # NOTE: if t == 0, <SOS>
                                aw_steps_sub.append(aw_step_sub)

                    # Rescoreing
                    complete[i_beam]['score'] += (score_c2w - score_c2w_until_space) * \
                        score_sub_weight

                    complete[i_beam]['hyp_sub'] = charseq
                    complete[i_beam]['aw_steps_sub'] = aw_steps_sub

            complete = sorted(
                complete, key=lambda x: x['score'], reverse=True)
            best_hyps.append(np.array(complete[0]['hyp'][1:]))
            aw.append(complete[0]['aw_steps'][1:])

            best_hyps_sub.append(
                np.array(complete[0]['hyp_sub'][1:]))
            aw_sub.append(complete[0]['aw_steps_sub'][1:])

        # Concatenate in T_out dimension
        for t_sub in range(len(aw)):
            for k in range(len(aw_sub[t_sub])):
                # TODO: fix for MHA
                aw_sub[t_sub][k] = aw_sub[t_sub][k][:, :, 0]
            aw_sub[t_sub] = self.var2np(
                torch.stack(aw_sub[t_sub], dim=1).squeeze(0))

            for k in range(len(aw[t_sub])):
                aw[t_sub][k] = aw[t_sub][k][:, :, 0]  # TODO: fix for MHA
            aw[t_sub] = self.var2np(
                torch.stack(aw[t_sub], dim=1).squeeze(0))

        return best_hyps, aw, best_hyps_sub, aw_sub
