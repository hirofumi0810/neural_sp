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

from models.pytorch.attention.attention_seq2seq import AttentionSeq2seq
from models.pytorch.linear import LinearND, Embedding, Embedding_LS
from models.pytorch.encoders.load_encoder import load
from models.pytorch.attention.rnn_decoder import RNNDecoder
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
                 decoding_order='attend_generate_update',
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
            loss (torch.FloatTensor or float): A tensor of size `[]`
            loss_main (torch.FloatTensor or float): A tensor of size `[]`
            loss_sub (torch.FloatTensor or float): A tensor of size `[]`
        """
        if is_eval:
            self.eval()
            with torch.no_grad():
                loss, loss_main, loss_sub = self._forward(
                    xs, ys, x_lens, y_lens, ys_sub, y_lens_sub)

                loss = loss.item()
                loss_main = loss_main.item()
                loss_sub = loss_sub.item()
        else:
            self.train()

            # Gaussian noise injection
            if self.weight_noise_injection:
                self.inject_weight_noise(mean=0, std=self.weight_noise_std)

            loss, loss_main, loss_sub = self._forward(
                xs, ys, x_lens, y_lens, ys_sub, y_lens_sub)

            # Update the probability of scheduled sampling
            self._step += 1
            if self.ss_prob > 0:
                self._ss_prob = min(
                    self.ss_prob, self.ss_prob / self.ss_max_step * self._step)

        return loss, loss_main, loss_sub

    def _forward(self, xs, ys, x_lens, y_lens, ys_sub, y_lens_sub):
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
        ys_in = self._create_tensor((ys.shape[0], ys.shape[1] + 1),
                                    fill_value=self.eos_0, dtype=torch.long)
        ys_in_sub = self._create_tensor((ys_sub.shape[0], ys_sub.shape[1] + 1),
                                        fill_value=self.eos_1, dtype=torch.long)
        ys_out = self._create_tensor((ys.shape[0], ys.shape[1] + 1),
                                     fill_value=-1, dtype=torch.long)
        ys_out_sub = self._create_tensor((ys_sub.shape[0], ys_sub.shape[1] + 1),
                                         fill_value=-1, dtype=torch.long)

        ys_in[:, 0] = self.sos_0
        ys_in_sub[:, 0] = self.sos_1
        for b in range(len(xs)):
            ys_in[b, 1:y_lens[b] + 1] = torch.from_numpy(ys[b, :y_lens[b]])
            ys_in_sub[b, 1:y_lens_sub[b] + 1] = torch.from_numpy(
                ys_sub_tmp[b, :y_lens_sub[b]])

            ys_out[b, :y_lens[b]] = torch.from_numpy(ys[b, :y_lens[b]])
            ys_out[b, y_lens[b]] = self.eos_0
            ys_out_sub[b, :y_lens_sub[b]] = torch.from_numpy(
                ys_sub_tmp[b, :y_lens_sub[b]])
            ys_out_sub[b, y_lens_sub[b]] = self.eos_1

        # Wrap by Tensor
        xs = self.np2tensor(xs, dtype=torch.float)
        x_lens = self.np2tensor(x_lens, dtype=torch.int)
        y_lens = self.np2tensor(y_lens, dtype=torch.int)
        y_lens_sub = self.np2tensor(y_lens_sub, dtype=torch.int)

        # Encode acoustic features
        xs, x_lens, xs_sub, x_lens_sub, perm_idx = self._encode(
            xs, x_lens, is_multi_task=True)

        # Permutate indices
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
            loss_main = self._create_tensor(
                (), fill_value=0, dtype=torch.float)
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
            # Wrap by Tensor
            ys_ctc_sub = self.np2tensor(ys_sub, dtype=torch.long)

            ys_ctc_sub = ys_ctc_sub.to(self.device)

            # Permutate indices
            ys_ctc_sub = ys_ctc_sub[perm_idx]

            ctc_loss_sub = self.compute_ctc_loss(
                xs_sub, ys_ctc_sub + 1,
                x_lens_sub, y_lens_sub, task=1) * self.ctc_loss_weight_sub
            loss += ctc_loss_sub

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
        with torch.no_grad():
            if task_index > 0 and self.ctc_loss_weight_sub > self.sub_loss_weight:
                # Decode by CTC decoder
                best_hyps, perm_idx = self.decode_ctc(
                    xs, x_lens, beam_width, task_index)

                return best_hyps, None, perm_idx
                # NOTE: None corresponds to aw in attention-based models
            else:
                # Wrap by Tensor
                xs = self.np2tensor(xs, dtype=torch.float)
                x_lens = self.np2tensor(x_lens, dtype=torch.int)

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
                    raise NotImplementedError

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
                        raise ValueError

                    # Permutate indices to the original order
                    perm_idx = self.tensor2np(perm_idx)

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
            perm_idx = self.tensor2np(perm_idx)

            return best_hyps, aw, perm_idx
