#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Nested attention-based sequence-to-sequence model (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import copy
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from models.pytorch_v3.attention.attention_seq2seq import AttentionSeq2seq
from models.pytorch_v3.linear import LinearND, Embedding, Embedding_LS
from models.pytorch_v3.encoders.load_encoder import load
from models.pytorch_v3.attention.rnn_decoder import RNNDecoder
from models.pytorch_v3.attention.attention_layer import AttentionMechanism
from models.pytorch_v3.criterion import cross_entropy_label_smoothing
from models.pytorch_v3.ctc.decoders.greedy_decoder import GreedyDecoder
from models.pytorch_v3.ctc.decoders.beam_search_decoder import BeamSearchDecoder


class NestedAttentionSeq2seq(AttentionSeq2seq):

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
                 sharpening_factor=1,  # TODO: change arg name
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
                 num_heads_sub=1,  # ***
                 num_heads_dec=1,  # ***
                 usage_dec_sub='all',  # or update_decoder
                 att_reg_weight=0,  # ***
                 dec_attend_temperature=1,  # ***
                 dec_sigmoid_smoothing=False,  # ***
                 relax_context_vec_dec=False,
                 dec_attention_type='content',  # ***
                 logits_injection=False,  # ***
                 gating=False):  # ***

        super(NestedAttentionSeq2seq, self).__init__(
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
        self.model_type = 'nested_attention'

        # Setting for the encoder
        self.encoder_num_units_sub = encoder_num_units
        if encoder_bidirectional:
            self.encoder_num_units_sub *= 2

        # Setting for the decoder in the sub task
        self.decoder_num_units_1 = decoder_num_units_sub
        self.decoder_num_layers_1 = decoder_num_layers_sub
        self.num_classes_sub = num_classes_sub + 2  # Add <EOS> class
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
        assert sub_loss_weight > 0
        self.sub_loss_weight = sub_loss_weight
        self.ctc_loss_weight_sub = ctc_loss_weight_sub
        if backward_sub:
            self.bwd_weight_1 = sub_loss_weight

        # Setting for decoder attention
        assert usage_dec_sub in ['update_decoder', 'all', 'softmax']
        self.usage_dec_sub = usage_dec_sub
        self.att_reg_weight = att_reg_weight
        self.num_heads_dec = num_heads_dec
        self.logits_injection = logits_injection
        if logits_injection:
            assert usage_dec_sub == 'softmax'
        self.gating = gating

        # Regularization
        self.relax_context_vec_dec = relax_context_vec_dec

        assert decoding_order == 'bahdanau'

        ##################################################
        # Encoder
        # NOTE: overide encoder
        ##################################################
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

        #############################################
        # Main task
        #############################################
        # Decoder (main)
        self.decoder_0_fwd = RNNDecoder(
            input_size=self.encoder_num_units + embedding_dim +
            decoder_num_units_sub if usage_dec_sub != 'softmax' else self.encoder_num_units + embedding_dim,
            rnn_type=decoder_type,
            num_units=decoder_num_units,
            num_layers=decoder_num_layers,
            dropout=dropout_decoder,
            residual=decoder_residual,
            dense_residual=decoder_dense_residual)

        # Attention layer (to encoder states)
        self.attend_0_fwd = AttentionMechanism(
            encoder_num_units=self.encoder_num_units,
            decoder_num_units=decoder_num_units,
            attention_type=attention_type,
            attention_dim=attention_dim,
            sharpening_factor=sharpening_factor,
            sigmoid_smoothing=sigmoid_smoothing,
            out_channels=attention_conv_num_channels,
            kernel_size=attention_conv_width,
            num_heads=num_heads)

        # Attention layer (to decoder states in the sub task)
        self.attend_dec_sub = AttentionMechanism(
            encoder_num_units=decoder_num_units,
            decoder_num_units=decoder_num_units,
            attention_type=dec_attention_type,
            attention_dim=attention_dim,
            sharpening_factor=1 / dec_attend_temperature,
            sigmoid_smoothing=dec_sigmoid_smoothing,
            out_channels=attention_conv_num_channels,
            kernel_size=21,
            num_heads=num_heads_dec)

        if relax_context_vec_dec:
            self.W_c_dec_relax = LinearND(
                decoder_num_units_sub, decoder_num_units_sub,
                dropout=dropout_decoder)

        # Output layer (main)
        self.W_d_0_fwd = LinearND(
            decoder_num_units, bottleneck_dim,
            dropout=dropout_decoder)
        self.W_c_0_fwd = LinearND(
            self.encoder_num_units, bottleneck_dim,
            dropout=dropout_decoder)
        self.fc_0_fwd = LinearND(bottleneck_dim, self.num_classes)

        # Usage of decoder states in the sub task
        if usage_dec_sub in ['all', 'softmax']:
            self.W_c_dec = LinearND(
                decoder_num_units, bottleneck_dim_sub,
                dropout=dropout_decoder)

        if logits_injection:
            self.W_logits_sub = LinearND(
                self.num_classes_sub, decoder_num_units_sub,
                dropout=dropout_decoder)
        if gating:
            self.W_gate = LinearND(
                decoder_num_units + decoder_num_units_sub, decoder_num_units_sub,
                dropout=dropout_decoder)

        #############################################
        # Sub task
        #############################################
        dir = 'bwd' if backward_sub else 'fwd'
        # Bridge layer between the encoder and decoder (sub)
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
        if getattr(self, 'init_dec_state_1_' + dir) != 'zero':
            setattr(self, 'W_dec_init_1_' + dir, LinearND(
                self.encoder_num_units_sub, decoder_num_units_sub))

        # Decoder (sub)
        setattr(self, 'decoder_1_' + dir, RNNDecoder(
                input_size=self.encoder_num_units_sub + embedding_dim_sub,
                rnn_type=decoder_type,
                num_units=decoder_num_units_sub,
                num_layers=decoder_num_layers_sub,
                dropout=dropout_decoder,
                residual=decoder_residual,
                dense_residual=decoder_dense_residual))

        # Attention layer (sub)
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

        # Output layer (sub)
        setattr(self, 'W_d_1_' + dir, LinearND(
            decoder_num_units_sub, bottleneck_dim_sub,
            dropout=dropout_decoder))
        setattr(self, 'W_c_1_' + dir, LinearND(
            self.encoder_num_units_sub, bottleneck_dim_sub,
            dropout=dropout_decoder))
        setattr(self, 'fc_1_' + dir, LinearND(
            bottleneck_dim_sub, self.num_classes_sub))

        # Embedding (sub)
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

        # CTC (sub)
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

    def forward(self, xs, ys, x_lens, y_lens, ys_sub=None, y_lens_sub=None, is_eval=False):
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

        second_pass = False
        if ys_sub is None:
            ys_sub = ys
            y_lens_sub = y_lens
            second_pass = True

        # Reverse the order
        if self.backward_1:
            ys_sub_tmp = copy.deepcopy(ys_sub)
            for b in range(len(xs)):
                ys_sub_tmp[b, :y_lens_sub[b]] = ys_sub[b, :y_lens_sub[b]][::-1]
        else:
            ys_sub_tmp = ys_sub

        # NOTE: ys and ys_sub are padded with -1 here
        # ys_in and ys_in_sub are padded with <EOS> in order to convert to
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
            ys_in_sub = ys_in_sub.cuda()
            ys_out = ys_out.cuda()
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
        # Main + Sub task (attention)
        ##################################################
        # Compute XE loss
        loss_main, loss_sub = self.compute_xe_loss(
            xs, ys_in, ys_out, x_lens, y_lens,
            xs_sub, ys_in_sub, ys_out_sub, x_lens_sub, y_lens_sub)
        loss = loss_main + loss_sub

        ##################################################
        # Sub task (CTC)
        ##################################################
        if self.ctc_loss_weight_sub > 0:
            ctc_loss_sub = self.compute_ctc_loss(
                xs_sub, ys_in_sub[:, 1:] + 1,
                x_lens_sub, y_lens_sub, task=1)

            ctc_loss_sub = ctc_loss_sub * self.ctc_loss_weight_sub
            loss += ctc_loss_sub

        if is_eval:
            loss = loss.data[0]
            loss_main = loss_main.data[0]
            loss_sub = loss_sub.data[0]
        else:
            # Update the probability of scheduled sampling
            self._step += 1
            if self.ss_prob > 0:
                self._ss_prob = min(
                    self.ss_prob, self.ss_prob / self.ss_max_step * self._step)

        if second_pass:
            return loss
        else:
            return loss, loss_main, loss_sub

    def compute_xe_loss(self, enc_out, ys_in, ys_out, x_lens, y_lens,
                        enc_out_sub, ys_in_sub, ys_out_sub, x_lens_sub, y_lens_sub):
        """Compute XE loss.
        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T_in, encoder_num_units]`
            ys_in (torch.autograd.Variable, long): A tensor of size
                `[B, T_out]`, which includes <SOS>
            ys_out (torch.autograd.Variable, long): A tensor of size
                `[B, T_out]`, which includes <EOS>
            x_lens (torch.autograd.Variable, int): A tensor of size `[B]`
            y_lens (torch.autograd.Variable, int): A tensor of size `[B]`

            enc_out_sub (torch.autograd.Variable, float): A tensor of size
                `[B, T_in_sub, encoder_num_units]`
            ys_in_sub (torch.autograd.Variable, long): A tensor of size
                `[B, T_out_sub]`, which includes <SOS>
            ys_out_sub (torch.autograd.Variable, long): A tensor of size
                `[B, T_out_sub]`, which includes <EOS>
            x_lens_sub (torch.autograd.Variable, int): A tensor of size `[B]`
            y_lens_sub (torch.autograd.Variable, int): A tensor of size `[B]`
        Returns:
            loss_main (torch.autograd.Variable, float): A tensor of size `[1]`
            loss_sub (torch.autograd.Variable, float): A tensor of size `[1]`
        """
        # Teacher-forcing
        logits_main, aw, logits_sub, aw_sub, aw_dec = self._decode_train(
            enc_out, x_lens, ys_in,
            enc_out_sub, x_lens_sub, ys_in_sub, y_lens_sub)

        ##################################################
        # Main task
        ##################################################
        if self.main_loss_weight > 0:
            # Output smoothing
            if self.logits_temperature != 1:
                logits_main /= self.logits_temperature

            # Compute XE sequence loss in the main task
            loss_main = F.cross_entropy(
                input=logits_main.view((-1, logits_main.size(2))),
                target=ys_out.view(-1),
                ignore_index=-1, size_average=False) / len(enc_out)

            # Label smoothing (with uniform distribution)
            if self.ls_prob > 0:
                loss_ls_main = cross_entropy_label_smoothing(
                    logits_main,
                    y_lens=y_lens + 1,  # Add <EOS>
                    label_smoothing_prob=self.ls_prob,
                    distribution='uniform',
                    size_average=True)
                loss_main = loss_main * (1 - self.ls_prob) + loss_ls_main

            loss_main = loss_main * self.main_loss_weight

            # Attention regularization
            if self.att_reg_weight > 0:
                loss_main += F.mse_loss(
                    torch.bmm(aw_dec, aw_sub),
                    Variable(aw.data).cuda(),
                    size_average=True, reduce=True) * self.att_reg_weight

        else:
            loss_main = self._create_zero_var((1,))

        ##################################################
        # Sub task
        ##################################################
        # Output smoothing
        if self.logits_temperature != 1:
            logits_sub /= self.logits_temperature

        # Compute XE sequence loss in the sub task
        loss_sub = F.cross_entropy(
            input=logits_sub.view((-1, logits_sub.size(2))),
            target=ys_out_sub.view(-1),
            ignore_index=-1, size_average=False) / len(enc_out_sub)

        # Label smoothing (with uniform distribution)
        if self.ls_prob > 0:
            loss_ls_sub = cross_entropy_label_smoothing(
                logits_sub,
                y_lens=y_lens_sub + 1,  # Add <EOS>
                label_smoothing_prob=self.ls_prob,
                distribution='uniform',
                size_average=True)
            loss_sub = loss_sub * (1 - self.ls_prob) + loss_ls_sub

        loss_sub = loss_sub * self.sub_loss_weight

        # Add coverage term
        if self.coverage_weight != 0:
            raise NotImplementedError

        return loss_main, loss_sub

    def _decode_train(self, enc_out, x_lens, ys,
                      enc_out_sub, x_lens_sub, ys_sub, y_lens_sub):
        """Decoding in the training stage.
        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T_in, encoder_num_units]`
            x_lens (torch.autograd.Variable, int): A tensor of size `[B]`
            ys (torch.autograd.Variable, long): A tensor of size `[B, T_out]`

            enc_out_sub (torch.autograd.Variable, float): A tensor of size
                `[B, T_in_sub, encoder_num_units]`
            x_lens_sub (torch.autograd.Variable, int): A tensor of size `[B]`
            ys_sub (torch.autograd.Variable, long): A tensor of size `[B, T_out_sub]`
            y_lens_sub (torch.autograd.Variable, long): A tensor of size `[B]`
        Returns:
            logits (torch.autograd.Variable, float): A tensor of size
                `[B, T_out, num_classes]`
            aw (torch.autograd.Variable, float): A tensor of size
                `[B, T_out, T_in]`
            logits_sub (torch.autograd.Variable, float): A tensor of size
                `[B, T_out_sub, num_classes_sub]`
            aw_sub (torch.autograd.Variable, float): A tensor of size
                `[B, T_out_sub, T_in_sub]`
            aw_dec (np.ndarray): A tensor of size
                `[B, T_out, T_out_sub]`
        """
        batch_size, max_time = enc_out.size()[:2]
        max_time_sub = enc_out_sub.size(1)
        dir = 'bwd' if self.backward_1 else 'fwd'

        # Pre-computation of embedding
        ys_emb_sub = [self.embed_1(ys_sub[:, t:t + 1])
                      for t in range(ys_sub.size(1))]
        ys_emb_sub = torch.cat(ys_emb_sub, dim=1)
        ys_emb = [self.embed_0(ys[:, t:t + 1])for t in range(ys.size(1))]
        ys_emb = torch.cat(ys_emb, dim=1)

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

        ##################################################
        # At first, compute logits of the character model
        ##################################################
        # Initialization for the character model
        dec_state_sub, dec_out_sub = self._init_dec_state(
            enc_out_sub, x_lens_sub, task=1, dir=dir)
        aw_step_sub = self._create_zero_var(
            (batch_size, max_time_sub, self.num_heads_1))
        context_vec_sub = self._create_zero_var(
            (batch_size, 1, enc_out_sub.size(-1)))

        dec_out_sub_seq, logits_sub, aw_sub = [], [], []
        for t in range(ys_sub.size(1)):
            # Sample for scheduled sampling
            is_sample = self.ss_prob > 0 and t > 0 and self._step > 0 and random.random(
            ) < self._ss_prob
            if is_sample:
                y_sub = self.embed_1(
                    torch.max(logits_sub[-1], dim=2)[1]).detach()
            else:
                y_sub = ys_emb_sub[:, t:t + 1]

            if t > 0:
                # Recurrency
                dec_in_sub = torch.cat(
                    [y_sub, context_vec_sub], dim=-1)
                dec_out_sub, dec_state_sub = getattr(self, 'decoder_1_' + dir)(
                    dec_in_sub, dec_state_sub)

            # Score
            context_vec_sub, aw_step_sub = getattr(self, 'attend_1_' + dir)(
                enc_out_sub, enc_out_sub_a, x_lens_sub, dec_out_sub, aw_step_sub)

            # Generate
            logits_step_sub = getattr(self, 'fc_1_' + dir)(F.tanh(
                getattr(self, 'W_d_1_' + dir)(dec_out_sub) +
                getattr(self, 'W_c_1_' + dir)(context_vec_sub)))

            dec_out_sub_seq.append(dec_out_sub)
            logits_sub.append(logits_step_sub)
            if self.backward_1:
                aw_sub = [aw_step_sub] + aw_sub
            else:
                aw_sub.append(aw_step_sub)
            # NOTE: for attention regularization

        # Concatenate in T_out dimension
        dec_out_sub_seq = torch.cat(dec_out_sub_seq, dim=1)
        logits_sub = torch.cat(logits_sub, dim=1)
        aw_sub = torch.stack(aw_sub, dim=1)

        if self.main_loss_weight == 0:
            return None, None, logits_sub, aw_sub, None

        if self.logits_injection:
            _logits_sub = self.W_logits_sub(logits_sub)

        ##################################################
        # Next, compute logits of the word model
        ##################################################
        # Pre-computation of encoder-side features computing scores
        dec_out_sub_seq_a = []
        for h in range(self.num_heads_dec):
            dec_out_sub_seq_a += [getattr(self.attend_dec_sub,
                                          'W_enc_head' + str(h))(dec_out_sub_seq)]
        dec_out_sub_seq_a = torch.stack(dec_out_sub_seq_a, dim=-1)

        # Initialization for the word model
        dec_state, dec_out = self._init_dec_state(
            enc_out, x_lens, task=0, dir='fwd')
        aw_step_enc = self._create_zero_var(
            (batch_size, max_time, self.num_heads_0))
        aw_step_dec = self._create_zero_var(
            (batch_size, dec_out_sub_seq.size(1), self.num_heads_dec))
        context_vec_enc = self._create_zero_var(
            (batch_size, 1, enc_out.size(-1)))
        context_vec_dec = self._create_zero_var(
            (batch_size, 1, dec_out.size(-1)))

        logits, aw, aw_dec = [], [], []
        for t in range(ys.size(1)):
            # Sample for scheduled sampling
            if self.ss_prob > 0 and t > 0 and self._step > 0 and random.random() < self._ss_prob:
                y = self.embed_0(torch.max(logits[-1], dim=2)[1]).detach()
            else:
                y = ys_emb[:, t:t + 1]

            if t > 0:
                # Recurrency
                dec_in = torch.cat([y, context_vec_enc], dim=-1)
                if self.usage_dec_sub != 'softmax':
                    dec_in = torch.cat(
                        [dec_in, context_vec_dec], dim=-1)
                dec_out, dec_state = self.decoder_0_fwd(dec_in, dec_state)

            # Score for the encoder
            context_vec_enc, aw_step_enc = self.attend_0_fwd(
                enc_out, enc_out_a, x_lens, dec_out, aw_step_enc)

            # Score for the second decoder states
            if self.logits_injection:
                _, aw_step_dec = self.attend_dec_sub(
                    dec_out_sub_seq, dec_out_sub_seq_a, y_lens_sub, dec_out, aw_step_dec)

                context_vec_dec = []
                for h in range(self.num_heads_dec):
                    # Compute context vector
                    context_vec_dec_head = torch.sum(
                        _logits_sub * aw_step_dec[:, :, h:h + 1], dim=1, keepdim=True)
                    context_vec_dec.append(context_vec_dec_head)

                # Concatenate all convtext vectors and attention distributions
                context_vec_dec = torch.cat(context_vec_dec, dim=-1)

                if self.num_heads_dec > 1:
                    context_vec_dec = self.attend_dec_sub.W_mha(
                        context_vec_dec)
            else:
                context_vec_dec, aw_step_dec = self.attend_dec_sub(
                    dec_out_sub_seq, dec_out_sub_seq_a, y_lens_sub, dec_out, aw_step_dec)

            if self.relax_context_vec_dec:
                context_vec_dec = self.W_c_dec_relax(context_vec_dec)

            # Fine-grained gating
            if self.gating:
                gate = F.sigmoid(self.W_gate(
                    torch.cat([dec_out, context_vec_dec], dim=-1)))
                context_vec_dec = gate * context_vec_dec

            # Generate
            out = self.W_d_0_fwd(dec_out) + self.W_c_0_fwd(context_vec_enc)
            if self.usage_dec_sub in ['all', 'softmax']:
                out += self.W_c_dec(context_vec_dec)
            logits_step = self.fc_0_fwd(F.tanh(out))

            logits.append(logits_step)
            aw.append(aw_step_enc)
            aw_dec.append(aw_step_dec)

        # Concatenate in T_out dimension
        logits = torch.cat(logits, dim=1)
        aw = torch.stack(aw, dim=1)
        aw_dec = torch.stack(aw_dec, dim=1)
        # NOTE; aw in the training stage may be used for computing the
        # coverage, so do not convert to numpy yet.

        # TODO: fix these
        aw = aw.squeeze(3)
        aw_sub = aw_sub.squeeze(3)
        aw_dec = aw_dec.squeeze(3)

        return logits, aw, logits_sub, aw_sub, aw_dec

    def decode(self, xs, x_lens, beam_width, max_decode_len, min_decode_len=0,
               beam_width_sub=1, max_decode_len_sub=None, min_decode_len_sub=0,
               length_penalty=0, coverage_penalty=0, task_index=0,
               teacher_forcing=False, ys_sub=None, y_lens_sub=None):
        """Decoding in the inference stage.
        Args:
            xs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            x_lens (np.ndarray): A tensor of size `[B]`
            beam_width (int): the size of beam in the main task
            max_decode_len (int): the maximum sequence length of tokens in the main task
            min_decode_len (int): the minimum sequence length of tokens in the main task
            beam_width_sub (int): the size of beam in the sub task
            max_decode_len_sub (int): the maximum sequence length of tokens in the sub task
            min_decode_len_sub (int): the minimum sequence length of tokens in the sub task
            length_penalty (float):
            coverage_penalty (float):
            task_index (int): the index of a task
            teacher_forcing (bool):
            ys_sub ():
            y_lens_sub ():
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B]`
            aw ():
            best_hyps_sub (np.ndarray): A tensor of size `[B]`
            aw_sub ():
            aw_dec ():
            perm_idx (np.ndarray): A tensor of size `[B]`
        """
        self.eval()

        if teacher_forcing:
            # Reverse the order
            if self.backward_1:
                ys_sub_tmp = copy.deepcopy(ys_sub)
                for b in range(len(xs)):
                    ys_sub_tmp[b, :y_lens_sub[b]
                               ] = ys_sub[b, :y_lens_sub[b]][::-1]
            else:
                ys_sub_tmp = ys_sub

            ys_in_sub = self._create_var((ys_sub.shape[0], ys_sub.shape[1] + 1),
                                         fill_value=self.eos_1, dtype='long')
            ys_in_sub.data[:, 0] = self.sos_1
            for b in range(len(xs)):
                ys_in_sub.data[b, 1:y_lens_sub[b] + 1] = torch.from_numpy(
                    ys_sub_tmp[b, :y_lens_sub[b]])

            if self.use_cuda:
                ys_in_sub = ys_in_sub.cuda()

            # Wrap by Variable
            y_lens_sub = self.np2var(y_lens_sub, dtype='int')
        else:
            ys_in_sub = None

        # Wrap by Variable
        xs = self.np2var(xs)
        x_lens = self.np2var(x_lens, dtype='int')

        dir = 'bwd'if self.backward_1 else 'fwd'

        # Encode acoustic features
        if task_index == 0:
            enc_out, x_lens, enc_out_sub, x_lens_sub, perm_idx = self._encode(
                xs, x_lens, is_multi_task=True)

            # Next, decode by word-based decoder with character outputs
            if teacher_forcing:
                ys_in_sub = ys_in_sub[perm_idx]

            if beam_width == 1:
                best_hyps, aw, best_hyps_sub, aw_sub, aw_dec = self._decode_infer_joint_greedy(
                    enc_out, x_lens, enc_out_sub, x_lens_sub,
                    max_decode_len=max_decode_len,
                    max_decode_len_sub=max_decode_len_sub,
                    teacher_forcing=teacher_forcing,
                    ys_sub=ys_in_sub)
            else:
                best_hyps, aw, best_hyps_sub, aw_sub, aw_dec = self._decode_infer_joint_beam(
                    enc_out, x_lens, enc_out_sub, x_lens_sub,
                    beam_width=beam_width,
                    max_decode_len=max_decode_len,
                    min_decode_len=min_decode_len,
                    beam_width_sub=beam_width_sub,
                    max_decode_len_sub=max_decode_len_sub,
                    min_decode_len_sub=min_decode_len_sub,
                    length_penalty=length_penalty,
                    coverage_penalty=coverage_penalty,
                    teacher_forcing=teacher_forcing,
                    ys_sub=ys_in_sub)

        elif task_index == 1:
            _, _, enc_out, x_lens, perm_idx = self._encode(
                xs, x_lens, is_multi_task=True)

            if beam_width == 1:
                best_hyps, aw = self._decode_infer_greedy(
                    enc_out, x_lens, max_decode_len, task=1, dir=dir)
            else:
                best_hyps, aw = self._decode_infer_beam(
                    enc_out, x_lens, beam_width, max_decode_len,
                    length_penalty, coverage_penalty, task=1, dir=dir)
        else:
            raise ValueError

        # TODO: fix this
        # aw = aw[:, :, :, 0]
        # aw_sub = aw_sub[:, :, :, 0]
        # aw_dec = aw_dec[:, :, :, 0]

        # Permutate indices to the original order
        if perm_idx is None:
            perm_idx = np.arange(0, len(xs), 1)
        else:
            perm_idx = self.var2np(perm_idx)

        if task_index == 0:
            return best_hyps, aw, best_hyps_sub, aw_sub, aw_dec, perm_idx
        elif task_index == 1:
            return best_hyps, aw, perm_idx

    def _decode_infer_joint_greedy(self, enc_out, x_lens, enc_out_sub, x_lens_sub,
                                   max_decode_len, max_decode_len_sub,
                                   teacher_forcing=False, ys_sub=None,
                                   reverse_backward=True):
        """Greedy decoding in the inference stage.
        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T_in, encoder_num_units]`
            x_lens (torch.autograd.Variable, int): A tensor of size `[B]`
            enc_out_sub (torch.autograd.Variable, float): A tensor of size
                `[B, T_in_sub, encoder_num_units]`
            x_lens_sub (torch.autograd.Variable, int): A tensor of size `[B]`
            max_decode_len (int): the maximum sequence length of tokens in the main task
            max_decode_len_sub (int): the maximum sequence length of tokens in the sub task
            teacher_forcing (bool):
            ys_sub ():
            reverse_backward (bool):
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B, T_out]`
            aw (np.ndarray): A tensor of size `[B, T_out, T_in]`
            best_hyps_sub (np.ndarray): A tensor of size `[B, T_out_sub]`
            aw_sub (np.ndarray): A tensor of size `[B, T_out_sub, T_in]`
            aw_dec (np.ndarray): A tensor of size `[B, T_out, T_out_sub]`
        """
        batch_size, max_time = enc_out.size()[:2]
        dir = 'bwd' if self.backward_1 else 'fwd'

        # Pre-computation of encoder-side features computing scores
        enc_out_a, enc_out_sub_a = [], []
        for h in range(self.num_heads_1):
            enc_out_sub_a += [getattr(self.attend_1_fwd,
                                      'W_enc_head' + str(h))(enc_out_sub)]
        for h in range(self.num_heads_0):
            enc_out_a += [getattr(self.attend_0_fwd,
                                  'W_enc_head' + str(h))(enc_out)]
        enc_out_a = torch.stack(enc_out_a, dim=-1)
        enc_out_sub_a = torch.stack(enc_out_sub_a, dim=-1)

        ##################################################
        # At first, decode by the second decoder
        ##################################################
        # Initialization for the character model
        dec_state_sub, dec_out_sub = self._init_dec_state(
            enc_out_sub, x_lens_sub, task=1, dir=dir)
        aw_step_sub = self._create_zero_var(
            (batch_size, enc_out_sub.size(1), self.num_heads_1), volatile=True)
        context_vec_sub = self._create_zero_var(
            (1, 1, enc_out_sub.size(-1)), volatile=True)

        # Start from <SOS>
        y_sub = self._create_var(
            (batch_size, 1), fill_value=self.sos_1, dtype='long')

        dec_out_sub_seq = []
        best_hyps_sub, aw_sub = [], []
        logits_sub = []
        y_lens_sub = np.zeros((batch_size,), dtype=np.int32)
        eos_flag = [False] * batch_size
        for t_sub in range(max_decode_len_sub + 1):
            if teacher_forcing:
                y_sub = ys_sub[:, t_sub:t_sub + 1]
            y_sub = self.embed_1(y_sub)

            if t_sub > 0:
                # Recurrency
                dec_in_sub = torch.cat([y_sub, context_vec_sub], dim=-1)
                dec_out_sub, dec_state_sub = getattr(self, 'decoder_1_' + dir)(
                    dec_in_sub, dec_state_sub)

            # Score
            context_vec_sub, aw_step_sub = getattr(self, 'attend_1_' + dir)(
                enc_out_sub, enc_out_sub_a, x_lens_sub, dec_out_sub, aw_step_sub)

            # Generate
            logits_step_sub = getattr(self, 'fc_1_' + dir)(F.tanh(
                getattr(self, 'W_d_1_' + dir)(dec_out_sub) +
                getattr(self, 'W_c_1_' + dir)(context_vec_sub)))

            # Pick up 1-best
            y_sub = torch.max(logits_step_sub.squeeze(1), dim=1)[
                1].unsqueeze(1)
            dec_out_sub_seq.append(dec_out_sub)
            logits_sub.append(logits_step_sub)
            aw_sub.append(aw_step_sub)
            best_hyps_sub.append(y_sub)

            for b in range(batch_size):
                if not eos_flag[b]:
                    y_lens_sub[b] += 1
                    # NOTE: include <EOS>
                    if y_sub.data.cpu().numpy()[b] == self.eos_1:
                        eos_flag[b] = True

            # Break if <EOS> is outputed in all mini-batch
            if torch.sum(y_sub.data == self.eos_1) == y_sub.numel():
                break

        # Concatenate in T_out dimension
        dec_out_sub_seq = torch.cat(dec_out_sub_seq, dim=1)
        logits_sub = torch.cat(logits_sub, dim=1)
        aw_sub = torch.stack(aw_sub, dim=1)
        best_hyps_sub = torch.cat(best_hyps_sub, dim=1)

        if self.logits_injection:
            logits_sub = self.W_logits_sub(logits_sub)

        ##################################################
        # Next, compute logits of the word model
        ##################################################
        # Pre-computation of encoder-side features computing scores
        dec_out_sub_seq_a = []
        for h in range(self.num_heads_dec):
            dec_out_sub_seq_a += [getattr(self.attend_dec_sub,
                                          'W_enc_head' + str(h))(dec_out_sub_seq)]
        dec_out_sub_seq_a = torch.stack(dec_out_sub_seq_a, dim=-1)

        # Initialization for the word model
        dec_state, dec_out = self._init_dec_state(
            enc_out, x_lens, task=0, dir='fwd')
        aw_step_enc = self._create_zero_var(
            (batch_size, max_time, self.num_heads_0), volatile=True)
        aw_step_dec = self._create_zero_var(
            (batch_size, dec_out_sub_seq.size(1), self.num_heads_dec),
            volatile=True)
        context_vec_enc = self._create_zero_var(
            (batch_size, 1, enc_out.size(-1)), volatile=True)
        context_vec_dec = self._create_zero_var(
            (batch_size, 1, dec_out.size(-1)), volatile=True)

        y_lens_sub = self.np2var(y_lens_sub + 1, dtype='int')
        # NOTE: add <SOS>
        # assert max(y_lens_sub.data) > 0

        # Start from <SOS>
        y = self._create_var(
            (batch_size, 1), fill_value=self.sos_0, dtype='long')

        best_hyps, aw_enc = [], []
        aw_dec = []
        for t in range(max_decode_len + 1):
            y = self.embed_0(y)

            if t > 0:
                # Recurrency
                dec_in = torch.cat([y, context_vec_enc], dim=-1)
                if self.usage_dec_sub != 'softmax':
                    dec_in = torch.cat(
                        [dec_in, context_vec_dec], dim=-1)
                dec_out, dec_state = self.decoder_0_fwd(dec_in, dec_state)

            # Score for the encoder
            context_vec_enc, aw_step_enc = self.attend_0_fwd(
                enc_out, enc_out_a, x_lens, dec_out, aw_step_enc)

            # Score for the second decoder states
            if self.logits_injection:
                _, aw_step_dec = self.attend_dec_sub(
                    dec_out_sub_seq, dec_out_sub_seq_a, y_lens_sub, dec_out, aw_step_dec)

                context_vec_dec = []
                for h in range(self.num_heads_dec):
                    # Compute context vector
                    context_vec_dec_head = torch.sum(
                        logits_sub * aw_step_dec[:, :, h:h + 1], dim=1, keepdim=True)
                    context_vec_dec.append(context_vec_dec_head)

                # Concatenate all convtext vectors and attention distributions
                context_vec_dec = torch.cat(context_vec_dec, dim=-1)

                if self.num_heads_dec > 1:
                    context_vec_dec = self.attend_dec_sub.W_mha(
                        context_vec_dec)
            else:
                context_vec_dec, aw_step_dec = self.attend_dec_sub(
                    dec_out_sub_seq, dec_out_sub_seq_a, y_lens_sub, dec_out, aw_step_dec)

            if self.relax_context_vec_dec:
                context_vec_dec = self.W_c_dec_relax(context_vec_dec)

            # Fine-grained gating
            if self.gating:
                gate = F.sigmoid(self.W_gate(
                    torch.cat([dec_out, context_vec_dec], dim=-1)))
                context_vec_dec = gate * context_vec_dec

            # Generate
            out = self.W_d_0_fwd(dec_out) + self.W_c_0_fwd(context_vec_enc)
            if self.usage_dec_sub in ['all', 'softmax']:
                out += self.W_c_dec(context_vec_dec)
            logits_step = self.fc_0_fwd(F.tanh(out))

            # Pick up 1-best
            y = torch.max(logits_step.squeeze(1), dim=1)[1].unsqueeze(1)
            aw_enc.append(aw_step_enc)
            aw_dec.append(aw_step_dec)
            best_hyps.append(y)

            # Break if <EOS> is outputed in all mini-batch
            if torch.sum(y.data == self.eos_0) == y.numel():
                break

        # Concatenate in T_out dimension
        aw_enc = torch.stack(aw_enc, dim=1)
        aw_dec = torch.stack(aw_dec, dim=1)
        best_hyps = torch.cat(best_hyps, dim=1)

        # Convert to numpy
        aw_enc = self.var2np(aw_enc)
        aw_sub = self.var2np(aw_sub)
        aw_dec = self.var2np(aw_dec)
        best_hyps = self.var2np(best_hyps)
        best_hyps_sub = self.var2np(best_hyps_sub)

        # Reverse the order
        if self.backward_1 and reverse_backward:
            y_lens_sub = self.var2np(y_lens_sub)
            for b in range(batch_size):
                best_hyps_sub[b, :y_lens_sub[b]
                              ] = best_hyps_sub[b, :y_lens_sub[b]][::-1]

        return best_hyps, aw_enc, best_hyps_sub, aw_sub, aw_dec

    def _decode_infer_joint_beam(self, enc_out, x_lens, enc_out_sub, x_lens_sub,
                                 beam_width, max_decode_len, min_decode_len,
                                 beam_width_sub, max_decode_len_sub, min_decode_len_sub,
                                 length_penalty, coverage_penalty,
                                 teacher_forcing=False, ys_sub=None,
                                 reverse_backward=True):
        """Greedy decoding in the inference stage.
        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T_in, encoder_num_units]`
            x_lens (torch.autograd.Variable, int): A tensor of size `[B]`
            enc_out_sub (torch.autograd.Variable, float): A tensor of size
                `[B, T_in_sub, encoder_num_units]`
            x_lens_sub (torch.autograd.Variable, int): A tensor of size `[B]`
            beam_width (int): the size of beam in the main task
            max_decode_len (int): the maximum sequence length of tokens in the main task
            min_decode_len (int): the minimum sequence length of tokens in the main task
            beam_width_sub (int): the size of beam in the sub task
            max_decode_len_sub (int): the maximum sequence length of tokens in the sub task
            min_decode_len_sub (int): the minimum sequence length of tokens in the sub task
            length_penalty (float): length penalty in beam search decoding
            coverage_penalty (float): coverage penalty in beam search decoding
            teacher_forcing (bool):
            ys_sub ():
            reverse_backward (bool):
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B, T_out]`
            aw (np.ndarray): A tensor of size `[B, T_out, T_in]`
            best_hyps_sub (np.ndarray): A tensor of size `[B, T_out_sub]`
            aw_sub (np.ndarray): A tensor of size `[B, T_out_sub, T_in]`
            aw_dec (np.ndarray): A tensor of size `[B, T_out, T_out_sub]`
        """
        batch_size, max_time = enc_out.size()[:2]
        dir = 'bwd' if self.backward_1 else 'fwd'
        if teacher_forcing:
            beam_width_sub = 1

        min_decode_len_ratio = 0.2

        # Pre-computation of encoder-side features computing scores
        enc_out_a, enc_out_sub_a = [], []
        for h in range(self.num_heads_1):
            enc_out_sub_a += [getattr(self.attend_1_fwd,
                                      'W_enc_head' + str(h))(enc_out_sub)]
        for h in range(self.num_heads_0):
            enc_out_a += [getattr(self.attend_0_fwd,
                                  'W_enc_head' + str(h))(enc_out)]
        enc_out_a = torch.stack(enc_out_a, dim=-1)
        enc_out_sub_a = torch.stack(enc_out_sub_a, dim=-1)

        ##################################################
        # At first, decode by the second decoder
        ##################################################
        dec_out_sub_seq, aw_sub, best_hyps_sub = [], [], []
        logits_sub = []
        y_lens_sub = np.zeros((batch_size,), dtype=np.int32)
        for b in range(batch_size):
            # Initialization for the character model per utterance
            dec_state_sub, dec_out_sub = self._init_dec_state(
                enc_out_sub[b: b + 1], x_lens_sub[b:b + 1], task=1, dir=dir)
            aw_step_sub = self._create_zero_var(
                (1, x_lens_sub[b].data[0], self.num_heads_1), volatile=True)
            context_vec_sub = self._create_zero_var(
                (1, 1, enc_out_sub.size(-1)), volatile=True)

            complete_sub = []
            beam_sub = [{'hyp': [self.sos_1],
                         'score': 0,  # log 1
                         'dec_state': dec_state_sub,
                         'dec_outs': [dec_out_sub],  # NOTE: keep all outputs
                         'context_vec': context_vec_sub,
                         'aw_steps': [aw_step_sub],
                         'logits_sub': []}]
            for t_sub in range(max_decode_len_sub + 1):
                new_beam_sub = []
                for i_beam in range(len(beam_sub)):
                    if teacher_forcing:
                        y_sub = ys_sub[:, t_sub:t_sub + 1]
                    else:
                        y_sub = self._create_var(
                            (1, 1), fill_value=beam_sub[i_beam]['hyp'][-1], dtype='long')
                    y_sub = self.embed_1(y_sub)

                    if t_sub == 0:
                        dec_out_sub = beam_sub[i_beam]['dec_outs'][-1]
                    else:
                        # Recurrency
                        dec_in_sub = torch.cat(
                            [y_sub, context_vec_sub], dim=-1)
                        dec_out_sub, dec_state_sub = getattr(
                            self, 'decoder_1_' + dir)(dec_in_sub, beam_sub[i_beam]['dec_state'])

                    # Score
                    context_vec_sub, aw_step_sub = getattr(self, 'attend_1_' + dir)(
                        enc_out_sub[b:b + 1, :x_lens_sub.data[b]],
                        enc_out_sub_a[b:b + 1, :x_lens_sub.data[b]],
                        x_lens_sub[b:b + 1],
                        dec_out_sub, beam_sub[i_beam]['aw_steps'][-1])

                    # Generate
                    logits_step_sub = getattr(self, 'fc_1_' + dir)(F.tanh(
                        getattr(self, 'W_d_1_' + dir)(dec_out_sub) +
                        getattr(self, 'W_c_1_' + dir)(context_vec_sub)))

                    # Path through the softmax layer & convert to log-scale
                    log_probs_sub = F.log_softmax(
                        logits_step_sub.squeeze(1), dim=1)
                    # NOTE: `[1 (B), 1, num_classes_sub]` -> `[1 (B), num_classes_sub]`

                    # Pick up the top-k scores
                    log_probs_sub_topk, indices_sub_topk = log_probs_sub.topk(
                        beam_width_sub, dim=1, largest=True, sorted=True)

                    if self.logits_injection:
                        logits_step_sub = self.W_logits_sub(logits_step_sub)

                    for k in range(beam_width_sub):
                        # Exclude short hypotheses
                        if indices_sub_topk[0, k].data[0] == self.eos_1 and len(beam_sub[i_beam]['hyp']) < min_decode_len_sub:
                            continue
                        # if indices_sub_topk[0, k].data[0] == self.eos_1 and len(beam_sub[i_beam]['hyp']) < x_lens_sub[b].data[0] * min_decode_len_ratio:
                        #     continue
                        # TODO: controll by beam width

                        # Add length penalty
                        score_sub = beam_sub[i_beam]['score'] + \
                            log_probs_sub_topk.data[0, k] + length_penalty

                        # Add coverage penalty
                        if coverage_penalty > 0:
                            raise NotImplementedError

                            threshold = 0.5
                            aw_steps_sub = torch.cat(
                                beam_sub[i_beam]['aw_steps'], dim=0).sum(0).squeeze(1)

                            # Google NMT
                            # cov_sum = torch.where(
                            #     aw_steps_sub < threshold, aw_steps_sub, torch.ones_like(aw_steps_sub) * threshold).sum(0)
                            # score_sub += torch.log(cov_sum) * coverage_penalty

                            # Toward better decoding
                            cov_sum = torch.where(
                                aw_steps_sub > threshold, aw_steps_sub, torch.zeros_like(aw_steps_sub)).sum(0)
                            score_sub += cov_sum * coverage_penalty

                        new_beam_sub.append(
                            {'hyp': beam_sub[i_beam]['hyp'] + [indices_sub_topk[0, k].data[0]],
                             'score': score_sub,
                             'dec_state': copy.deepcopy(dec_state_sub),
                             'dec_outs': beam_sub[i_beam]['dec_outs'] + [dec_out_sub],
                             'context_vec': context_vec_sub,
                             'aw_steps': beam_sub[i_beam]['aw_steps'] + [aw_step_sub],
                             'logits_sub': beam_sub[i_beam]['logits_sub'] + [logits_step_sub]})

                new_beam_sub = sorted(
                    new_beam_sub, key=lambda x: x['score'], reverse=True)

                # Remove complete hypotheses
                not_complete_sub = []
                for cand in new_beam_sub[:beam_width_sub]:
                    if cand['hyp'][-1] == self.eos_1:
                        complete_sub.append(cand)
                    else:
                        not_complete_sub.append(cand)

                if len(complete_sub) >= beam_width_sub:
                    complete_sub = complete_sub[:beam_width_sub]
                    break

                beam_sub = not_complete_sub[:beam_width_sub]

            if len(complete_sub) == 0:
                complete_sub = beam_sub

            # Renormalized hypotheses by length
            if length_penalty > 0:
                for j in range(len(complete_sub)):
                    complete_sub[j]['score'] += len(complete_sub[j]
                                                    ['hyp']) * length_penalty

            complete_sub = sorted(
                complete_sub, key=lambda x: x['score'], reverse=True)

            dec_out_sub_seq.append(
                torch.cat(complete_sub[0]['dec_outs'][1:], dim=1))
            aw_sub.append(complete_sub[0]['aw_steps'][1:])
            logits_sub.append(
                torch.cat(complete_sub[0]['logits_sub'], dim=1))
            best_hyps_sub.append(np.array(complete_sub[0]['hyp'][1:]))
            y_lens_sub[b] = len(complete_sub[0]['hyp'][1:])

        ##################################################
        # Next, decode by the first decoder
        ##################################################
        y_lens_sub = self.np2var(y_lens_sub, dtype='int')
        # assert max(y_lens_sub.data[0]) > 0

        best_hyps, aw, aw_dec = [], [], []
        for b in range(enc_out.size(0)):
            # Pre-computation of encoder-side features computing scores
            dec_out_sub_seq_a = []
            for h in range(self.num_heads_dec):
                dec_out_sub_seq_a += [getattr(self.attend_dec_sub,
                                              'W_enc_head' + str(h))(dec_out_sub_seq[b])]
            dec_out_sub_seq_a = torch.stack(dec_out_sub_seq_a, dim=-1)

            # Initialization for the word model per utterance
            dec_state, dec_out = self._init_dec_state(
                enc_out[b:b + 1], x_lens[b:b + 1], task=0, dir='fwd')
            aw_step_enc = self._create_zero_var(
                (1, x_lens[b].data[0], self.num_heads_0), volatile=True)
            aw_step_dec = self._create_zero_var(
                (1, dec_out_sub_seq[b].size(1), self.num_heads_dec), volatile=True)
            context_vec_enc = self._create_zero_var(
                (1, 1, enc_out.size(-1)), volatile=True)
            context_vec_dec = self._create_zero_var(
                (1, 1, dec_out.size(-1)), volatile=True)

            complete = []
            beam = [{'hyp': [self.sos_0],
                     'score': 0,  # log 1
                     'dec_state': dec_state,
                     'dec_out': dec_out,
                     'context_vec_enc': context_vec_enc,
                     'context_vec_dec': context_vec_dec,
                     'aw_steps_enc': [aw_step_enc],
                     'aw_steps_dec': [aw_step_dec]}]
            for t in range(max_decode_len + 1):
                new_beam = []
                for i_beam in range(len(beam)):
                    y = self._create_var(
                        (1, 1), fill_value=beam[i_beam]['hyp'][-1], dtype='long')
                    y = self.embed_0(y)

                    if t == 0:
                        dec_out = beam[i_beam]['dec_out']
                    else:
                        # Recurrency
                        dec_in = torch.cat([y, context_vec_enc], dim=-1)
                        if self.usage_dec_sub != 'softmax':
                            dec_in = torch.cat(
                                [dec_in, context_vec_dec], dim=-1)
                        dec_out, dec_state = self.decoder_0_fwd(
                            dec_in, beam[i_beam]['dec_state'])

                    # Score for the encoder
                    context_vec_enc, aw_step_enc = self.attend_0_fwd(
                        enc_out[b:b + 1, :x_lens.data[b]],
                        enc_out_a[b:b + 1, :x_lens.data[b]],
                        x_lens[b:b + 1],
                        dec_out, beam[i_beam]['aw_steps_enc'][-1])

                    # Score for the second decoder states
                    if self.logits_injection:
                        _, aw_step_dec = self.attend_dec_sub(
                            dec_out_sub_seq[b][: y_lens_sub.data[b]],
                            dec_out_sub_seq_a[0:1, :y_lens_sub.data[b]],
                            y_lens_sub[b:b + 1],
                            dec_out, beam[i_beam]['aw_steps_dec'][-1])

                        context_vec_dec = []
                        for h in range(self.num_heads_dec):
                            # Compute context vector
                            context_vec_dec_head = torch.sum(
                                logits_sub[b] * aw_step_dec[0:1, :, h:h + 1], dim=1, keepdim=True)
                            context_vec_dec.append(context_vec_dec_head)

                        # Concatenate all convtext vectors and attention distributions
                        context_vec_dec = torch.cat(
                            context_vec_dec, dim=-1)

                        if self.num_heads_dec > 1:
                            context_vec_dec = self.attend_dec_sub.W_mha(
                                context_vec_dec)
                    else:
                        context_vec_dec, aw_step_dec = self.attend_dec_sub(
                            dec_out_sub_seq[b][: y_lens_sub.data[b]],
                            dec_out_sub_seq_a[0:1, :y_lens_sub.data[b]],
                            y_lens_sub[b:b + 1],
                            dec_out, beam[i_beam]['aw_steps_dec'][-1])

                    if self.relax_context_vec_dec:
                        context_vec_dec = self.W_c_dec_relax(
                            context_vec_dec)

                    # Fine-grained gating
                    if self.gating:
                        gate = F.sigmoid(self.W_gate(
                            torch.cat([dec_out, context_vec_dec], dim=-1)))
                        context_vec_dec = gate * context_vec_dec

                    # Generate
                    out = self.W_d_0_fwd(dec_out) + \
                        self.W_c_0_fwd(context_vec_enc)
                    if self.usage_dec_sub in ['all', 'softmax']:
                        out += self.W_c_dec(context_vec_dec)
                    logits_step = self.fc_0_fwd(F.tanh(out))

                    # Path through the softmax layer & convert to log-scale
                    log_probs = F.log_softmax(logits_step.squeeze(1), dim=1)

                    # Pick up the top-k scores
                    log_probs_topk, indices_topk = log_probs.topk(
                        beam_width, dim=1, largest=True, sorted=True)

                    for k in range(beam_width):
                        # Exclude short hypotheses
                        if indices_topk[0, k].data[0] == self.eos_0 and len(beam[i_beam]['hyp']) < min_decode_len:
                            continue

                        # Add length penalty
                        score = beam[i_beam]['score'] + \
                            log_probs_topk.data[0, k] + length_penalty

                        # Add coverage penalty
                        if coverage_penalty > 0:
                            raise NotImplementedError

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
                            {'hyp': beam[i_beam]['hyp'] + [indices_topk[0, k].data[0]],
                             'score': score,
                             'dec_state': copy.deepcopy(dec_state),
                             'dec_out': dec_out,
                             'context_vec_enc': context_vec_enc,
                             'context_vec_dec': context_vec_dec,
                             'aw_steps_enc': beam[i_beam]['aw_steps_enc'] + [aw_step_enc],
                             'aw_steps_dec': beam[i_beam]['aw_steps_dec'] + [aw_step_dec]})

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

            # Renormalized hypotheses by length
            if length_penalty > 0:
                for j in range(len(complete)):
                    complete[j]['score'] += len(complete[j]
                                                ['hyp']) * length_penalty

            complete = sorted(
                complete, key=lambda x: x['score'], reverse=True)
            aw.append(complete[0]['aw_steps_enc'][1:])
            aw_dec.append(complete[0]['aw_steps_dec'][1:])
            best_hyps.append(np.array(complete[0]['hyp'][1:]))

        # Concatenate in T_out dimension
        for j in range(len(aw)):
            for k in range(len(aw_sub[j])):
                aw_sub[j][k] = aw_sub[j][k][:, :, 0]  # TODO: fix for MHA
            aw_sub[j] = self.var2np(torch.stack(aw_sub[j], dim=1).squeeze(0))

            for k in range(len(aw[j])):
                aw[j][k] = aw[j][k][:, :, 0]  # TODO: fix for MHA
                aw_dec[j][k] = aw_dec[j][k][:, :, 0]  # TODO: fix for MHA
            aw[j] = self.var2np(torch.stack(aw[j], dim=1).squeeze(0))
            aw_dec[j] = self.var2np(torch.stack(aw_dec[j], dim=1).squeeze(0))

        # Reverse the order
        if self.backward_1 and reverse_backward:
            y_lens_sub = self.var2np(y_lens_sub)
            for b in range(batch_size):
                best_hyps_sub[b][:y_lens_sub[b]
                                 ] = best_hyps_sub[b][:y_lens_sub[b]][::-1]

        return best_hyps, aw, best_hyps_sub, aw_sub, aw_dec
