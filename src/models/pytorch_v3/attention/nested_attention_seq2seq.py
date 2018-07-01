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

from src.models.pytorch_v3.attention.attention_seq2seq import AttentionSeq2seq
from src.models.pytorch_v3.linear import LinearND, Embedding
from src.models.pytorch_v3.encoders.load_encoder import load
from src.models.pytorch_v3.attention.rnn_decoder import RNNDecoder
from src.models.pytorch_v3.attention.attention_layer import AttentionMechanism, MultiheadAttentionMechanism
from src.models.pytorch_v3.criterion import cross_entropy_label_smoothing
from src.models.pytorch_v3.ctc.decoders.greedy_decoder import GreedyDecoder
from src.models.pytorch_v3.ctc.decoders.beam_search_decoder import BeamSearchDecoder
from src.models.pytorch_v3.utils import np2var, var2np, pad_list


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
                 init_dec_state='zero',
                 sharpening_factor=1,  # TODO: change arg name
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
            input_type='speech',
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
            num_heads=num_heads)
        self.model_type = 'nested_attention'

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

        # Setting for the RNNLM fusion
        self.rnnlm_fusion_type = False
        self.rnnlm_1_fwd = None
        self.rnnlm_weight = 0

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
        self._fix_second_decoder = False

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
        if num_heads > 1:
            self.attend_0_fwd = MultiheadAttentionMechanism(
                encoder_num_units=self.encoder_num_units,
                decoder_num_units=decoder_num_units,
                attention_type=attention_type,
                attention_dim=attention_dim,
                sharpening_factor=sharpening_factor,
                sigmoid_smoothing=sigmoid_smoothing,
                out_channels=attention_conv_num_channels,
                kernel_size=attention_conv_width,
                num_heads=num_heads)
        else:
            self.attend_0_fwd = AttentionMechanism(
                encoder_num_units=self.encoder_num_units,
                decoder_num_units=decoder_num_units,
                attention_type=attention_type,
                attention_dim=attention_dim,
                sharpening_factor=sharpening_factor,
                sigmoid_smoothing=sigmoid_smoothing,
                out_channels=attention_conv_num_channels,
                kernel_size=attention_conv_width)

        # Attention layer (to decoder states in the sub task)
        if num_heads_dec > 1:
            self.attend_dec_sub = MultiheadAttentionMechanism(
                encoder_num_units=decoder_num_units,
                decoder_num_units=decoder_num_units,
                attention_type=dec_attention_type,
                attention_dim=attention_dim,
                sharpening_factor=1 / dec_attend_temperature,
                sigmoid_smoothing=dec_sigmoid_smoothing,
                out_channels=attention_conv_num_channels,
                kernel_size=21,
                num_heads=num_heads_dec)
        else:
            self.attend_dec_sub = AttentionMechanism(
                encoder_num_units=decoder_num_units,
                decoder_num_units=decoder_num_units,
                attention_type=dec_attention_type,
                attention_dim=attention_dim,
                sharpening_factor=1 / dec_attend_temperature,
                sigmoid_smoothing=dec_sigmoid_smoothing,
                out_channels=attention_conv_num_channels,
                kernel_size=21)

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
        if num_heads_sub > 1:
            setattr(self, 'attend_1_' + dir, MultiheadAttentionMechanism(
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
            setattr(self, 'attend_1_' + dir, AttentionMechanism(
                encoder_num_units=self.encoder_num_units_sub,
                decoder_num_units=decoder_num_units_sub,
                attention_type=attention_type,
                attention_dim=attention_dim,
                sharpening_factor=sharpening_factor,
                sigmoid_smoothing=sigmoid_smoothing,
                out_channels=attention_conv_num_channels,
                kernel_size=attention_conv_width))

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

        # Initialize weight matrices
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

    def forward(self, xs, ys, ys_sub=None, is_eval=False):
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

        second_pass = False
        if ys_sub is None:
            ys_sub = copy.deepcopy(ys)
            second_pass = True

        # Sort by lenghts in the descending order
        if is_eval and self.encoder_type != 'cnn':
            perm_idx = sorted(list(range(0, len(xs), 1)),
                              key=lambda i: len(xs[i]), reverse=True)
            xs = [xs[i] for i in perm_idx]
            ys = [ys[i] for i in perm_idx]
            ys_sub = [ys_sub[i] for i in perm_idx]
            # NOTE: must be descending order for pack_padded_sequence
            # NOTE: assumed that xs is already sorted in the training stage

        # Wrap by Variable
        ys = [np2var(np.fromiter(y, dtype=np.int64), self.device_id).long()
              for y in ys]
        if self.backward_1:
            ys_sub = [np2var(np.fromiter(y[::-1], dtype=np.int64), self.device_id).long()
                      for y in ys_sub]
        else:
            ys_sub = [np2var(np.fromiter(y, dtype=np.int64), self.device_id).long()
                      for y in ys_sub]

        # Encode acoustic features
        xs, x_lens, xs_sub, x_lens_sub = self._encode(xs, is_multi_task=True)

        # Compute XE loss (main + sub)
        loss_main, loss_sub, acc_main, acc_sub = self.compute_xe_loss_mtl(
            xs, ys, x_lens, xs_sub, ys_sub, x_lens_sub)
        loss = loss_main + loss_sub

        # Sub task (CTC)
        if self.ctc_loss_weight_sub > 0:
            # Wrap by Variable
            ys_sub_ctc = [np2var(np.fromiter(y, dtype=np.int64), self.device_id).long()
                          for y in ys_sub]

            ctc_loss_sub = self.compute_ctc_loss(
                xs_sub, ys_sub_ctc, x_lens_sub, task=1) * self.ctc_loss_weight_sub
            loss += ctc_loss_sub

        if not is_eval:
            # Update the probability of scheduled sampling
            self._step += 1
            if self.ss_prob > 0:
                self._ss_prob = min(
                    self.ss_prob, self.ss_prob / self.ss_max_step * self._step)

        if second_pass:
            acc = acc_main * self.main_loss_weight + acc_sub * self.sub_loss_weight
            return loss, acc
        else:
            return loss, loss_main.data[0], loss_sub.data[0], acc_main, acc_sub

    def compute_xe_loss_mtl(self, enc_out, ys, x_lens,
                            enc_out_sub, ys_sub, x_lens_sub):
        """Compute XE loss.
        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, encoder_num_units]`
            ys (list): A list of length `[B]`, which contains Variables of size `[L]`
            x_lens (list): A list of length `[B]`
            enc_out_sub (torch.autograd.Variable, float): A tensor of size
                `[B, T_sub, encoder_num_units]`
            ys_sub (list): A list of length `[B]`, which contains Variables of size `[L]`
            x_lens_sub (list): A list of length `[B]`
        Returns:
            loss_main (torch.autograd.Variable, float): A tensor of size `[1]`
            loss_sub (torch.autograd.Variable, float): A tensor of size `[1]`
            acc_main (float): Token-level accuracy in teacher-forcing in the main task
            acc_sub (float): Token-level accuracy in teacher-forcing in the sub task
        """
        sos = Variable(enc_out.data.new(1,).fill_(self.sos_0).long())
        eos = Variable(enc_out.data.new(1,).fill_(self.eos_0).long())
        sos_sub = Variable(enc_out.data.new(1,).fill_(self.sos_1).long())
        eos_sub = Variable(enc_out.data.new(1,).fill_(self.eos_1).long())

        # Append <SOS> and <EOS>
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        ys_in_sub = [torch.cat([sos_sub, y], dim=0) for y in ys_sub]
        ys_out_sub = [torch.cat([y, eos_sub], dim=0) for y in ys_sub]
        y_lens_sub = [y.size(0) for y in ys_in_sub]

        # Convert list to Variable
        ys_in = pad_list(ys_in, self.eos_0)
        ys_out = pad_list(ys_out, -1)
        ys_in_sub = pad_list(ys_in_sub, self.eos_1)
        ys_out_sub = pad_list(ys_out_sub, -1)

        # Teacher-forcing
        logits, aw, logits_sub, aw_sub, aw_dec = self._decode_train(
            enc_out, x_lens, ys_in,
            enc_out_sub, x_lens_sub, ys_in_sub, y_lens_sub)

        # Output smoothing
        if self.logits_temp != 1:
            logits /= self.logits_temp
            logits_sub /= self.logits_temp

        # Compute XE sequence loss in the main task
        if self.main_loss_weight > 0:
            if self.ls_prob > 0:
                # Label smoothing (with uniform distribution)
                y_lens = [y.size(0) + 1 for y in ys]  # Add <EOS>
                loss_main = cross_entropy_label_smoothing(
                    logits, ys=ys_out, y_lens=y_lens,
                    label_smoothing_prob=self.ls_prob,
                    label_smoothing_type=self.ls_type, size_average=True)
            else:
                loss_main = F.cross_entropy(
                    input=logits.view((-1, logits.size(2))),
                    target=ys_out.view(-1),  # Long
                    ignore_index=self.pad_index, size_average=False) / len(enc_out)
            loss_main *= self.main_loss_weight

            # Attention regularization
            if self.att_reg_weight > 0:
                loss_main += F.mse_loss(
                    torch.bmm(aw_dec, aw_sub),
                    Variable(aw.data).cuda(self.device_id),
                    size_average=True, reduce=True) * self.att_reg_weight
        else:
            loss_main = Variable(enc_out.data.new(1,).fill_(0.))

        # Compute token-level accuracy in teacher-forcing in th main task
        pad_pred = logits.data.view(ys_out.size(
            0), ys_out.size(1), logits.size(-1)).max(2)[1]
        mask = ys_out.data != self.pad_index
        numerator = torch.sum(pad_pred.masked_select(
            mask) == ys_out.data.masked_select(mask))
        denominator = torch.sum(mask)
        acc_main = float(numerator) / float(denominator)

        # Compute XE sequence loss in the sub task
        if self.ls_prob > 0:
            # Label smoothing (with uniform distribution)
            y_lens_sub = [y.size(0) + 1 for y in ys_sub]   # Add <EOS>
            loss_sub = cross_entropy_label_smoothing(
                logits_sub, ys=ys_out_sub, y_lens=y_lens_sub,
                label_smoothing_prob=self.ls_prob,
                label_smoothing_type=self.ls_type, size_average=True)
        else:
            loss_sub = F.cross_entropy(
                input=logits_sub.view((-1, logits_sub.size(2))),
                target=ys_out_sub.view(-1),  # long
                ignore_index=self.pad_index, size_average=False) / len(enc_out_sub)
        loss_sub = loss_sub * self.sub_loss_weight

        # Add coverage term
        if self.coverage_weight != 0:
            raise NotImplementedError

        # Compute token-level accuracy in teacher-forcing in th main task
        pad_pred_sub = logits_sub.data.view(ys_out_sub.size(
            0), ys_out_sub.size(1), logits_sub.size(-1)).max(2)[1]
        mask_sub = ys_out_sub.data != self.pad_index
        numerator_sub = torch.sum(pad_pred_sub.masked_select(
            mask_sub) == ys_out_sub.data.masked_select(mask_sub))
        denominator_sub = torch.sum(mask_sub)
        acc_sub = float(numerator_sub) / float(denominator_sub)

        return loss_main, loss_sub, acc_main, acc_sub

    def fix_second_decoder(self):
        self._fix_second_decoder = True

        # dir_sub = 'bwd' if self.backward_1 else 'fwd'

        # Fix parameters in the sub task
        # for param in getattr(self, 'attend_1_' + dir_sub).parameters():
        #     param.requires_grad = False
        #
        # for param in getattr(self, 'decoder_1_' + dir_sub).parameters():
        #     param.requires_grad = False
        #
        # for param in self.embed_1.parameters():
        #     param.requires_grad = False
        #
        # for param in getattr(self, 'fc_1_' + dir_sub).parameters():
        #     param.requires_grad = False
        #
        # for param in getattr(self, 'W_c_1_' + dir_sub).parameters():
        #     param.requires_grad = False
        #
        # for param in getattr(self, 'W_d_1_' + dir_sub).parameters():
        #     param.requires_grad = False
        #
        # if self.logits_injection:
        #     for param in self.W_logits_sub.parameters():
        #         param.requires_grad = False

    def _decode_train(self, enc_out, x_lens, ys,
                      enc_out_sub, x_lens_sub, ys_sub, y_lens_sub):
        """Decoding in the training stage.
        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, encoder_num_units]`
            x_lens (list): A list of length `[B]`
            ys (list): A list of length `[B]`, which contains Variables of size `[L]`
            enc_out_sub (torch.autograd.Variable, float): A tensor of size
                `[B, T_sub, encoder_num_units]`
            x_lens_sub (list): A list of length `[B]`
            ys_sub (list): A list of length `[B]`, which contains Variables of size `[L_sub]`
            y_lens_sub (list):
        Returns:
            logits (torch.autograd.Variable, float): A tensor of size
                `[B, L, num_classes]`
            aw (torch.autograd.Variable, float): A tensor of size
                `[B, L, T]`
            logits_sub (torch.autograd.Variable, float): A tensor of size
                `[B, L_sub, num_classes_sub]`
            aw_sub (torch.autograd.Variable, float): A tensor of size
                `[B, L_sub, T_sub]`
            aw_dec (np.ndarray): A tensor of size
                `[B, L, L_sub]`
        """
        batch_size, max_time = enc_out.size()[:2]
        max_time_sub = enc_out_sub.size(1)
        dir_sub = 'bwd' if self.backward_1 else 'fwd'

        # TODO: add cold fusion

        ##################################################
        # At first, compute logits in the sub task
        ##################################################
        # Pre-computation of embedding
        ys_emb_sub = self.embed_1(ys_sub)

        # Initialization
        dec_out_sub, hx_list_sub, cx_list_sub = self._init_dec_state(
            enc_out_sub, x_lens_sub, task=1, dir=dir_sub)
        aw_step_sub = Variable(enc_out.data.new(
            batch_size, max_time_sub, self.num_heads_1).fill_(0.))
        getattr(self, 'attend_1_' + dir_sub).reset()
        aw_step_sub = None

        dec_outs_sub, logits_sub, aw_sub = [], [], []
        for t in range(ys_sub.size(1)):

            if t > 0:
                # Sample for scheduled sampling
                if self._fix_second_decoder:
                    is_sample = True
                else:
                    is_sample = self.ss_prob > 0 and t > 0 and self._step > 0 and random.random(
                    ) < self._ss_prob * 2
                    # double sampling prob.
                if is_sample:
                    y_sub = self.embed_1(
                        torch.max(logits_sub[-1], dim=2)[1]).detach()
                else:
                    y_sub = ys_emb_sub[:, t:t + 1]

                # Recurrency
                dec_in_sub = torch.cat([y_sub, context_vec_sub], dim=-1)
                dec_out_sub, hx_list_sub, cx_list_sub = getattr(self, 'decoder_1_' + dir_sub)(
                    dec_in_sub, hx_list_sub, cx_list_sub)

            # Score
            context_vec_sub, aw_step_sub = getattr(self, 'attend_1_' + dir_sub)(
                enc_out_sub, x_lens_sub, dec_out_sub, aw_step_sub)

            # Generate
            logits_step_sub = getattr(self, 'fc_1_' + dir_sub)(F.tanh(
                getattr(self, 'W_d_1_' + dir_sub)(dec_out_sub) +
                getattr(self, 'W_c_1_' + dir_sub)(context_vec_sub)))

            dec_outs_sub += [dec_out_sub]
            logits_sub += [logits_step_sub]
            aw_sub = [aw_step_sub] + aw_sub

        dec_outs_sub = torch.cat(dec_outs_sub, dim=1)
        logits_sub = torch.cat(logits_sub, dim=1)
        if self.backward_1:
            aw_sub = aw_sub[::-1]
            # NOTE: reverse for attention regularization
        aw_sub = torch.stack(aw_sub, dim=1)

        if self.main_loss_weight == 0:
            return None, None, logits_sub, aw_sub, None

        if self.logits_injection:
            _logits_sub = self.W_logits_sub(logits_sub)

        ##################################################
        # Next, compute logits in the main task
        ##################################################
        # Pre-computation of embedding
        ys_emb = self.embed_0(ys)

        # Initialization
        dec_out, hx_list, cx_list = self._init_dec_state(
            enc_out, x_lens, task=0, dir='fwd')
        self.attend_0_fwd.reset()
        self.attend_dec_sub.reset()
        aw_step_enc, aw_step_dec = None, None

        logits, aw_enc, aw_dec = [], [], []
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
                    dec_in = torch.cat([dec_in, context_vec_dec], dim=-1)
                dec_out, hx_list, cx_list = self.decoder_0_fwd(
                    dec_in, hx_list, cx_list)

            # Score for the encoder
            context_vec_enc, aw_step_enc = self.attend_0_fwd(
                enc_out, x_lens, dec_out, aw_step_enc)

            # Score for the second decoder states
            if self.logits_injection:
                _, aw_step_dec = self.attend_dec_sub(
                    dec_outs_sub, y_lens_sub, dec_out, aw_step_dec)

                if self.num_heads_dec > 1:
                    context_vec_dec = []
                    for h in range(self.num_heads_dec):
                        # Compute context vector
                        context_vec_dec_head = torch.sum(
                            _logits_sub * aw_step_dec[:, :, h:h + 1], dim=1, keepdim=True)
                        context_vec_dec += [context_vec_dec_head]

                    # Concatenate all convtext vectors and attention distributions
                    context_vec_dec = torch.cat(context_vec_dec, dim=-1)

                    if self.num_heads_dec > 1:
                        context_vec_dec = self.attend_dec_sub.W_mha(
                            context_vec_dec)
                else:
                    # Compute context vector (weighted sum of encoder outputs)
                    context_vec_dec = torch.sum(
                        _logits_sub * aw_step_dec.unsqueeze(2), dim=1, keepdim=True)
            else:
                context_vec_dec, aw_step_dec = self.attend_dec_sub(
                    dec_outs_sub, y_lens_sub, dec_out, aw_step_dec)

            if self.relax_context_vec_dec:
                context_vec_dec = self.W_c_dec_relax(context_vec_dec)

            # Fine-grained gating
            if self.gating:
                gate = F.sigmoid(self.W_gate(
                    torch.cat([dec_out, context_vec_dec], dim=-1)))
                context_vec_dec = gate * context_vec_dec

            # Generate
            out = self.W_d_0_fwd(dec_out) + self.W_c_0_fwd(context_vec_enc)
            if self.usage_dec_sub != 'update_decoder':
                out += self.W_c_dec(context_vec_dec)
            logits_step = self.fc_0_fwd(F.tanh(out))

            logits += [logits_step]
            aw_enc += [aw_step_enc]
            aw_dec += [aw_step_dec]

        logits = torch.cat(logits, dim=1)
        aw_enc = torch.stack(aw_enc, dim=1)
        aw_dec = torch.stack(aw_dec, dim=1)

        # TODO: fix these
        if self.num_heads_0 > 1:
            aw_enc = aw_enc.squeeze(3)
        if self.num_heads_1 > 1:
            aw_sub = aw_sub.squeeze(3)
        if self.num_heads_dec > 1:
            aw_dec = aw_dec.squeeze(3)

        return logits, aw_enc, logits_sub, aw_sub, aw_dec

    def decode(self, xs, beam_width, max_decode_len, min_decode_len=0, min_decode_len_ratio=0,
               length_penalty=0, coverage_penalty=0, rnnlm_weight=0,
               beam_width_sub=1, max_decode_len_sub=None, min_decode_len_sub=0, min_decode_len_ratio_sub=0,
               length_penalty_sub=0, coverage_penalty_sub=0, rnnlm_weight_sub=0,
               task_index=0, teacher_forcing=False, ys_sub=None, exclude_eos=True):
        """Decoding in the inference stage.
        Args:
            xs (np.ndarray): A tensor of size `[B, T, input_size]`
            beam_width (int): the size of beam in the main task
            max_decode_len (int): the maximum sequence length of tokens in the main task
            min_decode_len (int): the minimum sequence length of tokens in the main task
            min_decode_len_ratio (float):
            length_penalty (float): length penalty of the main task
            coverage_penalty (float): coverage penalty of the main task
            rnnlm_weight (float): the weight of RNNLM score of the main task
            beam_width_sub (int): the size of beam in the sub task
            max_decode_len_sub (int): the maximum sequence length of tokens in the sub task
            min_decode_len_sub (int): the minimum sequence length of tokens in the sub task
            min_decode_len_ratio_sub (float):
            length_penalty_sub (float): length penalty of the sub task
            coverage_penalty_sub (float): coverage penalty of the sub task
            rnnlm_weight_sub (float): the weight of RNNLM score of the sub task
            task_index (int): the index of a task
            teacher_forcing (bool):
            ys_sub (list):
            exclude_eos (bool): if True, exclude <EOS> from best_hyps
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw (list): A list of length `[B]`, which contains arrays of size `[L, T]`
            best_hyps_sub (list): A list of length `[B]`, which contains arrays of size `[L_sub]`
            aw_sub (list): A list of length `[B]`, which contains arrays of size `[L_sub, T_sub]`
            aw_dec (list): A list of length `[B]`, which contains arrays of size `[L, L_sub]`
            perm_idx (list): A list of length `[B]`
        """
        self.eval()

        # Sort by lenghts in the descending order
        if self.encoder_type != 'cnn':
            perm_idx = sorted(list(range(0, len(xs), 1)),
                              key=lambda i: len(xs[i]), reverse=True)
            xs = [xs[i] for i in perm_idx]
            # NOTE: must be descending order for pack_padded_sequence
        else:
            perm_idx = list(range(0, len(xs), 1))

        # Encode acoustic features
        if task_index == 0:
            enc_out, x_lens, enc_out_sub, x_lens_sub = self._encode(
                xs, is_multi_task=True)

            # Next, decode by word-based decoder with character outputs
            if teacher_forcing:
                # Reverse the order
                if self.backward_1:
                    for b in range(len(xs)):
                        ys_sub[b] = ys_sub[b][::-1]

                ys_sub = [np2var(np.fromiter(y, dtype=np.int64), self.device_id).long()
                          for y in ys_sub]
                sos_sub = Variable(enc_out.data.new(
                    1,).fill_(self.sos_1).long())
                ys_sub = [ys_sub[i] for i in perm_idx]
                ys_in_sub = [torch.cat([sos_sub, y], dim=0) for y in ys_sub]
                ys_in_sub = pad_list(ys_in_sub, self.eos_1)

                beam_width_sub = 1
            else:
                ys_in_sub = None

            if beam_width == 1 and beam_width_sub == 1:
                best_hyps, aw, best_hyps_sub, aw_sub, aw_dec = self._decode_infer_joint_greedy(
                    enc_out, x_lens, enc_out_sub, x_lens_sub,
                    max_decode_len=max_decode_len,
                    max_decode_len_sub=max_decode_len_sub,
                    teacher_forcing=teacher_forcing,
                    ys_sub=ys_in_sub,
                    reverse_backward=False,
                    exclude_eos=exclude_eos)
            else:
                best_hyps, aw, best_hyps_sub, aw_sub, aw_dec = self._decode_infer_joint_beam(
                    enc_out, x_lens, enc_out_sub, x_lens_sub,
                    beam_width=beam_width,
                    max_decode_len=max_decode_len,
                    min_decode_len=min_decode_len,
                    min_decode_len_ratio=min_decode_len_ratio,
                    rnnlm_weight=rnnlm_weight,
                    length_penalty=length_penalty,
                    coverage_penalty=coverage_penalty,
                    beam_width_sub=beam_width_sub,
                    max_decode_len_sub=max_decode_len_sub,
                    min_decode_len_sub=min_decode_len_sub,
                    min_decode_len_ratio_sub=min_decode_len_ratio_sub,
                    length_penalty_sub=length_penalty_sub,
                    coverage_penalty_sub=coverage_penalty_sub,
                    rnnlm_weight_sub=rnnlm_weight_sub,
                    teacher_forcing=teacher_forcing,
                    ys_sub=ys_in_sub,
                    reverse_backward=False,
                    exclude_eos=exclude_eos)

        elif task_index == 1:
            _, _, enc_out, x_lens = self._encode(xs, is_multi_task=True)

            dir_sub = 'bwd'if self.backward_1 else 'fwd'

            if beam_width == 1:
                best_hyps, aw = self._decode_infer_greedy(
                    enc_out, x_lens, max_decode_len, 1, dir_sub, exclude_eos)
            else:
                best_hyps, aw = self._decode_infer_beam(
                    enc_out, x_lens, beam_width, max_decode_len, min_decode_len,
                    min_decode_len_ratio, length_penalty, coverage_penalty,
                    rnnlm_weight, 1, dir_sub, exclude_eos)
        else:
            raise ValueError

        if task_index == 0:
            return best_hyps, aw, best_hyps_sub, aw_sub, aw_dec, perm_idx
        elif task_index == 1:
            return best_hyps, aw, perm_idx

    def _decode_infer_joint_greedy(self, enc_out, x_lens, enc_out_sub, x_lens_sub,
                                   max_decode_len, max_decode_len_sub,
                                   teacher_forcing, ys_sub, reverse_backward, exclude_eos):
        """Greedy decoding in the inference stage.
        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, encoder_num_units]`
            x_lens (list): A list of length `[B]`
            enc_out_sub (torch.autograd.Variable, float): A tensor of size
                `[B, T_sub, encoder_num_units]`
            x_lens_sub (list): A list of length `[B]`
            max_decode_len (int): the maximum sequence length of tokens in the main task
            max_decode_len_sub (int): the maximum sequence length of tokens in the sub task
            teacher_forcing (bool):
            ys_sub (torch.autograd.Variable):
            reverse_backward (bool):
            exclude_eos (bool): if True, exclude <EOS> from best_hyps
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw_enc (list): A list of length `[B]`, which contains arrays of size `[L, T]`
            best_hyps_sub (list): A list of length `[B]`, which contains arrays of size `[L_sub]`
            aw_sub (list): A list of length `[B]`, which contains arrays of size `[L_sub, T_sub]`
            aw_dec (list): A list of length `[B]`, which contains arrays of size `[L, L_sub]`
        """
        batch_size, max_time = enc_out.size()[:2]
        dir_sub = 'bwd' if self.backward_1 else 'fwd'

        ##################################################
        # At first, decode by the second decoder
        ##################################################
        # Initialization
        dec_out_sub, hx_list_sub, cx_list_sub = self._init_dec_state(
            enc_out_sub, x_lens_sub, task=1, dir=dir_sub)
        getattr(self, 'attend_1_' + dir_sub).reset()
        aw_step_sub = None

        # Start from <SOS>
        y_sub = Variable(enc_out.data.new(
            batch_size, 1).fill_(self.sos_1).long(), volatile=True)

        dec_outs_sub = []
        _best_hyps_sub, _aw_sub = [], []
        logits_sub = []
        y_lens_sub = np.zeros((batch_size,), dtype=np.int32)
        eos_flags_sub = [False] * batch_size
        for t_sub in range(max_decode_len_sub + 1):
            if teacher_forcing:
                y_sub = ys_sub[:, t_sub: t_sub + 1]
            y_sub = self.embed_1(y_sub)

            if t_sub > 0:
                # Recurrency
                dec_in_sub = torch.cat([y_sub, context_vec_sub], dim=-1)
                dec_out_sub, hx_list_sub, cx_list_sub = getattr(self, 'decoder_1_' + dir_sub)(
                    dec_in_sub, hx_list_sub, cx_list_sub)

            # Score
            context_vec_sub, aw_step_sub = getattr(self, 'attend_1_' + dir_sub)(
                enc_out_sub, x_lens_sub, dec_out_sub, aw_step_sub)

            # Generate
            logits_step_sub = getattr(self, 'fc_1_' + dir_sub)(F.tanh(
                getattr(self, 'W_d_1_' + dir_sub)(dec_out_sub) +
                getattr(self, 'W_c_1_' + dir_sub)(context_vec_sub)))

            # Pick up 1-best
            y_sub = torch.max(logits_step_sub.squeeze(1), dim=1)[
                1].unsqueeze(1)
            dec_outs_sub += [dec_out_sub]
            logits_sub += [logits_step_sub]
            _aw_sub += [aw_step_sub]
            _best_hyps_sub += [y_sub]

            for b in range(batch_size):
                if not eos_flags_sub[b]:
                    y_lens_sub[b] += 1
                    # NOTE: include <EOS>
                    if y_sub.data.cpu().numpy()[b] == self.eos_1:
                        eos_flags_sub[b] = True

            # Break if <EOS> is outputed in all mini-batch
            if sum(eos_flags_sub) == batch_size:
                break

            if teacher_forcing and t_sub == ys_sub.size(1) - 1:
                break

        # Concatenate in L dimension
        dec_outs_sub = torch.cat(dec_outs_sub, dim=1)
        logits_sub = torch.cat(logits_sub, dim=1)
        _aw_sub = torch.stack(_aw_sub, dim=1)
        _best_hyps_sub = torch.cat(_best_hyps_sub, dim=1)

        if self.logits_injection:
            logits_sub = self.W_logits_sub(logits_sub)

        ##################################################
        # Next, decode by the main decoder
        ##################################################
        # Initialization
        dec_out, hx_list, cx_list = self._init_dec_state(
            enc_out, x_lens, task=0, dir='fwd')
        self.attend_0_fwd.reset()
        self.attend_dec_sub.reset()
        aw_step_enc, aw_step_dec = None, None

        # Start from <SOS>
        y = Variable(enc_out.data.new(
            batch_size, 1).fill_(self.sos_0).long(), volatile=True)

        _best_hyps, _aw_enc, _aw_dec = [], [], []
        y_lens = np.zeros((batch_size,), dtype=np.int32)
        eos_flags = [False] * batch_size
        for t in range(max_decode_len + 1):
            y = self.embed_0(y)

            if t > 0:
                # Recurrency
                dec_in = torch.cat([y, context_vec_enc], dim=-1)
                if self.usage_dec_sub != 'softmax':
                    dec_in = torch.cat([dec_in, context_vec_dec], dim=-1)
                dec_out, hx_list, cx_list = self.decoder_0_fwd(
                    dec_in, hx_list, cx_list)

            # Score for the encoder
            context_vec_enc, aw_step_enc = self.attend_0_fwd(
                enc_out, x_lens, dec_out, aw_step_enc)

            # Score for the second decoder states
            if self.logits_injection:
                _, aw_step_dec = self.attend_dec_sub(
                    dec_outs_sub, y_lens_sub, dec_out, aw_step_dec)

                if self.num_heads_dec > 1:
                    context_vec_dec = []
                    for h in range(self.num_heads_dec):
                        # Compute context vector
                        context_vec_dec_head = torch.sum(
                            logits_sub * aw_step_dec[:, :, h:h + 1], dim=1, keepdim=True)
                        context_vec_dec += [context_vec_dec_head]

                    # Concatenate all convtext vectors and attention distributions
                    context_vec_dec = torch.cat(context_vec_dec, dim=-1)

                    if self.num_heads_dec > 1:
                        context_vec_dec = self.attend_dec_sub.W_mha(
                            context_vec_dec)
                else:
                    # Compute context vector (weighted sum of encoder outputs)
                    context_vec_dec = torch.sum(
                        logits_sub * aw_step_dec.unsqueeze(2), dim=1, keepdim=True)
            else:
                context_vec_dec, aw_step_dec = self.attend_dec_sub(
                    dec_outs_sub, y_lens_sub, dec_out, aw_step_dec)

            if self.relax_context_vec_dec:
                context_vec_dec = self.W_c_dec_relax(context_vec_dec)

            # Fine-grained gating
            if self.gating:
                gate = F.sigmoid(self.W_gate(
                    torch.cat([dec_out, context_vec_dec], dim=-1)))
                context_vec_dec = gate * context_vec_dec

            # Generate
            out = self.W_d_0_fwd(dec_out) + self.W_c_0_fwd(context_vec_enc)
            if self.usage_dec_sub != 'update_decoder':
                out += self.W_c_dec(context_vec_dec)
            logits_step = self.fc_0_fwd(F.tanh(out))

            # Pick up 1-best
            y = torch.max(logits_step.squeeze(1), dim=1)[1].unsqueeze(1)
            _aw_enc += [aw_step_enc]
            _aw_dec += [aw_step_dec]
            _best_hyps += [y]

            # Count lengths of hypotheses
            for b in range(batch_size):
                if not eos_flags[b]:
                    if y.data.cpu().numpy()[b] == self.eos_0:
                        eos_flags[b] = True
                    y_lens[b] += 1
                    # NOTE: include <EOS>

            # Break if <EOS> is outputed in all mini-batch
            if sum(eos_flags) == batch_size:
                break

        # Concatenate in L dimension
        _aw_enc = torch.stack(_aw_enc, dim=1)
        _aw_dec = torch.stack(_aw_dec, dim=1)
        _best_hyps = torch.cat(_best_hyps, dim=1)

        # Convert to numpy
        _aw_enc = var2np(_aw_enc)
        _aw_sub = var2np(_aw_sub)
        _aw_dec = var2np(_aw_dec)
        _best_hyps = var2np(_best_hyps)
        _best_hyps_sub = var2np(_best_hyps_sub)

        # TODO: fix for MHA
        if self.num_heads_0 > 1:
            _aw_enc = _aw_enc[:, :, :, 0]
        if self.num_heads_1 > 1:
            _aw_sub = _aw_sub[:, :, :, 0]
        if self.num_heads_dec > 1:
            _aw_dec = _aw_dec[:, :, :, 0]

        # Truncate by the first <EOS>
        best_hyps = [_best_hyps[b, :y_lens[b]] for b in range(batch_size)]
        aw_enc = [_aw_enc[b, :y_lens[b]] for b in range(batch_size)]
        aw_dec = [_aw_dec[b, : y_lens[b]] for b in range(batch_size)]
        if self.backward_1 and reverse_backward:
            # Reverse the order
            best_hyps_sub = [_best_hyps_sub[b, : y_lens_sub[b]][:: -1]
                             for b in range(batch_size)]
            aw_sub = [_aw_sub[b, : y_lens_sub[b]][:: -1]
                      for b in range(batch_size)]
        else:
            best_hyps_sub = [_best_hyps_sub[b, : y_lens_sub[b]]
                             for b in range(batch_size)]
            aw_sub = [_aw_sub[b, : y_lens_sub[b]] for b in range(batch_size)]

        return best_hyps, aw_enc, best_hyps_sub, aw_sub, aw_dec

    def _decode_infer_joint_beam(self, enc_out, x_lens, enc_out_sub, x_lens_sub,
                                 beam_width, max_decode_len, min_decode_len, min_decode_len_ratio,
                                 length_penalty, coverage_penalty, rnnlm_weight,
                                 beam_width_sub, max_decode_len_sub, min_decode_len_sub, min_decode_len_ratio_sub,
                                 length_penalty_sub, coverage_penalty_sub, rnnlm_weight_sub,
                                 teacher_forcing, ys_sub, reverse_backward, exclude_eos):
        """Beam search decoding in the inference stage.
        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, encoder_num_units]`
            x_lens (list): A list of length `[B]`
            enc_out_sub (torch.autograd.Variable, float): A tensor of size
                `[B, T_sub, encoder_num_units]`
            x_lens_sub (list): A list of length `[B]`
            beam_width (int): the size of beam in the main task
            max_decode_len (int): the maximum sequence length of tokens in the main task
            min_decode_len (int): the minimum sequence length of tokens in the main task
            min_decode_len_ratio (float):
            length_penalty (float): length penalty of the main task
            coverage_penalty (float): coverage penalty of the main task
            rnnlm_weight (float): the weight of RNNLM score of the main task
            beam_width_sub (int): the size of beam in the sub task
            max_decode_len_sub (int): the maximum sequence length of tokens in the sub task
            min_decode_len_sub (int): the minimum sequence length of tokens in the sub task
            min_decode_len_ratio_sub (float):
            length_penalty_sub (float): length penalty of the sub task
            coverage_penalty_sub (float): coverage penalty of the sub task
            rnnlm_weight_sub (float): the weight of RNNLM score of the sub task
            teacher_forcing (bool):
            ys_sub (list):
            reverse_backward (bool):
            exclude_eos (bool): if True, exclude <EOS> from best_hyps
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw_enc (list): A list of length `[B]`, which contains arrays of size `[L, T]`
            best_hyps_sub (list): A list of length `[B]`, which contains arrays of size `[L_sub]`
            aw_sub (list): A list of length `[B]`, which contains arrays of size `[L_sub, T_sub]`
            aw_dec (list): A list of length `[B]`, which contains arrays of size `[L, L_sub]`
        """
        batch_size, max_time = enc_out.size()[: 2]
        dir_sub = 'bwd' if self.backward_1 else 'fwd'
        if teacher_forcing:
            beam_width_sub = 1

        ##################################################
        # At first, decode by the second decoder
        ##################################################
        best_hyps, aw_enc, aw_dec = [], [], []
        best_hyps_sub, aw_sub = [], []
        eos_flags = [False] * batch_size
        eos_flags_sub = [False] * batch_size
        for b in range(batch_size):
            # Initialization per utterance
            dec_out_sub, hx_list_sub, cx_list_sub = self._init_dec_state(
                enc_out_sub[b: b + 1], x_lens_sub[b], task=1, dir=dir_sub)
            getattr(self, 'attend_1_' + dir_sub).reset()

            complete_sub = []
            beam_sub = [{'hyp': [self.sos_1],
                         'score': 0,  # log 1
                         'dec_outs': [dec_out_sub],  # NOTE: keep all outputs
                         'hx_list': hx_list_sub,
                         'cx_list': cx_list_sub,
                         'context_vec': None,
                         'aw_steps': [None],
                         'rnnlm_state': None,
                         'previous_coverage': 0,
                         'logits': []}]
            for t_sub in range(max_decode_len_sub + 1):
                new_beam_sub = []
                for i_beam in range(len(beam_sub)):
                    if teacher_forcing:
                        y_sub = ys_sub[:, t_sub: t_sub + 1]
                    else:
                        y_sub = Variable(enc_out.data.new(
                            1, 1).fill_(beam_sub[i_beam]['hyp'][-1]).long(), volatile=True)
                    y_sub = self.embed_1(y_sub)

                    if t_sub == 0:
                        dec_out_sub = beam_sub[i_beam]['dec_outs'][-1]
                    else:
                        # Recurrency
                        dec_in_sub = torch.cat(
                            [y_sub, beam_sub[i_beam]['context_vec']], dim=-1)
                        dec_out_sub, hx_list_sub, cx_list_sub = getattr(
                            self, 'decoder_1_' + dir_sub)(dec_in_sub, beam_sub[i_beam]['hx_list'], beam_sub[i_beam]['cx_list'])

                    # Score
                    context_vec_sub, aw_step_sub = getattr(self, 'attend_1_' + dir_sub)(
                        enc_out_sub[b: b + 1, : x_lens_sub[b]],
                        x_lens_sub[b: b + 1],
                        dec_out_sub, beam_sub[i_beam]['aw_steps'][-1])

                    # Generate
                    logits_step_sub = getattr(self, 'fc_1_' + dir_sub)(F.tanh(
                        getattr(self, 'W_d_1_' + dir_sub)(dec_out_sub) +
                        getattr(self, 'W_c_1_' + dir_sub)(context_vec_sub)))

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
                        if indices_sub_topk[0, k].data[0] == self.eos_1 and len(beam_sub[i_beam]['hyp']) < x_lens_sub[b] * min_decode_len_ratio_sub:
                            continue

                        # Add length penalty
                        score_sub = beam_sub[i_beam]['score'] + \
                            log_probs_sub_topk.data[0, k] + length_penalty_sub

                        # Add coverage penalty
                        if coverage_penalty_sub > 0:
                            # Recompute converage penalty in each step
                            score_sub -= beam_sub[i_beam]['previous_coverage'] * \
                                coverage_penalty_sub

                            cov_threshold = 0.5
                            # TODO: make this parameter
                            aw_steps_sub = torch.stack(
                                beam_sub[i_beam]['aw_steps'] + [aw_step_sub], dim=1)

                            if self.num_heads_1 > 1:
                                cov_sum_sub = aw_steps_sub.data[0,
                                                                :, :, 0].cpu().numpy()
                                # TODO: fix for MHA
                            else:
                                cov_sum_sub = aw_steps_sub.data[0].cpu(
                                ).numpy()

                            cov_sum_sub = np.sum(cov_sum_sub[np.where(
                                cov_sum_sub > cov_threshold)[0]])
                            score_sub += cov_sum_sub * coverage_penalty_sub
                        else:
                            cov_sum_sub = 0

                        # Add RNNLM score
                        if rnnlm_weight_sub > 0 and self.rnnlm_1_fwd is not None:
                            y_sub_rnnlm = Variable(enc_out.data.new(
                                1, 1).fill_(beam_sub[i_beam]['hyp'][-1]).long(), volatile=True)
                            y_sub_rnnlm = self.rnnlm_1_fwd.embed(y_sub_rnnlm)
                            rnnlm_logits_step_sub, rnnlm_out_sub, rnnlm_state_sub = self.rnnlm_1_fwd.predict(
                                y_sub_rnnlm, h=beam_sub[i_beam]['rnnlm_state'])
                            rnnlm_log_probs_sub = F.log_softmax(
                                rnnlm_logits_step_sub.squeeze(1), dim=1)
                            assert log_probs_sub.size() == rnnlm_log_probs_sub.size()
                            score_sub += rnnlm_log_probs_sub.data[0,
                                                                  indices_sub_topk.data[0, k]] * rnnlm_weight_sub
                        else:
                            rnnlm_state_sub = None

                        new_beam_sub.append(
                            {'hyp': beam_sub[i_beam]['hyp'] + [indices_sub_topk[0, k].data[0]],
                             'score': score_sub,
                             'dec_outs': beam_sub[i_beam]['dec_outs'] + [dec_out_sub],
                             'hx_list': copy.deepcopy(hx_list_sub),
                             'cx_list': copy.deepcopy(cx_list_sub),
                             'context_vec': context_vec_sub,
                             'aw_steps': beam_sub[i_beam]['aw_steps'] + [aw_step_sub],
                             'rnnlm_state': rnnlm_state_sub,
                             'previous_coverage': cov_sum_sub,
                             'logits': beam_sub[i_beam]['logits'] + [logits_step_sub]})

                if teacher_forcing and t_sub == ys_sub.size(1) - 1:
                    break

                new_beam_sub = sorted(
                    new_beam_sub, key=lambda x: x['score'], reverse=True)

                # Remove complete hypotheses
                not_complete_sub = []
                for cand in new_beam_sub[: beam_width_sub]:
                    if cand['hyp'][-1] == self.eos_1:
                        complete_sub += [cand]
                    else:
                        not_complete_sub += [cand]

                if len(complete_sub) >= beam_width_sub:
                    complete_sub = complete_sub[: beam_width_sub]
                    break

                beam_sub = not_complete_sub[: beam_width_sub]

            if len(complete_sub) == 0:
                complete_sub = beam_sub

            # Renormalized hypotheses by length
            if length_penalty > 0:
                for j in range(len(complete_sub)):
                    complete_sub[j]['score'] += len(complete_sub[j]
                                                    ['hyp']) * length_penalty

            complete_sub = sorted(
                complete_sub, key=lambda x: x['score'], reverse=True)
            best_hyps_sub += [np.array(complete_sub[0]['hyp'][1:])]
            dec_outs_sub = torch.cat(complete_sub[0]
                                     ['dec_outs'][1:], dim=1)
            aw_sub += [complete_sub[0]['aw_steps'][1:]]
            logits_sub = torch.cat(complete_sub[0]['logits'], dim=1)
            y_len_sub = len(complete_sub[0]['hyp'][1:])
            if complete_sub[0]['hyp'][-1] == self.eos_1:
                eos_flags_sub[b] = True

            ##################################################
            # Next, decode by the first decoder
            ##################################################
            # Initialization per utterance
            dec_out, hx_list, cx_list = self._init_dec_state(
                enc_out[b: b + 1], x_lens[b], task=0, dir='fwd')
            self.attend_0_fwd.reset()
            self.attend_dec_sub.reset()

            complete = []
            beam = [{'hyp': [self.sos_0],
                     'score': 0,  # log 1
                     'dec_out': dec_out,
                     'hx_list': hx_list,
                     'cx_list': cx_list,
                     'context_vec_enc': None,
                     'context_vec_dec': None,
                     'aw_steps_enc': [None],
                     'aw_steps_dec': [None],
                     'rnnlm_state': None,
                     'previous_coverage': 0}]
            for t in range(max_decode_len + 1):
                new_beam = []
                for i_beam in range(len(beam)):
                    y = Variable(enc_out.data.new(
                        1, 1).fill_(beam[i_beam]['hyp'][-1]).long(), volatile=True)
                    y = self.embed_0(y)

                    if t == 0:
                        dec_out = beam[i_beam]['dec_out']
                    else:
                        # Recurrency
                        dec_in = torch.cat(
                            [y, beam[i_beam]['context_vec_enc']], dim=-1)
                        if self.usage_dec_sub != 'softmax':
                            dec_in = torch.cat(
                                [dec_in, beam[i_beam]['context_vec_dec']], dim=-1)
                        dec_out, hx_list, cx_list = self.decoder_0_fwd(
                            dec_in, beam[i_beam]['hx_list'], beam[i_beam]['cx_list'])

                    # Score for the encoder
                    context_vec_enc, aw_step_enc = self.attend_0_fwd(
                        enc_out[b: b + 1, : x_lens[b]], x_lens[b: b + 1],
                        dec_out, beam[i_beam]['aw_steps_enc'][-1])

                    # Score for the second decoder states
                    if self.logits_injection:
                        _, aw_step_dec = self.attend_dec_sub(
                            dec_outs_sub, [y_len_sub],
                            dec_out, beam[i_beam]['aw_steps_dec'][-1])

                        if self.num_heads_dec > 1:
                            context_vec_dec = []
                            for h in range(self.num_heads_dec):
                                # Compute context vector
                                context_vec_dec_head = torch.sum(
                                    logits_sub * aw_step_dec[:, :, h: h + 1], dim=1, keepdim=True)
                                context_vec_dec += [context_vec_dec_head]

                            # Concatenate all convtext vectors and attention distributions
                            context_vec_dec = torch.cat(
                                context_vec_dec, dim=-1)

                            if self.num_heads_dec > 1:
                                context_vec_dec = self.attend_dec_sub.W_mha(
                                    context_vec_dec)
                        else:
                            # Compute context vector (weighted sum of encoder outputs)
                            context_vec_dec = torch.sum(
                                logits_sub * aw_step_dec.unsqueeze(2), dim=1, keepdim=True)
                    else:
                        context_vec_dec, aw_step_dec = self.attend_dec_sub(
                            dec_outs_sub, [y_len_sub],
                            dec_out, beam[i_beam]['aw_steps_dec'][-1])

                    if self.relax_context_vec_dec:
                        context_vec_dec = self.W_c_dec_relax(context_vec_dec)

                    # Fine-grained gating
                    if self.gating:
                        gate = F.sigmoid(self.W_gate(
                            torch.cat([dec_out, context_vec_dec], dim=-1)))
                        context_vec_dec = gate * context_vec_dec

                    # Generate
                    out = self.W_d_0_fwd(dec_out) + \
                        self.W_c_0_fwd(context_vec_enc)
                    if self.usage_dec_sub != 'update_decoder':
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
                                beam[i_beam]['aw_steps'] + [aw_step_enc], dim=1)

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
                            y_rnnlm = Variable(enc_out.data.new(
                                1, 1).fill_(beam[i_beam]['hyp'][-1]).long(), volatile=True)
                            y_rnnlm = self.rnnlm_0_fwd.embed(y_rnnlm)
                            rnnlm_logits_step, rnnlm_out, rnnlm_state = self.rnnlm_0_fwd.predict(
                                y_rnnlm, h=beam[i_beam]['rnnlm_state'])
                            rnnlm_log_probs = F.log_softmax(
                                rnnlm_logits_step.squeeze(1), dim=1)
                            assert log_probs.size() == rnnlm_log_probs.size()
                            score += rnnlm_log_probs.data[0,
                                                          indices_topk.data[0, k]] * rnnlm_weight
                        else:
                            rnnlm_state = None

                        new_beam.append(
                            {'hyp': beam[i_beam]['hyp'] + [indices_topk[0, k].data[0]],
                             'score': score,
                             'dec_out': dec_out,
                             'hx_list': copy.deepcopy(hx_list),
                             'cx_list': copy.deepcopy(cx_list),
                             'context_vec_enc': context_vec_enc,
                             'context_vec_dec': context_vec_dec,
                             'aw_steps_enc': beam[i_beam]['aw_steps_enc'] + [aw_step_enc],
                             'aw_steps_dec': beam[i_beam]['aw_steps_dec'] + [aw_step_dec],
                             'rnnlm_state': rnnlm_state,
                             'previous_coverage': cov_sum})

                new_beam = sorted(
                    new_beam, key=lambda x: x['score'], reverse=True)

                # Remove complete hypotheses
                not_complete = []
                for cand in new_beam[: beam_width]:
                    if cand['hyp'][-1] == self.eos_0:
                        complete += [cand]
                    else:
                        not_complete += [cand]

                if len(complete) >= beam_width:
                    complete = complete[: beam_width]
                    break

                beam = not_complete[: beam_width]

            if len(complete) == 0:
                complete = beam

            # Renormalized hypotheses by length
            if length_penalty > 0:
                for j in range(len(complete)):
                    complete[j]['score'] += len(complete[j]
                                                ['hyp']) * length_penalty

            complete = sorted(
                complete, key=lambda x: x['score'], reverse=True)
            best_hyps += [np.array(complete[0]['hyp'][1:])]
            aw_enc += [complete[0]['aw_steps_enc'][1:]]
            aw_dec += [complete[0]['aw_steps_dec'][1:]]
            if complete[0]['hyp'][-1] == self.eos_0:
                eos_flags[b] = True

        # Concatenate in L dimension
        for b in range(len(aw_enc)):
            aw_sub[b] = var2np(torch.stack(aw_sub[b], dim=1).squeeze(0))
            if self.num_heads_1 > 1:
                aw_sub[b] = aw_sub[b][:, :, 0]
                # TODO: fix for MHA

            aw_enc[b] = var2np(torch.stack(aw_enc[b], dim=1).squeeze(0))
            aw_dec[b] = var2np(torch.stack(aw_dec[b], dim=1).squeeze(0))
            if self.num_heads_0 > 1:
                aw_enc[b] = aw_enc[b][:, :, 0]
                # TODO: fix for MHA
            if self.num_heads_dec > 1:
                aw_dec[b] = aw_dec[b][:, :, 0]
                # TODO: fix for MHA

        # Reverse the order
        if self.backward_1 and reverse_backward:
            best_hyps_sub = [best_hyps_sub[b][:: -1]
                             for b in range(batch_size)]
            aw_sub = [aw_sub[b][:: -1] for b in range(batch_size)]

        # Exclude <EOS>
        if exclude_eos:
            best_hyps = [best_hyps[b][:-1] if eos_flags[b]
                         else best_hyps[b] for b in range(batch_size)]
            best_hyps_sub = [best_hyps_sub[b][:-1] if eos_flags_sub[b]
                             else best_hyps_sub[b] for b in range(batch_size)]

        return best_hyps, aw_enc, best_hyps_sub, aw_sub, aw_dec
