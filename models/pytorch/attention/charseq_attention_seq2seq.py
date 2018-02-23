#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Hierarchical attention + character sequence attention (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np

import torch
import torch.nn.functional as F

from models.pytorch.linear import LinearND, Embedding, Embedding_LS
from models.pytorch.criterion import cross_entropy_label_smoothing
from models.pytorch.attention.attention_seq2seq import AttentionSeq2seq
from models.pytorch.encoders.load_encoder import load
# from models.pytorch.attention.rnn_decoder import RNNDecoder
from models.pytorch.attention.rnn_decoder_nstep import RNNDecoder
from models.pytorch.attention.attention_layer import AttentionMechanism
from models.pytorch.ctc.decoders.greedy_decoder import GreedyDecoder
from models.pytorch.ctc.decoders.beam_search_decoder import BeamSearchDecoder

LOG_1 = 0


class CharseqAttentionSeq2seq(AttentionSeq2seq):

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
                 curriculum_training=False,  # ***
                 composition_case='fine_grained_gating'):  # ***

        super(CharseqAttentionSeq2seq, self).__init__(
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
        self.model_type = 'charseq_attention'

        # Setting for the encoder
        self.encoder_num_layers_sub = encoder_num_layers_sub

        # Setting for the decoder
        self.decoder_num_units_sub = decoder_num_units_sub
        self.decoder_num_layers_sub = decoder_num_layers_sub
        self.embedding_dim_sub = embedding_dim_sub
        self.num_classes_sub = num_classes_sub + 2  # Add <EOS> class
        self.sos_index_sub = num_classes_sub
        self.eos_index_sub = num_classes_sub

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

        if self.sub_loss_weight > 0 and self.sub_loss_weight >= self.ctc_loss_weight:
            self.sub_decoder = 'attention'
        elif self.ctc_loss_weight_sub > 0 and self.ctc_loss_weight > self.sub_loss_weight:
            self.sub_decoder = 'ctc'
        else:
            raise ValueError

        assert composition_case in [
            'fine_grained_gating', 'scalar_gating', 'concat']
        self.composition_case = composition_case

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

        ###################################
        # Encoder (character sequence)
        ###################################
        self.encoder_charseq = load(encoder_type='lstm')(
            input_size=embedding_dim_sub,
            rnn_type='lstm',
            bidirectional=True,
            num_units=decoder_num_units,
            num_proj=0,
            num_layers=1,
            dropout_input=0,
            dropout_hidden=dropout_encoder,
            batch_first=True,
            merge_bidirectional=False,
            pack_sequence=True)
        # TODO: try CNN
        self.char2word = LinearND(decoder_num_units * 2, embedding_dim)

        ########################################
        # Attention layer (character sequence)
        ########################################
        self.attend_charseq = AttentionMechanism(
            encoder_num_units=embedding_dim,
            decoder_num_units=decoder_num_units,
            # attention_type='location',
            attention_type='content',
            # attention_type='dot_product',
            attention_dim=attention_dim,
            sharpening_factor=sharpening_factor,
            sigmoid_smoothing=sigmoid_smoothing,
            out_channels=attention_conv_num_channels,
            kernel_size=attention_conv_width)
        # TODO: fix bugs in location

        if composition_case == 'fine_grained_gating':
            self.gate_fn = LinearND(embedding_dim, embedding_dim)
        elif composition_case == 'scalar_gating':
            self.gate_fn = LinearND(embedding_dim, 1)

        ####################
        # Decoder
        ####################
        decoder_input_size = embedding_dim + self.encoder_num_units
        if composition_case == 'concat':
            decoder_input_size += embedding_dim
        self.decoder = RNNDecoder(
            input_size=decoder_input_size,
            rnn_type=decoder_type,
            num_units=decoder_num_units,
            num_layers=decoder_num_layers,
            dropout=dropout_decoder,
            residual=decoder_residual,
            dense_residual=decoder_dense_residual)

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

        self.is_bridge_sub = False
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

            #################################################################
            # Bridge layer between the encoder and decoder (sub)
            #################################################################
            if encoder_num_units != decoder_num_units_sub and attention_type == 'dot_product':
                self.bridge_sub = LinearND(
                    self.encoder_num_units, decoder_num_units_sub,
                    dropout=dropout_encoder)
                self.is_bridge_sub = True

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
            if self.is_bridge_sub:
                self.fc_ctc_sub = LinearND(
                    decoder_num_units_sub, num_classes_sub + 1)
            else:
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
        # NOTE: ys and ys_sub are padded with -1 here
        # ys_in and ys_sub_in areb padded with <EOS> in order to convert to
        # one-hot vector, and added <SOS> before the first token
        # ys_out and ys_sub_out are padded with -1, and added <EOS>
        # after the last token
        ys_in = self._create_var((ys.shape[0], ys.shape[1] + 1),
                                 fill_value=self.eos_index, dtype='long')
        ys_sub_in = self._create_var((ys_sub.shape[0], ys_sub.shape[1] + 1),
                                     fill_value=self.eos_index_sub, dtype='long')
        ys_out = self._create_var((ys.shape[0], ys.shape[1] + 1),
                                  fill_value=-1, dtype='long')
        ys_sub_out = self._create_var((ys_sub.shape[0], ys_sub.shape[1] + 1),
                                      fill_value=-1, dtype='long')
        for b in range(len(xs)):
            ys_in.data[b, 0] = self.sos_index
            ys_in.data[b, 1:y_lens[b] + 1] = torch.from_numpy(
                ys[b, :y_lens[b]])
            ys_sub_in.data[b, 0] = self.sos_index_sub
            ys_sub_in.data[b, 1:y_lens_sub[b] + 1] = torch.from_numpy(
                ys_sub[b, :y_lens_sub[b]])

            ys_out.data[b, :y_lens[b]] = torch.from_numpy(ys[b, :y_lens[b]])
            ys_out.data[b, y_lens[b]] = self.eos_index
            ys_sub_out.data[b, :y_lens_sub[b]] = torch.from_numpy(
                ys_sub[b, :y_lens_sub[b]])
            ys_sub_out.data[b, y_lens_sub[b]] = self.eos_index_sub

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

        if is_eval:
            self.eval()
        else:
            self.train()

            # Gaussian noise injection
            if self.weight_noise_injection:
                self.inject_weight_noise(mean=0, std=self.weight_noise_std)

        # Encode acoustic features
        xs, x_lens, xs_sub, x_lens_sub, perm_idx = self._encode(
            xs, x_lens, volatile=is_eval, is_multi_task=True)

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
            xs, ys_in, ys_out, x_lens, y_lens,
            ys_sub_in=ys_sub_in, y_lens_sub=y_lens_sub,
            size_average=True)

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

    def compute_xe_loss(self, enc_out, ys_in, ys_out, x_lens, y_lens,
                        ys_sub_in=None, y_lens_sub=None,
                        is_sub_task=False, size_average=False):
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
            ys_sub_in (torch.autograd.Variable, long): A tensor of size
                `[B, T_out_sub]`, which includes <SOS>
            y_lens_sub (torch.autograd.Variable, int): A tensor of size `[B]`
            is_sub_task (bool, optional):
            size_average (bool, optional):
        Returns:
            loss (torch.autograd.Variable, float): A tensor of size `[1]`
        """
        # Teacher-forcing
        logits, att_weights = self._decode_train(
            enc_out, x_lens, ys_in,
            ys_sub=ys_sub_in, y_lens_sub=y_lens_sub,
            is_sub_task=is_sub_task)

        # Output smoothing
        if self.logits_temperature != 1:
            logits /= self.logits_temperature

        # Compute XE sequence loss
        loss = F.cross_entropy(
            input=logits.view((-1, logits.size(2))),
            target=ys_out.view(-1),
            ignore_index=-1, size_average=False) / len(enc_out)

        # Label smoothing (with uniform distribution)
        if self.label_smoothing_prob > 0:
            loss_ls = cross_entropy_label_smoothing(
                logits,
                y_lens=y_lens + 1,  # Add <EOS>
                label_smoothing_prob=self.label_smoothing_prob,
                distribution='uniform',
                size_average=True)
            loss = loss * (1 - self.label_smoothing_prob) + loss_ls
            # print(loss_ls)

        # Add coverage term
        if self.coverage_weight != 0:
            raise NotImplementedError

        return loss

    def _decode_train(self, enc_out, x_lens, ys,
                      ys_sub=None, y_lens_sub=None, is_sub_task=False):
        """Decoding in the training stage.
        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T_in, encoder_num_units]`
            x_lens (torch.autograd.Variable, int): A tensor of size `[B]`
            ys (torch.autograd.Variable, long): A tensor of size `[B, T_out]`
            ys_sub (torch.autograd.Variable, long): A tensor of size `[B, T_out_sub]`
            y_lens_sub (torch.auto.Variable, int, optional): A tensor of size `[B]`
            is_sub_task (bool, optional):
        Returns:
            logits (torch.autograd.Variable, float): A tensor of size
                `[B, T_out, num_classes]`
            att_weights (torch.autograd.Variable, float): A tensor of size
                `[B, T_out, T_in]`
        """
        batch_size, max_time = enc_out.size()[:2]

        # Initialize decoder state, decoder output, attention_weights
        dec_state = self._init_decoder_state(enc_out)
        dec_out = self._create_var((batch_size, 1, self.decoder_num_units))
        context_vec = self._create_var((batch_size, 1, self.encoder_num_units))
        att_weights_step = self._create_var((batch_size, max_time))
        if not is_sub_task:
            # Path through character embedding
            chars_emb = []
            for c in range(ys_sub.size(1)):
                chars_emb.append(self.embed_sub(ys_sub[:, c:c + 1]))
            chars_emb = torch.cat(chars_emb, dim=1)
            # chars_emb: `[B, T_out, embedding_dim_sub]`

            char_mask = self._create_var(chars_emb.size())
            for b in range(batch_size):
                char_mask.data[:, :y_lens_sub[b].data[0]] = 1
            chars_emb *= char_mask

            # Encode characters
            char_enc_out, _, _ = self.encoder_charseq(
                chars_emb, y_lens_sub, volatile=False)
            char_repr = self.char2word(char_enc_out)

            # Initialize attention weights for characters
            char_att_weights_step = self._create_var(tuple(ys_sub.size()))

        logits = []
        att_weights = []
        for t in range(ys.size(1)):

            is_sample = self.sample_prob > 0 and t > 0 and self._step > 0 and random.random(
            ) < self._sample_prob

            if is_sample:
                # scheduled sampling
                y = torch.max(logits[-1], dim=2)[1]
            else:
                # teacher-forcing
                y = ys[:, t:t + 1]

            if is_sub_task:
                y = self.embed_sub(y)
            else:
                y = self.embed(y)

            if not is_sub_task:
                # Compute attention weights and context vector for characters
                char_context_vec, char_att_weights_step = self.attend_charseq(
                    char_repr, y_lens_sub, dec_out, char_att_weights_step)
                # NOTE: char_context_vec is a word representation

                # Compose with word embedding
                if self.composition_case in ['fine_grained_gating', 'scalar_gating']:
                    gate = F.sigmoid(self.gate_fn(y))
                    y = gate * y + (1 - gate) * char_context_vec
                    # char_context_vec: `[B, 1, word_embedding_dim]`
                    # y: `[B, 1, word_embedding_dim]`
                elif self.composition_case == 'concat':
                    y = torch.cat([y, char_context_vec], dim=-1)

                # TODO: char_context_vecを出力に使う
                # TODO: 前の出力単語とこれをcomposite

            dec_out, dec_state, context_vec, att_weights_step = self._decode_step(
                enc_out=enc_out,
                x_lens=x_lens,
                y=y,
                dec_state=dec_state,
                dec_out=dec_out,
                context_vec=context_vec,
                att_weights_step=att_weights_step,
                is_sub_task=is_sub_task)

            if is_sub_task:
                logits_step = self.fc_sub(F.tanh(
                    self.W_d_sub(dec_out) + self.W_c_sub(context_vec)))
            else:
                logits_step = self.fc(F.tanh(
                    self.W_d(dec_out) + self.W_c(context_vec)))

            logits.append(logits_step)
            att_weights.append(att_weights_step)

        # Concatenate in T_out-dimension
        logits = torch.cat(logits, dim=1)
        att_weights = torch.stack(att_weights, dim=1)
        # NOTE; att_weights in the training stage may be used for computing the
        # coverage, so do not convert to numpy yet.

        return logits, att_weights

    def attention_weights(self, xs, x_lens, max_decode_len, max_decode_len_sub,
                          beam_width=1):
        """Get attention weights for visualization.
        Args:
            xs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            x_lens (np.ndarray): A tensor of size `[B]`
            max_decode_len (int): the length of output sequences
                to stop prediction when EOS token have not been emitted

            is_sub_task (bool, optional):
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B, T_out]`
            best_hyps_sub (np.ndarray): A tensor of size `[B, T_out_sub]`
            att_weights (np.ndarray): A tensor of size `[B, T_out, T_in]`
            char_att_weights (np.ndarray): A tensor of size
                `[B, T_out, T_out_sub]`
        """
        # Wrap by Variable
        xs = self.np2var(xs, volatile=True)
        x_lens = self.np2var(x_lens, dtype='int', volatile=True)

        # Encode acoustic features
        enc_out, x_lens, enc_out_sub, x_lens_sub, perm_idx = self._encode(
            xs, x_lens, volatile=True, is_multi_task=True)

        # Change to evaluation mode
        self.eval()

        # At first, decode by character-based decoder
        if self.ctc_loss_weight_sub > self.sub_loss_weight:
            # Decode by CTC decoder

            # Path through the softmax layer
            batch_size, max_time_sub = enc_out_sub.size()[:2]
            enc_out_sub = enc_out_sub.view(
                batch_size * max_time_sub, -1).contiguous()
            logits_ctc_sub = self.fc_ctc_sub(enc_out_sub)
            logits_ctc_sub = logits_ctc_sub.view(batch_size, max_time_sub, -1)

            if beam_width == 1:
                best_hyps_sub = self._decode_ctc_greedy_np(
                    self.var2np(logits_ctc_sub),
                    self.var2np(x_lens_sub))
            else:
                best_hyps_sub = self._decode_ctc_beam_np(
                    self.var2np(F.log_softmax(logits_ctc_sub, dim=-1),
                                ),
                    self.var2np(x_lens_sub), beam_width=beam_width)

            # NOTE: index 0 is reserved for blank in warpctc_pytorch
            best_hyps_sub -= 1
        else:
            # Decode by attention decoder
            if beam_width == 1:
                best_hyps_sub, _ = self._decode_infer_greedy(
                    enc_out_sub, x_lens_sub, max_decode_len, is_sub_task=True)
            else:
                best_hyps_sub, _ = self._decode_infer_beam(
                    enc_out_sub, x_lens_sub, max_decode_len, is_sub_task=True)

        y_lens_sub = np2var(
            np.array([len(y) for y in best_hyps_sub]),
            dtype='int', use_cuda=self.use_cuda, volatile=True)
        # assert max(y_lens_sub.data) > 0

        ys_sub = self._create_var(
            (len(xs), max(y_lens_sub.data)), dtype='long')
        for b in range(len(xs)):
            ys_sub.data[b, :len(best_hyps_sub[b])] = torch.from_numpy(
                best_hyps_sub[b])

        # Next, decode by word-based decoder with character outputs
        best_hyps, att_weights, char_att_weights = self._decode_infer_greedy(
            enc_out, x_lens,
            max_decode_len=max_decode_len,
            ys_sub=ys_sub,
            y_lens_sub=y_lens_sub)

        # Permutate indices to the original order
        if perm_idx is None:
            perm_idx = np.arange(0, len(xs), 1)
        else:
            perm_idx = self.var2np(perm_idx)

        return best_hyps, best_hyps_sub, att_weights, char_att_weights

    def decode(self, xs, x_lens, beam_width, max_decode_len,
               max_decode_len_sub=None, is_sub_task=False):
        """Decoding in the inference stage.
        Args:
            xs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            x_lens (np.ndarray): A tensor of size `[B]`
            beam_width (int): the size of beam
            max_decode_len (int): the length of output sequences
                to stop prediction when EOS token have not been emitted
            max_decode_len_sub (int, optional):
            is_sub_task (bool, optional):
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B]`
            best_hyps_sub (np.ndarray): A tensor of size `[B]`
            perm_idx (np.ndarray): A tensor of size `[B]`
        """
        # Wrap by Variable
        xs = self.np2var(xs, volatile=True)
        x_lens = self.np2var(x_lens, dtype='int', volatile=True)

        # Encode acoustic features
        if is_sub_task:
            _, _, enc_out_sub, x_lens_sub, perm_idx = self._encode(
                xs, x_lens, volatile=True, is_multi_task=True)
        else:
            enc_out, x_lens, enc_out_sub, x_lens_sub, perm_idx = self._encode(
                xs, x_lens, volatile=True, is_multi_task=True)

        # Change to evaluation mode
        self.eval()

        # At first, decode by character-based decoder
        if self.ctc_loss_weight_sub > self.sub_loss_weight:
            # Decode by CTC decoder

            # Path through the softmax layer
            batch_size, max_time_sub = enc_out_sub.size()[:2]
            enc_out_sub = enc_out_sub.view(
                batch_size * max_time_sub, -1).contiguous()
            logits_ctc_sub = self.fc_ctc_sub(enc_out_sub)
            logits_ctc_sub = logits_ctc_sub.view(batch_size, max_time_sub, -1)

            if beam_width == 1:
                best_hyps_sub = self._decode_ctc_greedy_np(
                    self.var2np(logits_ctc_sub),
                    self.var2np(x_lens_sub))
            else:
                best_hyps_sub = self._decode_ctc_beam_np(
                    self.var2np(F.log_softmax(logits_ctc_sub, dim=-1)),
                    self.var2np(x_lens_sub), beam_width=beam_width)

            # NOTE: index 0 is reserved for blank in warpctc_pytorch
            best_hyps_sub -= 1
        else:
            # Decode by attention decoder
            if beam_width == 1:
                best_hyps_sub, _ = self._decode_infer_greedy(
                    enc_out_sub, x_lens_sub, max_decode_len, is_sub_task=True)
            else:
                best_hyps_sub, _ = self._decode_infer_beam(
                    enc_out_sub, x_lens_sub, max_decode_len, is_sub_task=True)

        if is_sub_task:
            best_hyps = best_hyps_sub
        else:
            y_lens_sub = np2var(
                np.array([len(y) for y in best_hyps_sub]),
                dtype='int', use_cuda=self.use_cuda, volatile=True)
            # assert max(y_lens_sub.data) > 0

            ys_sub = self._create_var(
                (len(xs), max(y_lens_sub.data)), dtype='long')
            for b in range(len(xs)):
                ys_sub.data[b, :len(best_hyps_sub[b])] = torch.from_numpy(
                    best_hyps_sub[b])

            # Next, decode by word-based decoder with character outputs
            if beam_width == 1:
                best_hyps, _, _ = self._decode_infer_greedy(
                    enc_out, x_lens,
                    max_decode_len=max_decode_len,
                    ys_sub=ys_sub,
                    y_lens_sub=y_lens_sub)
            else:
                raise NotImplementedError

        # Permutate indices to the original order
        if perm_idx is None:
            perm_idx = np.arange(0, len(xs), 1)
        else:
            perm_idx = self.var2np(perm_idx)

        if is_sub_task:
            return best_hyps, perm_idx
        else:
            return best_hyps, best_hyps_sub, perm_idx

    def _decode_infer_greedy(self, enc_out, x_lens, max_decode_len,
                             ys_sub=None, y_lens_sub=None, is_sub_task=False):
        """Greedy decoding in the inference stage.
        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T_in, encoder_num_units]`
            x_lens (torch.autograd.Variable, int): A tensor of size `[B]`
            max_decode_len (int): the length of output sequences
                to stop prediction when EOS token have not been emitted
            ys_sub (torch.autograd.Variable(float), optional):
                A tensor of size `[B, T_out_sub]`
            y_lens_sub (torch.autograd.Variable(int), optional):
                A tensor of size `[B]`
            is_sub_task (bool, optional):
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B, T_out]`
            att_weights (np.ndarray): A tensor of size `[B, T_out, T_in]`
            char_att_weights (np.ndarray): A tensor of size
                `[B, T_out, T_in_char]`
        """
        batch_size, max_time = enc_out.size()[:2]

        # Initialize decoder state
        dec_state = self._init_decoder_state(enc_out, volatile=True)
        dec_out = self._create_var(
            (batch_size, 1, self.decoder_num_units), volatile=True)
        context_vec = self._create_var(
            (batch_size, 1, self.encoder_num_units), volatile=True)
        att_weights_step = self._create_var(
            (batch_size, max_time), volatile=True)
        if not is_sub_task:
            # Path through character embedding
            chars_emb = []
            for c in range(ys_sub.size(1)):
                chars_emb.append(self.embed_sub(ys_sub[:, c:c + 1]))
            chars_emb = torch.cat(chars_emb, dim=1)
            # chars_emb: `[B, T_out, embedding_dim_sub]`

            char_mask = self._create_var(chars_emb.size(), volatile=True)
            for b in range(batch_size):
                char_mask.data[:, :y_lens_sub[b].data[0]] = 1
            chars_emb *= char_mask

            # Encode characters
            char_enc_out, y_lens_sub, _ = self.encoder_charseq(
                chars_emb, y_lens_sub, volatile=True)
            char_repr = self.char2word(char_enc_out)

            # Initialize attention weights for characters
            char_att_weights_step = self._create_var(
                tuple(ys_sub.size()), volatile=True)

        # Start from <SOS>
        sos = self.sos_index_sub if is_sub_task else self.sos_index
        eos = self.eos_index_sub if is_sub_task else self.eos_index
        y = self._create_var((batch_size, 1), fill_value=sos, dtype='long')

        best_hyps = []
        att_weights = []
        char_att_weights = []
        for _ in range(max_decode_len):

            if is_sub_task:
                y = self.embed_sub(y)
            else:
                y = self.embed(y)

            if not is_sub_task:
                # Compute attention weights and context vector for characters
                char_context_vec, char_att_weights_step = self.attend_charseq(
                    char_repr, y_lens_sub, dec_out, char_att_weights_step)
                # NOTE: char_context_vec is a word representation

                # Compose with word embedding
                if self.composition_case in ['fine_grained_gating', 'scalar_gating']:
                    gate = F.sigmoid(self.gate_fn(y))
                    y = gate * y + (1 - gate) * char_context_vec
                    # char_context_vec: `[B, 1, word_embedding_dim]`
                    # y: `[B, 1, word_embedding_dim]`
                elif self.composition_case == 'concat':
                    y = torch.cat([y, char_context_vec], dim=-1)

            dec_out, dec_state, context_vec, att_weights_step = self._decode_step(
                enc_out=enc_out,
                x_lens=x_lens,
                y=y,
                dec_state=dec_state,
                dec_out=dec_out,
                context_vec=context_vec,
                att_weights_step=att_weights_step,
                is_sub_task=is_sub_task)

            if is_sub_task:
                logits = self.fc_sub(F.tanh(
                    self.W_d_sub(dec_out) + self.W_c_sub(context_vec)))
            else:
                logits = self.fc(F.tanh(
                    self.W_d(dec_out) + self.W_c(context_vec)))

            # Pick up 1-best
            y = torch.max(logits.squeeze(1), dim=1)[1].unsqueeze(1)
            # logits: `[B, 1, num_classes]` -> `[B, num_classes]`
            best_hyps.append(y)
            att_weights.append(att_weights_step)
            if not is_sub_task:
                char_att_weights.append(char_att_weights_step)

            # Break if <EOS> is outputed in all mini-batch
            if torch.sum(y.data == eos) == y.numel():
                break

        # Concatenate in T_out dimension
        best_hyps = torch.cat(best_hyps, dim=1)
        att_weights = torch.stack(att_weights, dim=1)

        # Convert to numpy
        best_hyps = self.var2np(best_hyps)
        att_weights = self.var2np(att_weights)

        if not is_sub_task:
            char_att_weights = torch.stack(char_att_weights, dim=1)
            char_att_weights = self.var2np(char_att_weights)

            return best_hyps, att_weights, char_att_weights
        else:
            return best_hyps, att_weights
