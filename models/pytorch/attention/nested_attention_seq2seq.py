#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Nested attention-based sequence-to-sequence model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from warpctc_pytorch import CTCLoss
    # ctc_loss_fn = CTCLoss()
except:
    raise ImportError('Install warpctc_pytorch.')

import random

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from models.pytorch.linear import LinearND, Embedding
from models.pytorch.attention.attention_seq2seq import AttentionSeq2seq
from models.pytorch.encoders.load_encoder import load
from models.pytorch.attention.rnn_decoder import RNNDecoder
from models.pytorch.attention.attention_layer import AttentionMechanism
from models.pytorch.ctc.decoders.greedy_decoder import GreedyDecoder
from models.pytorch.ctc.decoders.beam_search_decoder import BeamSearchDecoder
from utils.io.variable import np2var, var2np
from models.pytorch.attention.char2word import LSTMChar2Word

LOG_1 = 0


class NestedAttentionSeq2seq(AttentionSeq2seq):

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
                 subsample_type='drop',
                 init_dec_state='final',
                 sharpening_factor=1,
                 logits_temperature=1,
                 sigmoid_smoothing=False,
                 input_feeding=False,
                 coverage_weight=0,
                 ctc_loss_weight_sub=0,
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
                 composition_case=None,  # ***
                 space_index=None):  # ***

        super(NestedAttentionSeq2seq, self).__init__(
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
            weight_noise_std=weight_noise_std)

        # Setting for the encoder
        self.encoder_num_layers_sub = encoder_num_layers_sub

        # Setting for the decoder
        self.decoder_num_units_sub = decoder_num_units_sub
        self.decoder_num_layers_sub = decoder_num_layers_sub
        self.embedding_dim_sub = embedding_dim_sub
        self.num_classes_sub = num_classes_sub + 2  # Add <SOS> and <EOS> class
        self.sos_index_sub = num_classes_sub + 1
        self.eos_index_sub = num_classes_sub

        if embedding_dim == 0:
            raise NotImplementedError

        # Setting for MTL
        self.main_loss_weight = main_loss_weight
        self.sub_loss_weight = 1 - main_loss_weight - ctc_loss_weight_sub
        self.ctc_loss_weight_sub = ctc_loss_weight_sub
        assert self.sub_loss_weight > 0

        # Setting for composition
        self.composition_case = composition_case
        self.space_index = space_index

        ####################
        # Encoder
        ####################
        if encoder_type in ['lstm', 'gru', 'rnn']:
            self.encoder = load(encoder_type=encoder_type)(
                input_size=input_size,
                rnn_type=encoder_type,
                bidirectional=encoder_bidirectional,
                num_units=encoder_num_units,
                num_proj=encoder_num_proj,
                num_layers=encoder_num_layers,
                num_layers_sub=encoder_num_layers_sub,
                dropout=encoder_dropout,
                subsample_list=subsample_list,
                subsample_type=subsample_type,
                use_cuda=self.use_cuda,
                batch_first=True,
                merge_bidirectional=False,
                num_stack=num_stack,
                splice=splice,
                conv_channels=conv_channels,
                conv_kernel_sizes=conv_kernel_sizes,
                conv_strides=conv_strides,
                poolings=poolings,
                activation=activation,
                batch_norm=batch_norm)
        else:
            raise NotImplementedError

        ####################
        # Decoder
        ####################
        if composition_case == 'embedding':
            decoder_input_size = embedding_dim + decoder_num_units
        elif composition_case == 'hidden':
            decoder_input_size = embedding_dim + decoder_num_units + decoder_num_units_sub
        elif composition_case == 'hidden_embedding':
            decoder_input_size = embedding_dim + decoder_num_units + decoder_num_units_sub
        elif composition_case == 'multiscale':
            decoder_input_size = embedding_dim + decoder_num_units + decoder_num_units_sub

        self.decoder = RNNDecoder(
            input_size=decoder_input_size,
            rnn_type=decoder_type,
            num_units=decoder_num_units,
            num_layers=decoder_num_layers,
            dropout=decoder_dropout,
            use_cuda=self.use_cuda,
            batch_first=True)
        self.decoder_sub = RNNDecoder(
            input_size=decoder_num_units_sub + embedding_dim_sub,
            rnn_type=decoder_type,
            num_units=decoder_num_units_sub,
            num_layers=decoder_num_layers_sub,
            dropout=decoder_dropout,
            use_cuda=self.use_cuda,
            batch_first=True)

        ##############################
        # Attention layer
        ##############################
        self.attend = AttentionMechanism(
            decoder_num_units=decoder_num_units,
            attention_type=attention_type,
            attention_dim=attention_dim,
            sharpening_factor=sharpening_factor,
            sigmoid_smoothing=sigmoid_smoothing,
            out_channels=attention_conv_num_channels,
            kernel_size=attention_conv_width)
        self.attend_sub = AttentionMechanism(
            decoder_num_units=decoder_num_units_sub,
            attention_type=attention_type,
            attention_dim=attention_dim,
            sharpening_factor=sharpening_factor,
            sigmoid_smoothing=sigmoid_smoothing,
            out_channels=attention_conv_num_channels,
            kernel_size=attention_conv_width)
        # NOTE: encoder's outputs will be mapped to the same dimension as the
        # decoder states

        ##################################################
        # Bridge layer between the encoder and decoder
        ##################################################
        if encoder_bidirectional or encoder_num_units != decoder_num_units:
            if encoder_bidirectional:
                self.bridge = LinearND(
                    encoder_num_units * 2, decoder_num_units,
                    dropout=decoder_dropout)
            else:
                self.bridge = LinearND(encoder_num_units, decoder_num_units,
                                       dropout=decoder_dropout)
            self.is_bridge = True
        else:
            self.is_bridge = False
        if encoder_bidirectional or encoder_num_units != decoder_num_units_sub:
            if encoder_bidirectional:
                self.bridge_sub = LinearND(
                    encoder_num_units * 2, decoder_num_units_sub,
                    dropout=decoder_dropout)
            else:
                self.bridge_sub = LinearND(
                    encoder_num_units, decoder_num_units_sub,
                    dropout=decoder_dropout)
            self.is_bridge_sub = True
        else:
            self.is_bridge_sub = False

        self.embed = Embedding(num_classes=self.num_classes,
                               embedding_dim=embedding_dim,
                               dropout=decoder_dropout)
        self.embed_sub = Embedding(num_classes=self.num_classes_sub,
                                   embedding_dim=embedding_dim_sub,
                                   dropout=decoder_dropout)

        if composition_case in ['hidden', 'hidden_embedding']:
            self.proj_layer = LinearND(
                decoder_num_units * 2 + decoder_num_units_sub, decoder_num_units,
                dropout=decoder_dropout)
        else:
            self.proj_layer = LinearND(
                decoder_num_units * 2, decoder_num_units,
                dropout=decoder_dropout)
        self.proj_layer_sub = LinearND(
            decoder_num_units_sub * 2, decoder_num_units_sub,
            dropout=decoder_dropout)
        self.fc = LinearND(decoder_num_units, self.num_classes - 1)
        self.fc_sub = LinearND(
            decoder_num_units_sub, self.num_classes_sub - 1)
        # NOTE: <SOS> is removed because the decoder never predict <SOS> class
        # TODO: consider projection

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
            # NOTE: index 0 is reserved for blank in warpctc_pytorch

        if composition_case in ['embedding', 'hidden_embedding', 'multiscale']:
            ##############################
            # C2W model
            ##############################
            self.c2w = LSTMChar2Word(num_units=decoder_num_units,
                                     num_layers=1,
                                     bidirectional=True,
                                     char_embedding_dim=embedding_dim_sub,
                                     word_embedding_dim=embedding_dim,
                                     use_cuda=self.use_cuda)

            self.gate_fn = LinearND(embedding_dim, embedding_dim)

        # Initialize all weights with uniform distribution
        self.init_weights(
            parameter_init, distribution='uniform', ignore_keys=['bias'])

        # Initialize all biases with 0
        self.init_weights(0, distribution='uniform', keys=['bias'])

        # Recurrent weights are orthogonalized
        # self.init_weights(parameter_init, distribution='orthogonal',
        #                   keys=['lstm', 'weight'], ignore_keys=['bias'])

        # Initialize bias in forget gate with 1
        self.init_forget_gate_bias_with_one()

    def forward(self, inputs, labels, labels_sub, inputs_seq_len,
                labels_seq_len, labels_seq_len_sub, is_eval=False):
        """Forward computation.
        Args:
            inputs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            labels (np.ndarray): A tensor of size `[B, T_out]`
            labels_sub (np.ndarray): A tensor of size `[B, T_out_sub]`
            inputs_seq_len (np.ndarray): A tensor of size `[B]`
            labels_seq_len (np.ndarray): A tensor of size `[B]`
            labels_seq_len_sub (np.ndarray): A tensor of size `[B]`
            is_eval (bool, optional): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (FloatTensor): A tensor of size `[1]`
            xe_loss_main (FloatTensor): A tensor of size `[1]`
            xe_loss_sub (FloatTensor): A tensor of size `[1]`
        """
        # Wrap by Variable
        xs = np2var(inputs, use_cuda=self.use_cuda, backend='pytorch')
        ys = np2var(
            labels, dtype='long', use_cuda=self.use_cuda, backend='pytorch')
        ys_sub = np2var(
            labels_sub, dtype='long', use_cuda=self.use_cuda, backend='pytorch')
        x_lens = np2var(
            inputs_seq_len, dtype='int', use_cuda=self.use_cuda, backend='pytorch')
        y_lens = np2var(
            labels_seq_len, dtype='int', use_cuda=self.use_cuda, backend='pytorch')
        y_lens_sub = np2var(
            labels_seq_len_sub, dtype='int', use_cuda=self.use_cuda, backend='pytorch')

        if is_eval:
            self.eval()
        else:
            self.train()

            # Gaussian noise injection
            if self.weight_noise_injection:
                self._inject_weight_noise(mean=0, std=self.weight_noise_std)

        # Encode acoustic features
        xs, _, xs_sub, x_lens_sub, perm_idx = self._encode(
            xs, x_lens, volatile=is_eval, is_multi_task=True)

        # Permutate indices
        if perm_idx is not None:
            ys = ys[perm_idx]
            ys_sub = ys_sub[perm_idx]
            y_lens = y_lens[perm_idx]
            y_lens_sub = y_lens_sub[perm_idx]

        # Teacher-forcing
        logits, logits_sub, att_weights, att_weights_sub = self._decode_train_joint(
            xs, xs_sub, ys, ys_sub, y_lens, y_lens_sub)

        # Output smoothing
        if self.logits_temperature != 1:
            logits = logits / self.logits_temperature
            logits_sub = logits_sub / self.logits_temperature

        # Compute XE sequence loss in the main task
        batch_size, label_num, num_classes = logits.size()
        logits = logits.view((-1, num_classes))
        ys_1d = ys[:, 1:].contiguous().view(-1)
        loss_main = F.cross_entropy(
            logits, ys_1d, ignore_index=self.sos_index, size_average=False)
        # NOTE: ys are padded by <SOS>

        # Compute XE sequence loss in the sub task
        batch_size, label_num_sub, num_classes_sub = logits_sub.size()
        logits_sub = logits_sub.view((-1, num_classes_sub))
        ys_sub_1d = ys_sub[:, 1:].contiguous().view(-1)
        loss_sub = F.cross_entropy(
            logits_sub, ys_sub_1d, ignore_index=self.sos_index_sub, size_average=False)
        # NOTE: ys_sub are padded by <SOS>

        # Label smoothing (with uniform distribution)
        if self.label_smoothing_prob > 0:
            log_probs = F.log_softmax(logits, dim=-1)
            uniform = Variable(torch.FloatTensor(
                batch_size, label_num, num_classes).fill_(np.log(1 / num_classes)))
            log_probs_sub = F.log_softmax(logits_sub, dim=-1)
            uniform_sub = Variable(torch.FloatTensor(
                batch_size, label_num, num_classes_sub).fill_(np.log(1 / num_classes_sub)))

            if self.use_cuda:
                uniform = uniform.cuda()
                uniform_sub = uniform_sub.cuda()

            loss_main = loss_main * (1 - self.label_smoothing_prob) + F.kl_div(
                log_probs, uniform,
                size_average=False, reduce=True) * self.label_smoothing_prob
            loss_sub = loss_sub * (1 - self.label_smoothing_prob) + F.kl_div(
                log_probs_sub, uniform_sub,
                size_average=False, reduce=True) * self.label_smoothing_prob

        # Add coverage term
        if self.coverage_weight != 0:
            pass
            # TODO: sub taskも入れる？

        loss_main = loss_main * self.main_loss_weight / batch_size
        loss_sub = loss_sub * self.sub_loss_weight / batch_size
        loss = loss_main + loss_sub

        ##################################################
        # Sub task (CTC)
        ##################################################
        if self.ctc_loss_weight_sub > 0:
            ctc_loss_sub = self.compute_ctc_loss(
                xs_sub, ys_sub, x_lens_sub, y_lens_sub, is_sub_task=True)
            # NOTE: including modifying inputs_seq_len_sub

            ctc_loss_sub = ctc_loss_sub * self.ctc_loss_weight_sub / batch_size
            loss += ctc_loss_sub

        if is_eval:
            loss = loss.data[0]
            loss_main = loss_main.data[0]
            loss_sub = loss_sub.data[0]
        else:
            self._step += 1

            # Update the probability of scheduled sampling
            if self.sample_prob > 0:
                self._sample_prob = min(
                    self.sample_prob,
                    self.sample_prob / self.sample_ramp_max_step * self._step)

        return loss, loss_main, loss_sub

    def _decode_train_joint(self, enc_out, enc_out_sub,
                            ys, ys_sub, y_lens, y_lens_sub):
        """Decoding of composition models in the training stage.
        Args:
            enc_out (FloatTensor): A tensor of size
                `[B, T_in, decoder_num_units]`
            enc_out_sub (FloatTensor): A tensor of size
                `[B, T_in, decoder_num_units_sub]`
            ys (LongTensor): A tensor of size `[B, T_out]`
            ys_sub (LongTensor): A tensor of size `[B, T_out_sub]`
            y_lens (np.ndarray): A tensor of size `[B]`
            y_lens_sub (np.ndarray): A tensor of size `[B]`
        Returns:
            logits (FloatTensor): A tensor of size `[B, T_out, num_classes]`
            logits_sub (FloatTensor): A tensor of size
                `[B, T_out_sub, num_classes_sub]`
            att_weights (FloatTensor): A tensor of size
                `[B, T_out, T_in]`
            att_weights_sub (FloatTensor): A tensor of size
                `[B, T_out_sub, T_in]`
        """
        batch_size = enc_out.size(0)
        max_time = enc_out.size(1)
        max_time_sub = enc_out_sub.size(1)
        assert max_time == max_time_sub

        logits = []
        logits_sub = []
        att_weights = []
        att_weights_sub = []

        # Batch-mode
        if self.composition_case == 'hidden':

            # Initialize decoder state
            dec_state = self._init_decoder_state(enc_out)
            dec_state_sub = self._init_decoder_state(enc_out_sub)

            # Initialize attention weights with uniform distribution
            # att_weights_step = Variable(torch.zeros(batch_size, max_time))
            # att_weights_step_sub = Variable(
            #     torch.zeros(batch_size, max_time_sub))
            att_weights_step = Variable(
                torch.ones(batch_size, max_time)) / max_time
            att_weights_step_sub = Variable(
                torch.oens(batch_size, max_time_sub)) / max_time_sub

            # Initialize context vector
            context_vec = Variable(torch.zeros(
                batch_size, 1, enc_out.size(2) * 2))
            context_vec_sub = Variable(torch.zeros(
                batch_size, 1, enc_out_sub.size(2)))

            if self.use_cuda:
                att_weights_step = att_weights_step.cuda()
                att_weights_step_sub = att_weights_step_sub.cuda()
                context_vec = context_vec.cuda()
                context_vec_sub = context_vec_sub.cuda()

            # Decode by character-level model at first
            ys_sub_max_seq_len = ys_sub.size(1)
            for t in range(ys_sub_max_seq_len - 1):

                is_sample = self.sample_prob > 0 and t > 0 and self._step > 0 and random.random(
                ) < self._sample_prob

                if is_sample:
                    # Scheduled sampling
                    y_prev_sub = torch.max(logits_sub[-1], dim=2)[1]
                    y_prev_sub = self.embed_sub(y_prev_sub)
                else:
                    # Teacher-forcing
                    y_prev_sub = self.embed_sub(ys_sub[:, t:t + 1])

                dec_in_sub = torch.cat([y_prev_sub, context_vec_sub], dim=-1)
                dec_out_sub, dec_state_sub, context_vec_sub, att_weights_step_sub = self._decode_step(
                    enc_out=enc_out_sub,
                    dec_in=dec_in_sub,
                    dec_state=dec_state_sub,
                    att_weights_step=att_weights_step_sub,
                    is_sub_task=True)

                concat_sub = torch.cat([dec_out_sub, context_vec_sub], dim=-1)
                attentional_vec_sub = F.tanh(self.proj_layer_sub(concat_sub))
                logits_step_sub = self.fc_sub(attentional_vec_sub)

                logits_sub.append(logits_step_sub)
                att_weights_sub.append(att_weights_step_sub)

            # Decode by word-level model
            ys_max_seq_len = ys.size(1)
            for t in range(ys_max_seq_len - 1):

                is_sample = self.sample_prob > 0 and t > 0 and self._step > 0 and random.random(
                ) < self._sample_prob

                if is_sample:
                    # Scheduled sampling
                    y_prev = torch.max(logits[-1], dim=2)[1]
                    y_prev = self.embed(y_prev)
                else:
                    # Teacher-forcing
                    y_prev = self.embed(ys[:, t:t + 1])

                dec_in = torch.cat([y_prev, context_vec], dim=-1)
                dec_out, dec_state, context_vec, att_weights_step = self._decode_step(
                    enc_out=enc_out,
                    dec_in=dec_in,
                    dec_state=dec_state,
                    att_weights_step=att_weights_step)

                # Compute character-level context vector
                context_vec_char = torch.sum(
                    enc_out_sub * att_weights_step.unsqueeze(2),
                    dim=1, keepdim=True)

                # Mix word-level and character-level context vector
                context_vec = torch.cat(
                    [context_vec, context_vec_char], dim=-1)

                concat = torch.cat([dec_out, context_vec], dim=-1)
                attentional_vec = F.tanh(self.proj_layer(concat))
                logits_step = self.fc(attentional_vec)

                logits.append(logits_step)
                att_weights.append(att_weights_step)

            # Concatenate in time-dimension
            logits = torch.cat(logits, dim=1)
            logits_sub = torch.cat(logits_sub, dim=1)
            att_weights = torch.stack(att_weights, dim=1)
            att_weights_sub = torch.stack(att_weights_sub, dim=1)

        # Process per utterance
        elif self.composition_case in ['embedding', 'hidden_embedding', 'multiscale']:

            # TODO: Cache the same word
            self.c2w_cache = {}

            ys_max_seq_len = ys.size(1)
            ys_sub_max_seq_len = ys_sub.size(1)

            for i_batch in range(batch_size):

                logits_i = []
                logits_sub_i = []
                att_weights_i = []
                att_weights_sub_i = []

                # Initialize decoder state
                dec_state = self._init_decoder_state(
                    enc_out[i_batch:i_batch + 1, :max_time])
                dec_state_sub = self._init_decoder_state(
                    enc_out_sub[i_batch:i_batch + 1, :max_time_sub])

                # Initialize attention weights with uniform distribution
                # att_weights_step = Variable(torch.zeros(1, max_time))
                # att_weights_step_sub = Variable(torch.zeros(1, max_time_sub))
                att_weights_step = Variable(torch.ones(1, max_time)) / max_time
                att_weights_step_sub = Variable(
                    torch.ones(1, max_time_sub)) / max_time_sub

                # Initialize context vector
                if self.composition_case in ['embedding', 'multiscale']:
                    context_vec = Variable(torch.zeros(1, 1, enc_out.size(2)))
                elif self.composition_case == 'hidden_embedding':
                    context_vec = Variable(
                        torch.zeros(1, 1, enc_out.size(2) * 2))
                context_vec_sub = Variable(
                    torch.zeros(1, 1, enc_out_sub.size(2)))

                if self.use_cuda:
                    att_weights_step = att_weights_step.cuda()
                    att_weights_step_sub = att_weights_step_sub.cuda()
                    context_vec = context_vec.cuda()
                    context_vec_sub = context_vec_sub.cuda()

                # Compute length of characters per word
                char_lens = [1]
                local_char_counter = 0
                # NOTE: remove <SOS> and <EOS>
                for char in ys_sub[i_batch, 1:y_lens_sub[i_batch].data[0] - 1]:
                    if char.data[0] == self.space_index:
                        char_lens.append(local_char_counter)
                        local_char_counter = 0
                    else:
                        local_char_counter += 1

                # The last word
                char_lens.append(local_char_counter)

                # <EOS>
                char_lens.append(1)

                word_len = y_lens[i_batch].data[0]
                global_char_counter = 0
                for t in range(word_len - 1):  # loop of words

                    ########################################
                    # Decode by character-level decoder
                    ########################################
                    char_embs = []
                    for i_char in range(char_lens[t]):  # loop of characters

                        is_sample = self.sample_prob > 0 and t > 0 and self._step > 0 and random.random(
                        ) < self._sample_prob

                        if is_sample:
                            # Scheduled sampling
                            y_prev_sub = torch.max(logits_sub_i[-1], dim=2)[1]
                            y_prev_sub = self.embed_sub(y_prev_sub)
                        else:
                            # Teacher-forcing
                            y_prev_sub = self.embed_sub(
                                ys_sub[i_batch:i_batch + 1, global_char_counter:global_char_counter + 1])

                        dec_in_sub = torch.cat(
                            [y_prev_sub, context_vec_sub], dim=-1)
                        dec_out_sub, dec_state_sub, context_vec_sub, att_weights_step_sub = self._decode_step(
                            enc_out=enc_out_sub[i_batch: i_batch +
                                                1, :max_time],
                            dec_in=dec_in_sub,
                            dec_state=dec_state_sub,
                            att_weights_step=att_weights_step_sub,
                            is_sub_task=True)

                        concat_sub = torch.cat(
                            [dec_out_sub, context_vec_sub], dim=-1)
                        attentional_vec_sub = F.tanh(
                            self.proj_layer_sub(concat_sub))
                        logits_step_sub = self.fc_sub(attentional_vec_sub)

                        logits_sub_i.append(logits_step_sub)
                        att_weights_sub_i.append(att_weights_step_sub)

                        char_emb = self.embed_sub(
                            ys_sub[i_batch:i_batch + 1, global_char_counter:global_char_counter + 1])
                        # NOTE: char_emb: `[1, 1, embedding_dim_sub]`
                        char_embs.append(char_emb)
                        global_char_counter += 1

                    ###############
                    # For space
                    ###############
                    if 0 < t < word_len - 2:

                        is_sample = self.sample_prob > 0 and t > 0 and self._step > 0 and random.random(
                        ) < self._sample_prob

                        if is_sample:
                            # Scheduled sampling
                            y_prev_sub = torch.max(logits_sub_i[-1], dim=2)[1]
                            y_prev_sub = self.embed_sub(y_prev_sub)
                        else:
                            # Teacher-forcing
                            y_prev_sub = self.embed_sub(
                                ys_sub[i_batch:i_batch + 1, global_char_counter:global_char_counter + 1])

                        dec_in_sub = torch.cat(
                            [y_prev_sub, context_vec_sub], dim=-1)
                        dec_out_sub, dec_state_sub, context_vec_sub, att_weights_step_sub = self._decode_step(
                            enc_out=enc_out_sub[i_batch: i_batch +
                                                1, :max_time],
                            dec_in=dec_in_sub,
                            dec_state=dec_state_sub,
                            att_weights_step=att_weights_step_sub,
                            is_sub_task=True)

                        concat_sub = torch.cat(
                            [dec_out_sub, context_vec_sub], dim=-1)
                        attentional_vec_sub = F.tanh(
                            self.proj_layer_sub(concat_sub))
                        logits_step_sub = self.fc_sub(attentional_vec_sub)

                        logits_sub_i.append(logits_step_sub)
                        att_weights_sub_i.append(att_weights_step_sub)
                        global_char_counter += 1

                    char_embs = torch.cat(char_embs, dim=1)
                    # NOTE: char_embs: `[1, char_num, embedding_dim_sub]`

                    # Get Word representation from a sequence of characters
                    word_repr = self.c2w(char_embs)
                    # NOTE: word_repr: `[1, 1, embedding_dim]`

                    ########################################
                    # Decode by word-level decoder
                    ########################################
                    is_sample = self.sample_prob > 0 and t > 0 and self._step > 0 and random.random(
                    ) < self._sample_prob

                    if is_sample:
                        # Scheduled sampling
                        y_prev = torch.max(logits_i[-1], dim=2)[1]
                        y_prev = self.embed(y_prev)
                    else:
                        # Teacher-forcing
                        y_prev = self.embed(ys[i_batch:i_batch + 1, t:t + 1])

                    if self.composition_case in ['embedding', 'hidden_embedding']:
                        # Mix word embedding and word representation form C2W
                        gate = F.sigmoid(self.gate_fn(y_prev))
                        y_composition = (1 - gate) * y_prev + gate * word_repr
                        # TODO: 足し算じゃなくて，concatでもいいかも
                        dec_in = torch.cat(
                            [y_composition, context_vec], dim=-1)
                    elif self.composition_case == 'multiscale':
                        dec_in = torch.cat([y_prev, context_vec], dim=-1)
                        dec_in = torch.cat([dec_in, dec_out_sub], dim=-1)

                    dec_out, dec_state, context_vec, att_weights_step = self._decode_step(
                        enc_out=enc_out[i_batch: i_batch + 1, :max_time],
                        dec_in=dec_in,
                        dec_state=dec_state,
                        att_weights_step=att_weights_step)

                    if self.composition_case == 'hidden_embedding':
                        # Compute character-level context vector
                        context_vec_char = torch.sum(
                            enc_out_sub[i_batch: i_batch + 1, :max_time] *
                            att_weights_step.unsqueeze(2), dim=1, keepdim=True)

                        # Mix word-level and character-level context vector
                        context_vec = torch.cat(
                            [context_vec, context_vec_char], dim=-1)

                    concat = torch.cat([dec_out, context_vec], dim=-1)
                    attentional_vec = F.tanh(self.proj_layer(concat))
                    logits_step = self.fc(attentional_vec)

                    logits_i.append(logits_step)
                    att_weights_i.append(att_weights_step)

                # Pad tensor
                for _ in range(ys_max_seq_len - len(logits_i) - 1):
                    logits_i.append(torch.zeros_like(logits_step))
                    att_weights_i.append(torch.zeros_like(att_weights_step))
                for _ in range(ys_sub_max_seq_len - len(logits_sub_i) - 1):
                    logits_sub_i.append(torch.zeros_like(logits_step_sub))
                    att_weights_sub_i.append(
                        torch.zeros_like(att_weights_step_sub))
                # NOTE: remove <EOS>

                # Concatenate in time-dimension
                logits_i = torch.cat(logits_i, dim=1)
                logits_sub_i = torch.cat(logits_sub_i, dim=1)
                att_weights_i = torch.stack(att_weights_i, dim=1)
                att_weights_sub_i = torch.stack(att_weights_sub_i, dim=1)

                logits.append(logits_i)
                logits_sub.append(logits_sub_i)
                # att_weights.append(att_weights_i)
                # att_weights_sub.append(att_weights_sub_i)

            # Concatenate in batch-dimension
            logits = torch.cat(logits, dim=0)
            logits_sub = torch.cat(logits_sub, dim=0)
            # att_weights = torch.stack(att_weights, dim=0)
            # att_weights_sub = torch.stack(att_weights_sub, dim=0)
            att_weights = None
            att_weights_sub = None

        return logits, logits_sub, att_weights, att_weights_sub

    def _decode_step(self, enc_out, dec_in, dec_state,
                     att_weights_step, is_sub_task=False):
        """Decoding step.
        Args:
            enc_out (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            dec_in (FloatTensor): A tensor of size
                `[B, 1, embedding_dim + decoder_num_units]`
            dec_state (FloatTensor): A tensor of size
                `[decoder_num_layers, B, decoder_num_units]`
            att_weights_step (FloatTensor): A tensor of size `[B, T_in]`
            is_sub_task (bool, optional):
        Returns:
            dec_out (FloatTensor): A tensor of size
                `[B, 1, decoder_num_units]`
            dec_state (FloatTensor): A tensor of size
                `[decoder_num_layers, B, decoder_num_units]`
            content_vector (FloatTensor): A tensor of size
                `[B, 1, encoder_num_units]`
            att_weights_step (FloatTensor): A tensor of size `[B, T_in]`
        """
        if is_sub_task:
            dec_out, dec_state = self.decoder_sub(dec_in, dec_state)
            context_vec, att_weights_step = self.attend_sub(
                enc_out, dec_out, att_weights_step)
        else:
            dec_out, dec_state = self.decoder(dec_in, dec_state)
            context_vec, att_weights_step = self.attend(
                enc_out, dec_out, att_weights_step)

        return dec_out, dec_state, context_vec, att_weights_step

    def attention_weights(self, inputs, inputs_seq_len, beam_width=1,
                          max_decode_len=100):
        """Get attention weights for visualization.
        Args:
            inputs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            inputs_seq_len (np.ndarray): A tensor of size `[B]`
            beam_width (int, optional): the size of beam
            max_decode_len (int, optional): the length of output sequences
                to stop prediction when EOS token have not been emitted
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B, T_out]`
                Note that best_hyps includes <SOS> tokens.
            att_weights (np.ndarray): A tensor of size `[B, T_out, T_in]`
        """
        # Wrap by Variable
        xs = np2var(
            inputs, use_cuda=self.use_cuda, volatile=True, backend='pytorch')
        x_lens = np2var(
            inputs_seq_len, dtype='int', use_cuda=self.use_cuda, volatile=True, backend='pytorch')

        # Encode acoustic features
        enc_out, x_lens, perm_idx = self._encode(xs, x_lens, volatile=True)

        if beam_width == 1:
            best_hyps, att_weights = self._decode_infer_greedy(
                enc_out, max_decode_len)
        else:
            best_hyps, att_weights = self._decode_infer_beam(
                enc_out, x_lens, beam_width, max_decode_len)

        # Permutate indices to the original order
        if perm_idx is not None:
            perm_idx = var2np(perm_idx)
            best_hyps = best_hyps[perm_idx]
            att_weights = att_weights[perm_idx]

        return best_hyps, att_weights

    def decode(self, inputs, inputs_seq_len, beam_width=1,
               max_decode_len=100, is_sub_task=False):
        """Decoding in the inference stage.
        Args:
            inputs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            inputs_seq_len (np.ndarray): A tensor of size `[B]`
            beam_width (int, optional): the size of beam
            max_decode_len (int, optional): the length of output sequences
                to stop prediction when EOS token have not been emitted
            is_sub_task (bool, optional):
        Returns:
            best_hyps (np.ndarray): A tensor of size `[]`
        """
        # Wrap by Variable
        xs = np2var(
            inputs, use_cuda=self.use_cuda, volatile=True, backend='pytorch')
        x_lens = np2var(
            inputs_seq_len, dtype='int', use_cuda=self.use_cuda, volatile=True, backend='pytorch')

        # Change to evaluation mode
        self.eval()

        # Encode acoustic features
        if is_sub_task:
            _, _, enc_out, _, perm_idx = self._encode(
                xs, x_lens, volatile=True, is_multi_task=True)
        else:
            enc_out, _, enc_out_sub, _, perm_idx = self._encode(
                xs, x_lens, volatile=True, is_multi_task=True)

        if beam_width == 1:
            if is_sub_task:
                best_hyps, _ = self._decode_infer_greedy(
                    enc_out, max_decode_len, is_sub_task=True)
            else:
                best_hyps, _, _, _ = self._decode_infer_greedy_joint(
                    enc_out, enc_out_sub, max_decode_len)
        else:
            raise NotImplementedError

        if is_sub_task or self.composition_case == 'hidden':
            # Permutate indices to the original order
            if perm_idx is not None:
                perm_idx = var2np(perm_idx)
                best_hyps = best_hyps[perm_idx]
        # TODO: fix this

        return best_hyps

    def _decode_infer_greedy_joint(self, enc_out, enc_out_sub, max_decode_len):
        """Greedy decoding in the inference stage.
        Args:
            enc_out (FloatTensor): A tensor of size
                `[B, T_in, decoder_num_units]`
            enc_out_sub (FloatTensor): A tensor of size
                `[B, T_in, decoder_num_units_sub]`
            max_decode_len (int): the length of output sequences
                to stop prediction when EOS token have not been emitted
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B, T_out]`
                Note that best_hyps includes <SOS> tokens.
            best_hyps_sub (np.ndarray): A tensor of size `[B, T_out_sub]`
                Note that best_hyps_sub includes <SOS> tokens.
            att_weights (np.ndarray): A tensor of size `[B, T_out, T_in]`
            att_weights_sub (np.ndarray): A tensor of size
                `[B, T_out_sub, T_in]`
        """
        batch_size = enc_out.size(0)
        max_time = enc_out.size(1)
        max_time_sub = enc_out_sub.size(1)
        assert max_time == max_time_sub

        # Batch-mode
        if self.composition_case == 'hidden':

            # Initialize decoder state
            dec_state = self._init_decoder_state(enc_out, volatile=True)

            # Initialize attention weights with uniform distribution
            # att_weights_step = Variable(torch.zeros(batch_size, max_time))
            att_weights_step = Variable(torch.ones(batch_size, max_time))
            att_weights_step.volatile = True

            # Initialize context vector
            context_vec = Variable(
                torch.zeros(batch_size, 1, enc_out.size(2) * 2))
            context_vec.volatile = True

            if self.use_cuda:
                att_weights_step = att_weights_step.cuda()
                context_vec = context_vec.cuda()

            # Start from <SOS>
            y = self._create_token(value=self.sos_index, batch_size=batch_size)

            best_hyps = [y]
            best_hyps_sub = []
            att_weights = [att_weights_step]
            att_weights_sub = []

            for _ in range(max_decode_len):

                y = self.embed(y)

                dec_in = torch.cat([y, context_vec], dim=-1)
                dec_out, dec_state, context_vec, att_weights_step = self._decode_step(
                    enc_out=enc_out,
                    dec_in=dec_in,
                    dec_state=dec_state,
                    att_weights_step=att_weights_step)

                # Compute character-level context vector
                context_vec_char = torch.sum(
                    enc_out_sub * att_weights_step.unsqueeze(2),
                    dim=1, keepdim=True)

                # Mix word-level and character-level context vector
                context_vec = torch.cat(
                    [context_vec, context_vec_char], dim=-1)

                concat = torch.cat([dec_out, context_vec], dim=-1)
                attentional_vec = F.tanh(self.proj_layer(concat))
                logits = self.fc(attentional_vec)

                logits = logits.squeeze(1)
                # NOTE: `[B, 1, num_classes]` -> `[B, num_classes]`

                # Path through the softmax layer & convert to log-scale
                log_probs = F.log_softmax(logits, dim=-1)

                # Pick up 1-best
                y = torch.max(log_probs, dim=1)[1]
                y = y.unsqueeze(1)
                best_hyps.append(y)
                att_weights.append(att_weights_step)

                # Break if <EOS> is outputed in all mini-batch
                if torch.sum(y.data == self.eos_index) == y.numel():
                    break

            # Concatenate in T_out dimension
            best_hyps = torch.cat(best_hyps, dim=1)
            att_weights = torch.stack(att_weights, dim=1)

            # Convert to numpy
            best_hyps = var2np(best_hyps)
            att_weights = var2np(att_weights)

        # Process per utterance
        elif self.composition_case in ['embedding', 'hidden_embedding', 'multiscale']:

            # TODO: Cache the same word
            self.c2w_cache = {}

            best_hyps = []
            best_hyps_sub = []
            att_weights = []
            att_weights_sub = []

            for i_batch in range(batch_size):

                # Initialize decoder state
                dec_state = self._init_decoder_state(
                    enc_out[i_batch:i_batch + 1, :max_time], volatile=True)
                dec_state_sub = self._init_decoder_state(
                    enc_out_sub[i_batch:i_batch + 1, :max_time_sub], volatile=True)

                # Initialize attention weights with uniform distribution
                # att_weights_step = Variable(torch.zeros(1, max_time))
                # att_weights_step_sub = Variable(torch.zeros(1, max_time_sub))
                att_weights_step = Variable(torch.ones(1, max_time)) / max_time
                att_weights_step_sub = Variable(
                    torch.ones(1, max_time_sub)) / max_time_sub
                att_weights_step.volatile = True
                att_weights_step_sub.volatile = True

                # Initialize context vector
                if self.composition_case in ['embedding', 'multiscale']:
                    context_vec = Variable(torch.zeros(1, 1, enc_out.size(2)))
                elif self.composition_case == 'hidden_embedding':
                    context_vec = Variable(
                        torch.zeros(1, 1, enc_out.size(2) * 2))
                context_vec_sub = Variable(
                    torch.zeros(1, 1, enc_out_sub.size(2)))
                context_vec.volatile = True
                context_vec_sub.volatile = True

                if self.use_cuda:
                    att_weights_step = att_weights_step.cuda()
                    att_weights_step_sub = att_weights_step_sub.cuda()
                    context_vec = context_vec.cuda()
                    context_vec_sub = context_vec_sub.cuda()

                # Start from <SOS>
                y = self._create_token(value=self.sos_index, batch_size=1)
                y_sub = self._create_token(
                    value=self.sos_index_sub, batch_size=1)

                # best_hyps_i = [var2np(y)]
                # best_hyps_sub_i = [var2np(y_sub)]
                # att_weights_i = [var2np(att_weights_step)]
                # att_weights_sub_i = [var2np(att_weights_step_sub)]
                best_hyps_i = []
                best_hyps_sub_i = []
                att_weights_i = []
                att_weights_sub_i = []

                global_char_counter = 0
                for _ in range(max_decode_len):

                    ###########################################################
                    # Decode by character-level decoder until space is emitted
                    ###########################################################
                    local_char_counter = 0
                    char_embs = []
                    while local_char_counter < 10:
                        y_sub = self.embed_sub(y_sub)

                        dec_in_sub = torch.cat(
                            [y_sub, context_vec_sub], dim=-1)
                        dec_out_sub, dec_state_sub, context_vec_sub, att_weights_step_sub = self._decode_step(
                            enc_out=enc_out_sub[i_batch: i_batch +
                                                1, :max_time],
                            dec_in=dec_in_sub,
                            dec_state=dec_state_sub,
                            att_weights_step=att_weights_step_sub,
                            is_sub_task=True)

                        concat_sub = torch.cat(
                            [dec_out_sub, context_vec_sub], dim=-1)
                        attentional_vec_sub = F.tanh(
                            self.proj_layer_sub(concat_sub))
                        logits_sub = self.fc_sub(attentional_vec_sub)

                        logits_sub = logits_sub.squeeze(1)
                        # NOTE: `[B, 1, num_classes_sub]` ->
                        #       `[B, num_classes_sub]`

                        # Path through the softmax layer & convert to log-scale
                        log_probs_sub = F.log_softmax(logits_sub, dim=-1)

                        # Pick up 1-best
                        y_sub = torch.max(log_probs_sub, dim=1)[1]
                        y_sub = y_sub.unsqueeze(1)

                        local_char_counter += 1
                        global_char_counter += 1

                        if y_sub[0].data[0] in [self.eos_index_sub, self.space_index]:
                            break

                        emb_char = self.embed_sub(y_sub)
                        # NOTE: emb_char: `[1, 1, embedding_dim_sub]`
                        char_embs.append(emb_char)

                    # Break if no character is emitted
                    if len(char_embs) == 0:
                        break

                    char_embs = torch.cat(char_embs, dim=1)
                    # NOTE: char_embs: `[1, char_num, embedding_dim_sub]`

                    # Get Word representation from a sequence of character
                    word_repr = self.c2w(char_embs, volatile=True)
                    # NOTE: word_repr: `[1, 1, embedding_dim_sub * 2]`

                    ########################################
                    # Decode by word-level decoder
                    ########################################
                    y = self.embed(y)

                    if self.composition_case in ['embedding', 'hidden_embedding']:
                        # Mix word embedding and word representation form C2W
                        gate = F.sigmoid(self.gate_fn(y))
                        y_composition = (1 - gate) * y + gate * word_repr
                        # TODO: 足し算じゃなくて，concatでもいいかも
                        dec_in = torch.cat(
                            [y_composition, context_vec], dim=-1)
                    elif self.composition_case == 'multiscale':
                        dec_in = torch.cat([y, context_vec], dim=-1)
                        dec_in = torch.cat([dec_in, dec_out_sub], dim=-1)

                    dec_out, dec_state, context_vec, att_weights_step = self._decode_step(
                        enc_out=enc_out[i_batch: i_batch + 1, :max_time],
                        dec_in=dec_in,
                        dec_state=dec_state,
                        att_weights_step=att_weights_step)

                    if self.composition_case == 'hidden_embedding':
                        # Compute character-level context vector
                        context_vec_char = torch.sum(
                            enc_out_sub[i_batch: i_batch + 1, :max_time] *
                            att_weights_step.unsqueeze(2),
                            dim=1, keepdim=True)

                        # Mix word-level and character-level context vector
                        context_vec = torch.cat(
                            [context_vec, context_vec_char], dim=-1)

                    concat = torch.cat(
                        [dec_out, context_vec], dim=-1)
                    attentional_vec = F.tanh(self.proj_layer(concat))
                    logits = self.fc(attentional_vec)

                    logits = logits.squeeze(1)
                    # NOTE: `[B, 1, num_classes]` -> `[B, num_classes]`

                    # Path through the softmax layer & convert to log-scale
                    log_probs = F.log_softmax(logits, dim=-1)

                    # Pick up 1-best
                    y = torch.max(log_probs, dim=1)[1]
                    y = y.unsqueeze(1)
                    best_hyps_i.append(y.data[0, 0])
                    att_weights_i.append(var2np(att_weights_step))

                    # Break if <EOS> is outputed
                    if y.data[0, 0] == self.eos_index:
                        break

                best_hyps.append(best_hyps_i)
                best_hyps_sub.append(best_hyps_sub_i)
                att_weights.append(att_weights_i)
                att_weights_sub.append(att_weights_sub_i)

        return best_hyps, best_hyps_sub, att_weights, att_weights_sub
