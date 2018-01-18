#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention-based sequence-to-sequence model (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from warpctc_pytorch import CTCLoss
    ctc_loss = CTCLoss()
except:
    raise ImportError('Install warpctc_pytorch.')

import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from models.pytorch.base import ModelBase
from models.pytorch.linear import LinearND, Embedding
from models.pytorch.encoders.load_encoder import load
from models.pytorch.attention.rnn_decoder import RNNDecoder
from models.pytorch.attention.attention_layer import AttentionMechanism
from models.pytorch.ctc.ctc import _concatenate_labels
from models.pytorch.ctc.decoders.greedy_decoder import GreedyDecoder
from models.pytorch.ctc.decoders.beam_search_decoder import BeamSearchDecoder
from utils.io.variable import np2var, var2np

LOG_1 = 0


class AttentionSeq2seq(ModelBase):
    """The Attention-besed model.
    Args:
        input_size (int): the dimension of input features
        encoder_type (string): the type of the encoder. Set lstm or gru or rnn.
        encoder_bidirectional (bool): if True, create a bidirectional encoder
        encoder_num_units (int): the number of units in each layer of the encoder
        encoder_num_proj (int): the number of nodes in the projection layer of the encoder
        encoder_num_layers (int): the number of layers of the encoder
        encoder_dropout (float): the probability to drop nodes of the encoder
        attention_type (string): the type of attention
        attention_dim: (int) the dimension of the attention layer
        decoder_type (string): lstm or gru
        decoder_num_units (int): the number of units in each layer of the decoder
        decoder_num_layers (int): the number of layers of the decoder
        decoder_dropout (float): the probability to drop nodes of the decoder
        embedding_dim (int): the dimension of the embedding in target spaces.
            0 means that decoder inputs are represented by one-hot vectors.
        num_classes (int): the number of nodes in softmax layer
            (excluding <SOS> and <EOS> classes)
        parameter_init (float, optional): Range of uniform distribution to
            initialize weight parameters
        subsample_list (list, optional): subsample in the corresponding layers (True)
            ex.) [False, True, True, False] means that subsample is conducted
                in the 2nd and 3rd layers.
        subsample_type (string, optional): drop or concat
        init_dec_state (bool, optional): how to initialize decoder state
            zero => initialize with zero state
            mean => initialize with the mean of encoder outputs in all time steps
            final => initialize with tha final encoder state
        sharpening_factor (float, optional): a sharpening factor in the
            softmax layer for computing attention weights
        logits_temperature (float, optional): a parameter for smoothing the
            softmax layer in outputing probabilities
        sigmoid_smoothing (bool, optional): if True, replace softmax function
            in computing attention weights with sigmoid function for smoothing
        coverage_weight (float, optional): the weight parameter for coverage computation
        ctc_loss_weight (float): A weight parameter for auxiliary CTC loss
        attention_conv_num_channels (int, optional): the number of channles of
            conv outputs. This is used for location-based attention.
        attention_conv_width (int, optional): the size of kernel.
            This must be the odd number.
        num_stack (int, optional): the number of frames to stack
        splice (int, optional): frames to splice. Default is 1 frame.
        conv_channels (list, optional): the number of channles in the
            convolution of the location-based attention
        conv_kernel_sizes (list, optional): the size of kernels in the
            convolution of the location-based attention
        conv_strides (list, optional): strides in the convolution
            of the location-based attention
        poolings (list, optional): the size of poolings in the convolution
            of the location-based attention
        activation (string, optional): The activation function of CNN layers.
            Choose from relu or prelu or hard_tanh or maxout
        batch_norm (bool, optional):
        scheduled_sampling_prob (float, optional):
        scheduled_sampling_ramp_max_step (float, optional):
        label_smoothing_prob (float, optional):
        weight_noise_std (flaot, optional):
        encoder_residual (bool, optional):
        encoder_dense_residual (bool, optional):
        decoder_residual (bool, optional):
        decoder_dense_residual (bool, optional):
    """

    def __init__(self,
                 input_size,
                 encoder_type,
                 encoder_bidirectional,
                 encoder_num_units,
                 encoder_num_proj,
                 encoder_num_layers,
                 encoder_dropout,
                 attention_type,
                 attention_dim,
                 decoder_type,
                 decoder_num_units,
                 decoder_num_layers,
                 decoder_dropout,
                 embedding_dim,
                 num_classes,
                 parameter_init=0.1,
                 subsample_list=[],
                 subsample_type='drop',
                 init_dec_state='final',
                 sharpening_factor=1,
                 logits_temperature=1,
                 sigmoid_smoothing=False,
                 coverage_weight=0,
                 ctc_loss_weight=0,
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

        super(ModelBase, self).__init__()

        # TODO: clip_activation

        # Setting for the encoder
        self.input_size = input_size
        self.num_stack = num_stack
        self.encoder_type = encoder_type
        self.encoder_bidirectional = encoder_bidirectional
        self.encoder_num_directions = 2 if encoder_bidirectional else 1
        self.encoder_num_units = encoder_num_units
        self.encoder_num_proj = encoder_num_proj
        self.encoder_num_layers = encoder_num_layers
        self.subsample_list = subsample_list

        # Setting for the decoder
        self.attention_type = attention_type
        self.attention_dim = attention_dim
        self.decoder_type = decoder_type
        self.decoder_num_units = decoder_num_units
        self.decoder_num_layers = decoder_num_layers
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes + 2  # Add <SOS> and <EOS> class
        self.sos_index = num_classes + 1
        self.eos_index = num_classes

        if embedding_dim == 0:
            self.decoder_input = 'onehot'
        else:
            self.decoder_input = 'embedding'
        # self.decoder_input = 'onehot_prob'

        # Setting for the attention
        if init_dec_state not in ['zero', 'mean', 'final']:
            raise ValueError(
                'init_dec_state must be "zero" or "mean" or "final".')
        self.init_dec_state = init_dec_state
        self.sharpening_factor = sharpening_factor
        self.logits_temperature = logits_temperature
        self.sigmoid_smoothing = sigmoid_smoothing
        self.coverage_weight = coverage_weight
        self.attention_conv_num_channels = attention_conv_num_channels
        self.attention_conv_width = attention_conv_width

        # Setting for regularization
        self.parameter_init = parameter_init
        self.weight_noise_injection = False
        self.weight_noise_std = float(weight_noise_std)
        if scheduled_sampling_prob > 0 and scheduled_sampling_ramp_max_step == 0:
            raise ValueError
        self.sample_prob = scheduled_sampling_prob
        self._sample_prob = scheduled_sampling_prob
        self.sample_ramp_max_step = scheduled_sampling_ramp_max_step
        self._step = 0
        self.label_smoothing_prob = label_smoothing_prob

        # Joint CTC-Attention
        self.ctc_loss_weight = ctc_loss_weight

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
                dropout=encoder_dropout,
                subsample_list=subsample_list,
                subsample_type=subsample_type,
                batch_first=True,
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
        elif encoder_type == 'cnn':
            assert num_stack == 1
            assert splice == 1
            self.encoder = load(encoder_type=encoder_type)(
                input_size=input_size,
                conv_channels=conv_channels,
                conv_kernel_sizes=conv_kernel_sizes,
                conv_strides=conv_strides,
                poolings=poolings,
                dropout=encoder_dropout,
                activation=activation,
                batch_norm=batch_norm)
            self.init_dec_state = 'zero'
        else:
            raise NotImplementedError

        ####################
        # Decoder
        ####################
        if self.decoder_input == 'embedding':
            decoder_input_size = decoder_num_units + embedding_dim
        elif self.decoder_input == 'onehot':
            decoder_input_size = decoder_num_units + self.num_classes
        elif self.decoder_input == 'onehot_prob':
            decoder_input_size = decoder_num_units + self.num_classes * 2
        else:
            raise TypeError
        self.decoder = RNNDecoder(
            input_size=decoder_input_size,
            rnn_type=decoder_type,
            num_units=decoder_num_units,
            num_layers=decoder_num_layers,
            dropout=decoder_dropout,
            batch_first=True,
            residual=decoder_residual,
            dense_residual=decoder_dense_residual)

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

        ##################################################
        # Bridge layer between the encoder and decoder
        ##################################################
        if encoder_bidirectional or encoder_num_units != decoder_num_units or encoder_type == 'cnn':
            if encoder_type == 'cnn':
                self.bridge = LinearND(
                    self.encoder.output_size, decoder_num_units,
                    dropout=decoder_dropout)
            elif encoder_bidirectional:
                self.bridge = LinearND(
                    encoder_num_units * 2, decoder_num_units,
                    dropout=decoder_dropout)
            else:
                self.bridge = LinearND(encoder_num_units, decoder_num_units,
                                       dropout=decoder_dropout)
            self.is_bridge = True
        else:
            self.is_bridge = False

        if self.decoder_input == 'embedding':
            self.embed = Embedding(num_classes=self.num_classes,
                                   embedding_dim=embedding_dim,
                                   dropout=decoder_dropout)

        self.proj_layer = LinearND(decoder_num_units * 2, decoder_num_units,
                                   dropout=decoder_dropout)
        if self.decoder_input == 'onehot_prob':
            self.fc = LinearND(decoder_num_units, self.num_classes)
        else:
            self.fc = LinearND(decoder_num_units, self.num_classes - 1)
            # NOTE: <SOS> is excluded because the decoder never predict <SOS>

        if ctc_loss_weight > 0:
            if self.is_bridge:
                self.fc_ctc = LinearND(decoder_num_units, num_classes + 1)
            else:
                self.fc_ctc = LinearND(
                    encoder_num_units * self.encoder_num_directions, num_classes + 1)

            # Set CTC decoders
            self._decode_ctc_greedy_np = GreedyDecoder(blank_index=0)
            self._decode_ctc_beam_np = BeamSearchDecoder(blank_index=0)
            # NOTE: index 0 is reserved for blank in warpctc_pytorch
            # TODO: set space index

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

    def forward(self, xs, ys, x_lens, y_lens, is_eval=False):
        """Forward computation.
        Args:
            xs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            ys (np.ndarray): A tensor of size `[B, T_out]`
            x_lens (np.ndarray): A tensor of size `[B]`
            y_lens (np.ndarray): A tensor of size `[B]`
            is_eval (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (FloatTensor or float): A tensor of size `[1]`
        """
        # Wrap by Variable
        xs = np2var(xs,  use_cuda=self.use_cuda, backend='pytorch')
        ys = np2var(
            ys, dtype='long', use_cuda=self.use_cuda, backend='pytorch')
        # NOTE: ys must be long
        x_lens = np2var(
            x_lens, dtype='int', use_cuda=self.use_cuda, backend='pytorch')
        y_lens = np2var(
            y_lens, dtype='int', use_cuda=self.use_cuda, backend='pytorch')

        if is_eval:
            self.eval()
        else:
            self.train()

            # Gaussian noise injection
            if self.weight_noise_injection:
                self._inject_weight_noise(mean=0, std=self.weight_noise_std)

        # Encode acoustic features
        xs, x_lens, perm_idx = self._encode(xs, x_lens, volatile=is_eval)

        # Permutate indices
        if perm_idx is not None:
            ys = ys[perm_idx]
            y_lens = y_lens[perm_idx]

        # Teacher-forcing
        logits, att_weights = self._decode_train(xs, ys)

        # Output smoothing
        if self.logits_temperature != 1:
            logits = logits / self.logits_temperature

        # Compute XE sequence loss
        loss = F.cross_entropy(
            input=logits.view((-1, logits.size(2))),
            target=ys[:, 1:].contiguous().view(-1),
            ignore_index=self.sos_index, size_average=False)
        # NOTE: ys are padded by <SOS>

        # Label smoothing (with uniform distribution)
        if self.label_smoothing_prob > 0 and self.decoder_input == 'embedding':
            batch_size, label_num, num_classes = logits.size()
            log_probs = F.log_softmax(logits, dim=-1)
            uniform = Variable(torch.FloatTensor(
                batch_size, label_num, num_classes).fill_(np.log(1 / num_classes)))
            if self.use_cuda:
                uniform = uniform.cuda()
            loss = loss * (1 - self.label_smoothing_prob) + F.kl_div(
                log_probs, uniform,
                size_average=False, reduce=True) * self.label_smoothing_prob

        # Add coverage term
        if self.coverage_weight != 0:
            raise NotImplementedError

        # Auxiliary CTC loss (optional)
        if self.ctc_loss_weight > 0:
            ctc_loss = self.compute_ctc_loss(xs, ys, x_lens, y_lens)
            loss = loss * (1 - self.ctc_loss_weight) + \
                ctc_loss * self.ctc_loss_weight

        # Average the loss by mini-batch
        loss = loss / len(xs)

        if is_eval:
            loss = loss.data[0]
        else:
            self._step += 1

            # Update the probability of scheduled sampling
            if self.sample_prob > 0:
                self._sample_prob = min(
                    self.sample_prob,
                    self.sample_prob / self.sample_ramp_max_step * self._step)

        return loss

    def compute_ctc_loss(self, enc_out, ys, x_lens, y_lens, is_sub_task=False):
        """Compute CTC loss.
        Args:
            enc_out (FloatTensor): A tensor of size
                `[B, T_in, decoder_num_units]`
            ys (LongTensor): A tensor of size `[B, T_out]`
            x_lens (IntTensor): A tensor of size `[B]`
            y_lens (IntTensor): A tensor of size `[B]`
            is_sub_task (bool, optional):
        Returns:
            loss (FloatTensor): A tensor of size `[1]`
        """
        if is_sub_task:
            logits_ctc = self.fc_ctc_sub(enc_out)
        else:
            logits_ctc = self.fc_ctc(enc_out)

        # Convert to time-major
        logits_ctc = logits_ctc.transpose(0, 1)

        _x_lens = x_lens.clone()
        _ys = ys.clone()[:, 1:] + 1
        _y_lens = y_lens.clone() - 2
        # NOTE: index 0 is reserved for blank
        # NOTE: Ignore <SOS> and <EOS>

        # Concatenate all _ys for warpctc_pytorch
        # `[B, T_out]` -> `[1,]`
        concatenated_labels = _concatenate_labels(_ys, _y_lens)

        loss = ctc_loss(logits_ctc, concatenated_labels.cpu(),
                        _x_lens.cpu(), _y_lens.cpu())

        if self.use_cuda:
            loss = loss.cuda()

        return loss

    def _encode(self, xs, x_lens, volatile, is_multi_task=False):
        """Encode acoustic features.
        Args:
            xs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            x_lens (IntTensor): A tensor of size `[B]`
            volatile (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
            is_multi_task (bool, optional):
        Returns:
            xs (FloatTensor): A tensor of size `[B, T_in, decoder_num_units]`
            x_lens ():
            OPTION:
                xs_sub (FloatTensor): A tensor of size `[B, T_in, decoder_num_units]`
                x_lens_sub ():
            perm_idx (LongTensor):
        """
        if is_multi_task:
            xs, x_lens, xs_sub, x_lens_sub, perm_idx = self.encoder(
                xs, x_lens, volatile)
        else:
            if self.encoder_type == 'cnn':
                xs, x_lens = self.encoder(xs, x_lens)
                perm_idx = None
            else:
                xs, x_lens, perm_idx = self.encoder(xs, x_lens, volatile)
        # NOTE: xs: `[B, T_in, encoder_num_units * encoder_num_directions]`
        # xs_sub: `[B, T_in, encoder_num_units * encoder_num_directions]`

        # Bridge between the encoder and decoder in the main task
        if self.is_bridge:
            xs = self.bridge(xs)

        if is_multi_task:
            # Bridge between the encoder and decoder in the sub task
            if self.sub_loss_weight > 0 and self.is_bridge_sub:
                xs_sub = self.bridge_sub(xs_sub)
            return xs, x_lens, xs_sub, x_lens_sub, perm_idx
        else:
            return xs, x_lens, perm_idx

    def _compute_coverage(self, att_weights):
        batch_size, max_time_outputs, max_time_inputs = att_weights.size()
        raise NotImplementedError

    def _decode_train(self, enc_out, ys, is_sub_task=False):
        """Decoding in the training stage.
        Args:
            enc_out (FloatTensor): A tensor of size
                `[B, T_in, decoder_num_units]`
            ys (LongTensor): A tensor of size `[B, T_out]`
            is_sub_task (bool, optional):
        Returns:
            logits (FloatTensor): A tensor of size `[B, T_out, num_classes]`
            att_weights (FloatTensor): A tensor of size
                `[B, T_out, T_in]`
        """
        batch_size, max_time = enc_out.size()[:2]
        labels_max_seq_len = ys.size(1)

        # Initialize decoder state
        dec_state = self._init_decoder_state(enc_out)

        # Initialize attention weights
        # att_weights_step = Variable(torch.zeros(batch_size, max_time))
        att_weights_step = Variable(
            torch.ones(batch_size, max_time)) / max_time
        # NOTE: with uniform distribution

        # Initialize context vector
        context_vec = Variable(torch.zeros(batch_size, 1, enc_out.size(2)))

        # Initialize logits
        if self.decoder_input == 'onehot_prob':
            if is_sub_task:
                logits = Variable(torch.ones(
                    batch_size, 1, self.num_classes_sub))
            else:
                logits = Variable(torch.ones(batch_size, 1, self.num_classes))
            logits = logits / logits.size(2)
            if self.use_cuda:
                logits = logits.cuda()

        if self.use_cuda:
            att_weights_step = att_weights_step.cuda()
            context_vec = context_vec.cuda()

        logits = []
        att_weights = []
        for t in range(labels_max_seq_len - 1):

            is_sample = self.sample_prob > 0 and t > 0 and self._step > 0 and random.random(
            ) < self._sample_prob

            if is_sub_task:
                if is_sample:
                    y = torch.max(logits[-1], dim=2)[1]  # scheduled sampling
                else:
                    y = ys[:, t:t + 1]  # teacher-forcing

                if self.decoder_input == 'embedding':
                    y = self.embed_sub(y)
                else:
                    y = to_onehot(y, n_dims=self.num_classes_sub).unsqueeze(1)
            else:
                if is_sample:
                    y = torch.max(logits[-1], dim=2)[1]  # scheduled sampling
                else:
                    y = ys[:, t:t + 1]  # teacher-forcing
                if self.decoder_input == 'embedding':
                    y = self.embed(y)
                else:
                    y = to_onehot(y, n_dims=self.num_classes).unsqueeze(1)

            # Label smoothing
            if self.label_smoothing_prob > 0 and self.decoder_input != 'embedding':
                y = y * (1 - self.label_smoothing_prob) + 1 / \
                    y.size(2) * self.label_smoothing_prob

            if self.decoder_input == 'onehot_prob':
                prob = F.softmax(logits[-1], dim=-1)
                y = torch.cat([y, prob], dim=-1)

            dec_in = torch.cat([y, context_vec], dim=-1)
            dec_out, dec_state, context_vec, att_weights_step = self._decode_step(
                enc_out=enc_out,
                dec_in=dec_in,
                dec_state=dec_state,
                att_weights_step=att_weights_step,
                is_sub_task=is_sub_task)

            concat = torch.cat([dec_out, context_vec], dim=-1)
            if is_sub_task:
                attentional_vec = F.tanh(self.proj_layer_sub(concat))
                logits_step = self.fc_sub(attentional_vec)
            else:
                attentional_vec = F.tanh(self.proj_layer(concat))
                logits_step = self.fc(attentional_vec)

            logits.append(logits_step)
            att_weights.append(att_weights_step)

        # Concatenate in T_out-dimension
        logits = torch.cat(logits, dim=1)
        att_weights = torch.stack(att_weights, dim=1)
        # NOTE; att_weights in the training stage may be used for computing the
        # coverage, so do not convert to numpy yet.

        return logits, att_weights

    def _init_decoder_state(self, enc_out, volatile=False):
        """Initialize decoder state.
        Args:
            enc_out (FloatTensor): A tensor of size
                `[B, T_in, decoder_num_units]`
            volatile (bool, optional): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            dec_state (FloatTensor or tuple): A tensor of size
                `[1, B, decoder_num_units]`
        """
        batch_size = enc_out.size(0)

        if self.init_dec_state == 'zero' or self.encoder_type != self.decoder_type:
            # Initialize zero state
            h_0 = Variable(torch.zeros(1, batch_size, self.decoder_num_units))
            if volatile:
                h_0.volatile = True
            if self.use_cuda:
                h_0 = h_0.cuda()
        else:
            if self.init_dec_state == 'mean':
                # Initialize with mean of all encoder outputs
                h_0 = enc_out.mean(dim=1, keepdim=True)
            elif self.init_dec_state == 'final':
                # Initialize with the final encoder output (forward)
                h_0 = enc_out[:, -2:-1, :]

            # Convert to time-major
            h_0 = h_0.transpose(0, 1).contiguous()

        if self.decoder_type == 'lstm':
            c_0 = Variable(torch.zeros(1, batch_size, self.decoder_num_units))
            if volatile:
                c_0.volatile = True
            if self.use_cuda:
                c_0 = c_0.cuda()
            dec_state = (h_0, c_0)
        else:
            dec_state = h_0

        return dec_state

    def _decode_step(self, enc_out, dec_in, dec_state,
                     att_weights_step, is_sub_task=False):
        """Decoding step.
        Args:
            enc_out (FloatTensor): A tensor of size
                `[B, T_in, decoder_num_units]`
            dec_in (FloatTensor): A tensor of size
                `[B, 1, embedding_dim + decoder_num_units]`
            dec_state (FloatTensor or tuple): A tensor of size
                `[decoder_num_layers, B, decoder_num_units]`
            att_weights_step (FloatTensor): A tensor of size `[B, T_in]`
            is_sub_task (bool, optional):
        Returns:
            dec_out (FloatTensor): A tensor of size
                `[B, 1, decoder_num_units]`
            dec_state (FloatTensor): A tensor of size
                `[decoder_num_layers, B, decoder_num_units]`
            content_vec (FloatTensor): A tensor of size
                `[B, 1, decoder_num_units]`
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

    def _create_token(self, value, batch_size):
        """Create 1 token per batch dimension.
        Args:
            value (int): the  value to pad
            batch_size (int): the size of mini-batch
        Returns:
            y (LongTensor): A tensor of size `[B, 1]`
        """
        y = np.full((batch_size, 1), fill_value=value, dtype=np.int64)
        y = torch.from_numpy(y)
        y = Variable(y, requires_grad=False)
        y.volatile = True
        if self.use_cuda:
            y = y.cuda()
        return y

    def attention_weights(self, xs, x_lens, max_decode_len=100, is_sub_task=False):
        """Get attention weights for visualization.
        Args:
            xs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            x_lens (np.ndarray): A tensor of size `[B]`
            max_decode_len (int, optional): the length of output sequences
                to stop prediction when EOS token have not been emitted
            is_sub_task (bool, optional):
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B, T_out]`
            att_weights (np.ndarray): A tensor of size `[B, T_out, T_in]`
        """
        # Wrap by Variable
        xs = np2var(
            xs, use_cuda=self.use_cuda, volatile=True, backend='pytorch')
        x_lens = np2var(
            x_lens, dtype='int', use_cuda=self.use_cuda, volatile=True, backend='pytorch')

        # Change to evaluation mode
        self.eval()

        # Encode acoustic features
        if hasattr(self, 'main_loss_weight'):
            if is_sub_task:
                _, _, enc_out, _, perm_idx = self._encode(
                    xs, x_lens, volatile=True, is_multi_task=True)
            else:
                enc_out, _, _, _, perm_idx = self._encode(
                    xs, x_lens, volatile=True, is_multi_task=True)
        else:
            enc_out, _, perm_idx = self._encode(xs, x_lens, volatile=True)

        # Permutate indices
        if perm_idx is not None:
            perm_idx = var2np(perm_idx, backend='pytorch')

        # NOTE: assume beam_width == 1
        best_hyps, att_weights = self._decode_infer_greedy(
            enc_out, max_decode_len, is_sub_task=is_sub_task)

        # Permutate indices to the original order
        if perm_idx is not None:
            best_hyps = best_hyps[perm_idx]
            att_weights = att_weights[perm_idx]

        return best_hyps, att_weights

    def decode(self, xs, x_lens, beam_width, max_decode_len):
        """Decoding in the inference stage.
        Args:
            xs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            x_lens (np.ndarray): A tensor of size `[B]`
            beam_width (int): the size of beam
            max_decode_len (int): the length of output sequences
                to stop prediction when EOS token have not been emitted
        Returns:
            best_hyps (np.ndarray): A tensor of size `[]`
        """
        # Wrap by Variable
        xs = np2var(
            xs, use_cuda=self.use_cuda, volatile=True, backend='pytorch')
        x_lens = np2var(
            x_lens, dtype='int', use_cuda=self.use_cuda, volatile=True, backend='pytorch')

        # Change to evaluation mode
        self.eval()

        # Encode acoustic features
        enc_out, x_lens, perm_idx = self._encode(xs, x_lens, volatile=True)

        if beam_width == 1:
            best_hyps, _ = self._decode_infer_greedy(enc_out, max_decode_len)
        else:
            best_hyps = self._decode_infer_beam(
                enc_out, x_lens, beam_width, max_decode_len)

        # Permutate indices to the original order
        if perm_idx is not None:
            perm_idx = var2np(perm_idx, backend='pytorch')
            best_hyps = best_hyps[perm_idx]

        return best_hyps

    def _decode_infer_greedy(self, enc_out, max_decode_len, is_sub_task=False):
        """Greedy decoding in the inference stage.
        Args:
            enc_out (FloatTensor): A tensor of size
                `[B, T_in, decoder_num_units]`
            max_decode_len (int): the length of output sequences
                to stop prediction when EOS token have not been emitted
            is_sub_task (bool, optional):
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B, T_out]`
            att_weights (np.ndarray): A tensor of size `[B, T_out, T_in]`
        """
        batch_size, max_time = enc_out.size()[:2]

        # Initialize decoder state
        dec_state = self._init_decoder_state(enc_out, volatile=True)

        # Initialize attention weights
        # att_weights_step = Variable(torch.zeros(batch_size, max_time))
        att_weights_step = Variable(
            torch.ones(batch_size, max_time)) / max_time
        # NOTE: with uniform distribution
        att_weights_step.volatile = True

        # Initialize context vector
        context_vec = Variable(torch.zeros(batch_size, 1, enc_out.size(2)))
        context_vec.volatile = True

        # Initialize logits
        if self.decoder_input == 'onehot_prob':
            if is_sub_task:
                logits = Variable(torch.ones(
                    batch_size, 1, self.num_classes_sub))
            else:
                logits = Variable(torch.ones(batch_size, 1, self.num_classes))
            logits = logits / logits.size(2)
            logits.volatile = True
            if self.use_cuda:
                logits = logits.cuda()

        if self.use_cuda:
            att_weights_step = att_weights_step.cuda()
            context_vec = context_vec.cuda()

        # Start from <SOS>
        sos = self.sos_index_sub if is_sub_task else self.sos_index
        eos = self.eos_index_sub if is_sub_task else self.eos_index
        y = self._create_token(value=sos, batch_size=batch_size)

        best_hyps = []
        att_weights = []
        for _ in range(max_decode_len):

            if is_sub_task:
                if self.decoder_input == 'embedding':
                    y = self.embed_sub(y)
                else:
                    y = to_onehot(y, n_dims=self.num_classes_sub).unsqueeze(1)
            else:
                if self.decoder_input == 'embedding':
                    y = self.embed(y)
                else:
                    y = to_onehot(y, n_dims=self.num_classes).unsqueeze(1)
            if self.decoder_input == 'onehot_prob':
                prob = F.softmax(logits, dim=-1)
                y = torch.cat([y, prob], dim=-1)

            dec_in = torch.cat([y, context_vec], dim=-1)
            dec_out, dec_state, context_vec, att_weights_step = self._decode_step(
                enc_out=enc_out,
                dec_in=dec_in,
                dec_state=dec_state,
                att_weights_step=att_weights_step,
                is_sub_task=is_sub_task)

            concat = torch.cat([dec_out, context_vec], dim=-1)
            if is_sub_task:
                attentional_vec = F.tanh(self.proj_layer_sub(concat))
                logits = self.fc_sub(attentional_vec)
            else:
                attentional_vec = F.tanh(self.proj_layer(concat))
                logits = self.fc(attentional_vec)

            # Pick up 1-best
            _, y = torch.max(logits.squeeze(1), dim=1)
            # logits: `[B, 1, num_classes]` -> `[B, num_classes]`
            y = y.unsqueeze(1)
            best_hyps.append(y)
            att_weights.append(att_weights_step)

            # Break if <EOS> is outputed in all mini-batch
            if torch.sum(y.data == eos) == y.numel():
                break

        # Concatenate in T_out dimension
        best_hyps = torch.cat(best_hyps, dim=1)
        att_weights = torch.stack(att_weights, dim=1)

        # Convert to numpy
        best_hyps = var2np(best_hyps, backend='pytorch')
        att_weights = var2np(att_weights, backend='pytorch')

        return best_hyps, att_weights

    def _decode_infer_beam(self, enc_out, x_lens,
                           beam_width, max_decode_len, is_sub_task=False):
        """Beam search decoding in the inference stage.
        Args:
            enc_out (FloatTensor): A tensor of size
                `[B, T_in, decoder_num_units]`
            x_lens (IntTensor): A tensor of size `[B]`
            beam_width (int): the size of beam
            max_decode_len (int, optional): the length of output sequences
                to stop prediction when EOS token have not been emitted
            is_sub_task (bool, optional):
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B]`
        """
        batch_size = enc_out.size(0)

        # Start from <SOS>
        sos = self.sos_index_sub if is_sub_task else self.sos_index
        eos = self.eos_index_sub if is_sub_task else self.eos_index

        best_hyps = []
        for i_batch in range(batch_size):

            max_time = x_lens[i_batch].data[0]

            # Initialize decoder state
            dec_state = self._init_decoder_state(
                enc_out[i_batch:i_batch + 1, :, :], volatile=True)

            # Initialize attention weights
            # att_weights_step = Variable(torch.zeros(1, max_time))
            att_weights_step = Variable(torch.ones(1, max_time)) / max_time
            # NOTE: with uniform distribution
            att_weights_step.volatile = True

            # Initialize context vector
            context_vec = Variable(torch.zeros(1, 1, enc_out.size(2)))
            context_vec.volatile = True

            # Initialize logits
            if self.decoder_input == 'onehot_prob':
                if is_sub_task:
                    logits = Variable(torch.ones(1, 1, self.num_classes_sub))
                else:
                    logits = Variable(torch.ones(1, 1, self.num_classes))
                logits = logits / logits.size(2)
                logits.volatile = True
                if self.use_cuda:
                    logits = logits.cuda()

            if self.use_cuda:
                att_weights_step = att_weights_step.cuda()
                context_vec = context_vec.cuda()

            complete = []
            beam = [{'hyp': [],
                     'score': LOG_1,
                     'dec_state': dec_state,
                     'att_weights_step': att_weights_step,
                     'context_vec': context_vec}]
            for t in range(max_decode_len):
                new_beam = []
                for i_beam in range(len(beam)):
                    idx_prev = beam[i_beam]['hyp'][-1] if t > 0 else sos
                    y = self._create_token(value=idx_prev, batch_size=1)

                    if is_sub_task:
                        if self.decoder_input == 'embedding':
                            y = self.embed_sub(y)
                        else:
                            y = to_onehot(
                                y, n_dims=self.num_classes_sub).unsqueeze(1)
                    else:
                        if self.decoder_input == 'embedding':
                            y = self.embed(y)
                        else:
                            y = to_onehot(
                                y, n_dims=self.num_classes).unsqueeze(1)
                    if self.decoder_input == 'onehot_prob':
                        prob = F.softmax(logits[-1], dim=-1)
                        y = torch.cat([y, prob], dim=-1)

                    dec_in = torch.cat(
                        [y, beam[i_beam]['context_vec']], dim=-1)
                    dec_out, dec_state, context_vec, att_weights_step = self._decode_step(
                        enc_out=enc_out[i_batch:i_batch + 1, :max_time],
                        dec_in=dec_in,
                        dec_state=beam[i_beam]['dec_state'],
                        att_weights_step=beam[i_beam]['att_weights_step'],
                        is_sub_task=is_sub_task)

                    concat = torch.cat([dec_out, context_vec], dim=-1)
                    if is_sub_task:
                        attentional_vec = F.tanh(self.proj_layer_sub(concat))
                        logits = self.fc_sub(attentional_vec)
                    else:
                        attentional_vec = F.tanh(self.proj_layer(concat))
                        logits = self.fc(attentional_vec)

                    # Path through the softmax layer & convert to log-scale
                    log_probs = F.log_softmax(logits.squeeze(1), dim=-1)
                    # NOTE: `[1 (B), 1, num_classes]` -> `[1 (B), num_classes]`

                    # Pick up the top-k scores
                    log_probs_topk, indices_topk = torch.topk(
                        log_probs, k=beam_width, dim=-1,
                        largest=True, sorted=True)

                    for i, log_prob in zip(indices_topk.data[0], log_probs_topk.data[0]):
                        new_hyp = beam[i_beam]['hyp'] + [i]

                        # numpy
                        new_score = np.logaddexp(
                            beam[i_beam]['score'], log_prob)

                        # torch
                        # new_score = _logsumexp(
                        #     [beam[i_beam]['score'], log_prob], dim=0)

                        new_beam.append({'hyp': new_hyp,
                                         'score': new_score,
                                         'dec_state': dec_state,
                                         'att_weights_step': att_weights_step,
                                         'context_vec': context_vec})

                new_beam = sorted(
                    new_beam, key=lambda x: x['score'], reverse=True)

                # Remove complete hypotheses
                for cand in new_beam[:beam_width]:
                    if cand['hyp'][-1] == eos:
                        complete.append(cand)
                if len(complete) >= beam_width:
                    complete = complete[:beam_width]
                    break
                beam = list(filter(lambda x: x['hyp'][-1] != eos, new_beam))
                beam = beam[:beam_width]

            complete = sorted(
                complete, key=lambda x: x['score'], reverse=True)
            if len(complete) == 0:
                complete = beam
            best_hyps.append(np.array(complete[0]['hyp']))

        return np.array(best_hyps)

    def decode_ctc(self, xs, x_lens, beam_width=1):
        """Decoding by the CTC layer in the inference stage.
            This is only used for Joint CTC-Attention model.
        Args:
            xs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            x_lens (np.ndarray): A tensor of size `[B]`
            beam_width (int, optional): the size of beam
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B]`
        """
        assert self.ctc_loss_weight > 0
        # TODO: add is_sub_task??

        # Wrap by Variable
        xs = np2var(
            xs, use_cuda=self.use_cuda, volatile=True, backend='pytorch')
        x_lens = np2var(
            x_lens, dtype='int', use_cuda=self.use_cuda, volatile=True, backend='pytorch')

        # Change to evaluation mode
        self.eval()

        # Encode acoustic features
        enc_out, x_lens, perm_idx = self._encode(xs, x_lens, volatile=True)

        # Path through the softmax layer
        batch_size, max_time = enc_out.size()[:2]
        enc_out = enc_out.view(batch_size * max_time, -1).contiguous()
        logits_ctc = self.fc_ctc(enc_out)
        logits_ctc = logits_ctc.view(batch_size, max_time, -1)
        log_probs = F.log_softmax(logits_ctc, dim=-1)

        if beam_width == 1:
            best_hyps = self._decode_ctc_greedy_np(
                var2np(log_probs, backend='pytorch'),
                var2np(x_lens, backend='pytorch'))
        else:
            best_hyps = self._decode_ctc_beam_np(
                var2np(log_probs, backend='pytorch'),
                var2np(x_lens, backend='pytorch'), beam_width=beam_width)

        best_hyps = best_hyps - 1
        # NOTE: index 0 is reserved for blank in warpctc_pytorch

        # Permutate indices to the original order
        if perm_idx is not None:
            perm_idx = var2np(perm_idx, backend='pytorch')
            best_hyps = best_hyps[perm_idx]

        return best_hyps


def to_onehot(y, n_dims):
    """Convert indices into one-hot encoding.
    Args:
        y (Variable): A tensor of size `[B, 1]`
        n_dims (int):
    Returns:
        y (Variable): A tensor of size `[B, ]`
    """
    batch_size = y.size(0)
    y_onehot = torch.FloatTensor(batch_size, n_dims).zero_()
    y_onehot.scatter_(1, y.data.cpu(), 1)
    y_onehot = Variable(y_onehot)
    if y.is_cuda:
        y_onehot = y_onehot.cuda()
    # if y.volatile:
    #     y_onehot.volatile = True
    return y_onehot


def _logsumexp(x, dim=None):
    """
    Args:
        x (list):
        dim (int, optional):
    Returns:
        (int) the summation of x in the log-scale
    """
    if dim is None:
        raise ValueError
        # TODO: fix this

    if isinstance(x, list):
        x = torch.FloatTensor(x)

    max_val, _ = torch.max(x, dim=dim)
    max_val += torch.log(torch.sum(torch.exp(x - max_val),
                                   dim=dim, keepdim=True))

    return torch.squeeze(max_val, dim=dim).numpy().tolist()[0]
