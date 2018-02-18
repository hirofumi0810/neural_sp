#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention-based sequence-to-sequence model (chainer)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np

import chainer
from chainer import functions as F
from chainer import Variable
from chainer import cuda
from models.chainer.ctc.ctc_loss_from_chainer import connectionist_temporal_classification

from models.chainer.base import ModelBase
from models.chainer.linear import LinearND, Embedding, Embedding_LS
from models.chainer.criterion import cross_entropy_label_smoothing
from models.chainer.encoders.load_encoder import load
from models.chainer.attention.rnn_decoder import RNNDecoder
from models.chainer.attention.attention_layer import AttentionMechanism
from models.pytorch.ctc.decoders.greedy_decoder import GreedyDecoder
from models.pytorch.ctc.decoders.beam_search_decoder import BeamSearchDecoder
from utils.io.variable import np2var, var2np

LOG_1 = 0


class AttentionSeq2seq(ModelBase):
    """Attention-based sequence-to-sequence model.
    Args:
        input_size (int): the dimension of input features
        encoder_type (string): the type of the encoder. Set lstm or gru or rnn.
        encoder_bidirectional (bool): if True, create a bidirectional encoder
        encoder_num_units (int): the number of units in each layer of the encoder
        encoder_num_proj (int): the number of nodes in the projection layer of the encoder
        encoder_num_layers (int): the number of layers of the encoder
        attention_type (string): the type of attention
        attention_dim: (int) the dimension of the attention layer
        decoder_type (string): lstm or gru
        decoder_num_units (int): the number of units in each layer of the decoder
        decoder_num_layers (int): the number of layers of the decoder
        embedding_dim (int): the dimension of the embedding in target spaces.
            0 means that decoder inputs are represented by one-hot vectors.
        dropout_input (float): the probability to drop nodes in input-hidden connection
        dropout_encoder (float): the probability to drop nodes in hidden-hidden
            connection of the encoder
        dropout_decoder (float): the probability to drop nodes of the decoder
        dropout_embedding (float): the probability to drop nodes of the embedding layer
        num_classes (int): the number of nodes in softmax layer
            (excluding <SOS> and <EOS> classes)
        parameter_init_distribution (string, optional): uniform or normal or
            orthogonal or constant distribution
        parameter_init (float, optional): Range of uniform distribution to
            initialize weight parameters
        recurrent_weight_orthogonal (bool, optional): if True, recurrent
            weights are orthogonalized
        init_forget_gate_bias_with_one (bool, optional): if True, initialize
            the forget gate bias with 1
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
                 attention_type,
                 attention_dim,
                 decoder_type,
                 decoder_num_units,
                 decoder_num_layers,
                 embedding_dim,
                 dropout_input,
                 dropout_encoder,
                 dropout_decoder,
                 dropout_embedding,
                 num_classes,
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
                 ctc_loss_weight=0,
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
                 decoder_dense_residual=False):

        super(ModelBase, self).__init__()
        self.model_type = 'attention'

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
        self.sos_index = 0
        self.eos_index = num_classes + 1

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

        with self.init_scope():
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
            elif encoder_type == 'cnn':
                assert num_stack == 1 and splice == 1
                self.encoder = load(encoder_type='cnn')(
                    input_size=input_size,
                    conv_channels=conv_channels,
                    conv_kernel_sizes=conv_kernel_sizes,
                    conv_strides=conv_strides,
                    poolings=poolings,
                    dropout_input=dropout_input,
                    dropout_hidden=dropout_encoder,
                    use_cuda=self.use_cuda,
                    activation=activation,
                    batch_norm=batch_norm)
                self.init_dec_state = 'zero'
            else:
                raise NotImplementedError

            if encoder_type != decoder_type:
                self.init_dec_state = 'zero'

            if self.init_dec_state != 'zero':
                self.W_dec_init = LinearND(
                    decoder_num_units, decoder_num_units)

            ####################
            # Decoder
            ####################
            self.decoder = RNNDecoder(
                input_size=decoder_num_units + embedding_dim,
                rnn_type=decoder_type,
                num_units=decoder_num_units,
                num_layers=decoder_num_layers,
                dropout=dropout_decoder,
                use_cuda=self.use_cuda,
                residual=decoder_residual,
                dense_residual=decoder_dense_residual)

            ##############################
            # Attention layer
            ##############################
            self.attend = AttentionMechanism(
                decoder_num_units=decoder_num_units,
                attention_type=attention_type,
                attention_dim=attention_dim,
                use_cuda=self.use_cuda,
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
                        dropout=dropout_encoder, use_cuda=self.use_cuda)
                elif encoder_bidirectional:
                    self.bridge = LinearND(
                        encoder_num_units * 2, decoder_num_units,
                        dropout=dropout_encoder, use_cuda=self.use_cuda)
                else:
                    self.bridge = LinearND(
                        encoder_num_units, decoder_num_units,
                        dropout=dropout_encoder, use_cuda=self.use_cuda)
                self.is_bridge = True
            else:
                self.is_bridge = False

            if label_smoothing_prob > 0:
                self.embed = Embedding_LS(
                    num_classes=self.num_classes,
                    embedding_dim=embedding_dim,
                    dropout=dropout_embedding,
                    label_smoothing_prob=label_smoothing_prob,
                    use_cuda=self.use_cuda)
            else:
                self.embed = Embedding(num_classes=self.num_classes,
                                       embedding_dim=embedding_dim,
                                       dropout=dropout_embedding,
                                       ignore_index=self.sos_index,
                                       use_cuda=self.use_cuda)

            self.proj_layer = LinearND(
                decoder_num_units * 2, decoder_num_units,
                dropout=dropout_decoder, use_cuda=self.use_cuda)
            self.fc = LinearND(decoder_num_units, self.num_classes,
                               use_cuda=self.use_cuda)

            if ctc_loss_weight > 0:
                if self.is_bridge:
                    self.fc_ctc = LinearND(decoder_num_units, num_classes + 1,
                                           use_cuda=self.use_cuda)
                else:
                    self.fc_ctc = LinearND(
                        encoder_num_units * self.encoder_num_directions, num_classes + 1,
                        use_cuda=self.use_cuda)

                self.blank_index = 0
                # NOTE: do not change

                # Set CTC decoders
                self._decode_ctc_greedy_np = GreedyDecoder(
                    blank_index=self.blank_index)
                self._decode_ctc_beam_np = BeamSearchDecoder(
                    blank_index=self.blank_index)
                # TODO: set space index

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
                if encoder_type != 'cnn':
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

    def __call__(self, xs, ys, x_lens, y_lens, is_eval=False):
        """Forward computation.
        Args:
            xs (list of np.ndarray): A tensor of size `[B, T_in, input_size]`
            ys (np.ndarray): A tensor of size `[B, T_out]`
            x_lens (np.ndarray): A tensor of size `[B]`
            y_lens (np.ndarray): A tensor of size `[B]`
            is_eval (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (chainer.Variable(float) or float): A tensor of size `[1]`
        """
        if is_eval:
            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                loss = self._forward(xs, ys, x_lens, y_lens).data
        else:
            loss = self._forward(xs, ys, x_lens, y_lens)

            # Update the probability of scheduled sampling
            self._step += 1
            if self.sample_prob > 0:
                self._sample_prob = min(
                    self.sample_prob,
                    self.sample_prob / self.sample_ramp_max_step * self._step)

        return loss

    def _forward(self, xs, ys, x_lens, y_lens):
        # Wrap by Variable
        xs = np2var(xs,  use_cuda=self.use_cuda, backend='chainer')
        ys = np2var(ys, use_cuda=self.use_cuda, backend='chainer')
        # x_lens = np2var(x_lens, use_cuda=self.use_cuda, backend='chainer')
        y_lens = np2var(y_lens, use_cuda=self.use_cuda, backend='chainer')

        # NOTE: index 0 is reserved for blank and <SOS>
        ys = ys + 1

        # Encode acoustic features
        xs, x_lens = self._encode(xs, x_lens)

        # Teacher-forcing
        logits, att_weights = self._decode_train(xs, x_lens, ys)

        # Output smoothing
        if self.logits_temperature != 1:
            logits /= self.logits_temperature

        # Compute XE sequence loss
        loss = F.softmax_cross_entropy(
            x=logits.reshape((-1, logits.shape[2])),
            t=ys[:, 1:].reshape(-1),  # NOTE: Exclude first <SOS>
            normalize=True, cache_score=True, class_weight=None,
            ignore_label=self.sos_index, reduce='no')
        # NOTE: ys are padded by <SOS>
        # NOTE: len(loss) = batch_size * max_time
        loss = F.sum(loss, axis=0) / len(xs)

        # Label smoothing (with uniform distribution)
        if self.label_smoothing_prob > 0:
            loss_ls = cross_entropy_label_smoothing(
                logits,
                y_lens=y_lens - 1,  # Exclude <SOS>
                label_smoothing_prob=self.label_smoothing_prob,
                distribution='uniform',
                size_average=False) / len(xs)
            loss = loss * (1 - self.label_smoothing_prob) + loss_ls
            # print(loss_ls)

        # Add coverage term
        if self.coverage_weight != 0:
            raise NotImplementedError

        # Auxiliary CTC loss (optional)
        if self.ctc_loss_weight > 0:
            x_lens = np2var(x_lens, use_cuda=self.use_cuda, backend='chainer')

            logits_ctc = self.fc_ctc(xs)
            ctc_loss = self.compute_ctc_loss(
                logits_ctc, ys, x_lens, y_lens, size_average=False) / len(xs)
            loss = loss * (1 - self.ctc_loss_weight) + \
                ctc_loss * self.ctc_loss_weight

        return loss

    def compute_ctc_loss(self, logits_ctc, ys, x_lens, y_lens,
                         size_average=False):
        """Compute CTC loss.
        Args:
            logits_ctc (chainer.Variable, float): A tensor of size
                `[B, T_in, decoder_num_units]`
            ys (chainer.Variable, int): A tensor of size `[B, T_out]`
            x_lens (chainer.Variable, int): A tensor of size `[B]`
            y_lens (chainer.Variable, int): A tensor of size `[B]`
            size_average (bool, optional):
        Returns:
            ctc_loss (chainer.Variable, float):
                if size_average is True, A tensor of size `[1]`
        """
        # Compute CTC loss
        ctc_loss = connectionist_temporal_classification(
            x=F.separate(logits_ctc, axis=1),  # list of Variable
            t=ys[:, 1:-1],  # NOTE: Exclude first <SOS> & last <EOS>
            blank_symbol=self.blank_index,
            input_length=x_lens,  # Variable
            label_length=y_lens - 2,  # NOTE: Ignore <SOS> & <EOS>
            reduce='no')
        ctc_loss = F.sum(ctc_loss, axis=0)

        # Label smoothing (with uniform distribution)
        if self.label_smoothing_prob > 0:
            # XE
            loss_ls_ctc = cross_entropy_label_smoothing(
                logits_ctc,
                y_lens=x_lens,  # NOTE: CTC is frame-synchronous
                label_smoothing_prob=self.label_smoothing_prob,
                distribution='uniform',
                size_average=False)
            ctc_loss = ctc_loss * (1 - self.label_smoothing_prob) + \
                loss_ls_ctc
            # print(loss_ls_ctc)

        if size_average:
            ctc_loss /= len(x_lens)

        return ctc_loss

    def _encode(self, xs, x_lens, is_multi_task=False):
        """Encode acoustic features.
        Args:
            xs (list of chainer.Variable(float)):
                A list of tensors of size `[T_in, input_size]`
            x_lens (np.ndarray): A tensor of size `[B]`
            is_multi_task (bool, optional):
        Returns:
            xs (chainer.Variable, float): A tensor of size
                `[B, T_in, decoder_num_units]`
            x_lens (np.ndarray): A tensor of size `[B]`
            OPTION:
                xs_sub (chainer.Variable, float): A tensor of size
                    `[B, T_in, decoder_num_units]`
                x_lens_sub (np.ndarray): A tensor of size `[B]`
        """
        if is_multi_task:
            xs, x_lens, xs_sub, x_lens_sub = self.encoder(xs, x_lens)
        else:
            xs, x_lens = self.encoder(xs, x_lens)
        # NOTE: xs: `[B, T_in, encoder_num_units * encoder_num_directions]`
        # xs_sub: `[B, T_in, encoder_num_units * encoder_num_directions]`

        # Concatenate
        xs = F.pad_sequence(xs, padding=0)

        # Bridge between the encoder and decoder in the main task
        if self.is_bridge:
            xs = self.bridge(xs)

        if is_multi_task:
            # Concatenate
            xs_sub = F.pad_sequence(xs_sub, padding=0)

            # Bridge between the encoder and decoder in the sub task
            if self.sub_loss_weight > 0 and self.is_bridge_sub:
                xs_sub = self.bridge_sub(xs_sub)
            return xs, x_lens, xs_sub, x_lens_sub
        else:
            return xs, x_lens

    def _compute_coverage(self, att_weights):
        batch_size, max_time_outputs, max_time_inputs = att_weights.shape
        raise NotImplementedError

    def _decode_train(self, enc_out, x_lens, ys, is_sub_task=False):
        """Decoding in the training stage.
        Args:
            enc_out (chainer.Variable, float): A tensor of size
                `[B, T_in, decoder_num_units]`
            x_lens (np.ndarray): A tensor of size `[B]`
            ys (chainer.Variable, int): A tensor of size `[B, T_out]`
            is_sub_task (bool, optional):
        Returns:
            logits (chainer.Variable, float): A tensor of size `[B, T_out, num_classes]`
            att_weights (chainer.Variable, float): A tensor of size
                `[B, T_out, T_in]`
        """
        batch_size, max_time, decoder_num_units = enc_out.shape
        labels_max_seq_len = ys.shape[1]

        # Initialize decoder state, decoder output, attention_weights
        dec_state = self._init_decoder_state(enc_out, is_sub_task=is_sub_task)
        dec_out = self._zero_init(
            (batch_size, 1, decoder_num_units), dtype=np.float32)
        context_vec = self._zero_init(
            (batch_size, 1, decoder_num_units), dtype=np.float32)
        att_weights_step = self._zero_init(
            (batch_size, max_time), dtype=np.float32)

        logits = []
        att_weights = []
        for t in range(labels_max_seq_len - 1):

            is_sample = self.sample_prob > 0 and t > 0 and self._step > 0 and random.random(
            ) < self._sample_prob

            if is_sub_task:
                if is_sample:
                    # scheduled sampling
                    y = self.embed_sub(F.argmax(logits[-1], axis=2))
                else:
                    # teacher-forcing
                    y = self.embed_sub(ys[:, t:t + 1])
            else:
                if is_sample:
                    # scheduled sampling
                    y = self.embed(F.argmax(logits[-1], axis=2))
                else:
                    # teacher-forcing
                    y = self.embed(ys[:, t:t + 1])

            dec_out, dec_state, context_vec, att_weights_step = self._decode_step(
                enc_out=enc_out,
                x_lens=x_lens,
                y=y,
                dec_state=dec_state,
                dec_out=dec_out,
                context_vec=context_vec,
                att_weights_step=att_weights_step,
                is_sub_task=is_sub_task)

            concat = F.concat([dec_out, context_vec], axis=-1)
            if is_sub_task:
                # logits_step =
                # self.fc_sub(F.tanh(self.proj_layer_sub(concat)))
                logits_step = self.fc_sub(self.proj_layer_sub(concat))
            else:
                # logits_step = self.fc(F.tanh(self.proj_layer(concat)))
                logits_step = self.fc(self.proj_layer(concat))

            logits.append(logits_step)
            att_weights.append(att_weights_step)

        # Concatenate in T_out-dimension
        logits = F.concat(logits, axis=1)
        att_weights = F.concat(att_weights, axis=1)
        # NOTE; att_weights in the training stage may be used for computing the
        # coverage, so do not convert to numpy yet.

        return logits, att_weights

    def _decode_step(self, enc_out, x_lens, y, dec_state, dec_out,
                     context_vec, att_weights_step, is_sub_task=False):
        """Decoding step.
        Args:
            enc_out (chainer.Variable, float): A tensor of size
                `[B, T_in, decoder_num_units]`
            x_lens (np.ndarray): A tensor of size `[B]`
            y (chainer.Variable, float): A tensor of size
                `[B, 1, embedding_dim]`
            dec_state (chainer.Variable(float) or tuple): A tensor of size
                `[decoder_num_layers, B, decoder_num_units]`
            dec_out (chainer.Variable, float): A tensor of size
                `[B, 1, decoder_num_units]`
            content_vector (chainer.Variable, float): A tensor of size
                `[B, 1, decoder_num_units]`
            att_weights_step (chainer.Variable, float): A tensor of size `[B, T_in]`
            is_sub_task (bool, optional):
        Returns:
            dec_out (chainer.Variable, float): A tensor of size
                `[B, 1, decoder_num_units]`
            dec_state (chainer.Variable, float): A tensor of size
                `[decoder_num_layers, B, decoder_num_units]`
            content_vector (chainer.Variable, float): A tensor of size
                `[B, 1, decoder_num_units]`
            att_weights_step (chainer.Variable, float): A tensor of size `[B, T_in]`
        """
        if is_sub_task:
            context_vec, att_weights_step = self.attend_sub(
                enc_out, x_lens, dec_out, att_weights_step)
            dec_in = F.concat([y, context_vec], axis=-1)
            dec_out, dec_state = self.decoder_sub(dec_in, dec_state)
        else:
            context_vec, att_weights_step = self.attend(
                enc_out, x_lens, dec_out, att_weights_step)
            dec_in = F.concat([y, context_vec], axis=-1)
            dec_out, dec_state = self.decoder(dec_in, dec_state)

        return dec_out, dec_state, context_vec, att_weights_step

    def _zero_init(self, size, dtype=np.float32):
        """Initialize a variable with zero.
        Args:
            size (tuple):
            dtype ():
        Returns:
            zero_var (chainer.Variable, float):
        """
        zero_var = Variable(np.zeros(size, dtype=dtype))
        if self.use_cuda:
            zero_var.to_gpu()
        return zero_var

    def _init_decoder_state(self, enc_out, is_sub_task=False):
        """Initialize decoder state.
        Args:
            enc_out (chainer.Variable, float): A tensor of size
                `[B, T_in, decoder_num_units]`
            is_sub_task (bool, optional):
        Returns:
            dec_state (chainer.Variable(float) or tuple): A tensor of size
                `[1, B, decoder_num_units]`
        """
        if self.init_dec_state == 'zero' or self.encoder_type != self.decoder_type:
            # Initialize with zero state
            h_0 = None
        else:
            if self.init_dec_state == 'mean':
                # Initialize with mean of all encoder outputs
                h_0 = F.mean(enc_out, axis=1, keepdims=True)
            elif self.init_dec_state == 'final':
                if self.encoder_bidirectional:
                    # Initialize with the final encoder output (backward)
                    h_0 = F.expand_dims(enc_out[:, -1, :], axis=1)
                else:
                    # Initialize with the final encoder output (forward)
                    h_0 = F.expand_dims(enc_out[:, -2, :], axis=1)

            # Path through the linear layer
            if is_sub_task:
                h_0 = self.W_dec_init_sub(h_0)
            else:
                h_0 = self.W_dec_init(h_0)

            # Convert to time-major
            h_0 = h_0.transpose(1, 0, 2)

        if self.decoder_type == 'lstm':
            # Initialize memory cell with zero state
            c_0 = None
            dec_state = (h_0, c_0)
        else:
            dec_state = h_0

        return dec_state

    def _create_token(self, value, batch_size):
        """Create 1 token per batch dimension.
        Args:
            value (int): the  value to pad
            batch_size (int): the size of mini-batch
        Returns:
            y (chainer.Variable, int): A tensor of size `[B, 1]`
        """
        y = Variable(
            np.full((batch_size, 1), fill_value=value, dtype=np.int32))
        if self.use_cuda:
            y.to_gpu()
        return y

    def attention_weights(self, xs, x_lens, max_decode_len, is_sub_task=False):
        """Get attention weights for visualization.
        Args:
            xs (list of np.ndarray): A tensor of size `[B, T_in, input_size]`
            x_lens (np.ndarray): A tensor of size `[B]`
            max_decode_len (int): the length of output sequences
                to stop prediction when EOS token have not been emitted
            is_sub_task (bool, optional):
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B, T_out]`
            att_weights (np.ndarray): A tensor of size `[B, T_out, T_in]`
        """
        with chainer.no_backprop_mode(), chainer.using_config('train', False):

            # Wrap by Variable
            xs = np2var(xs, use_cuda=self.use_cuda, backend='chainer')

            # Encode acoustic features
            if hasattr(self, 'main_loss_weight'):
                if is_sub_task:
                    _, _, enc_out, _, = self._encode(
                        xs, x_lens, is_multi_task=True)
                else:
                    enc_out, _, _, _ = self._encode(
                        xs, x_lens, is_multi_task=True)
            else:
                enc_out, _ = self._encode(xs, x_lens)

            # NOTE: assume beam_width == 1
            best_hyps, att_weights = self._decode_infer_greedy(
                enc_out, x_lens, max_decode_len, is_sub_task=is_sub_task)

        return best_hyps, att_weights

    def decode(self, xs, x_lens, beam_width, max_decode_len):
        """Decoding in the inference stage.
        Args:
            xs (list of np.ndarray): A tensor of size `[B, T_in, input_size]`
            x_lens (np.ndarray): A tensor of size `[B]`
            beam_width (int): the size of beam
            max_decode_len (int): the length of output sequences
                to stop prediction when EOS token have not been emitted
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B]`
            perm_idx (np.ndarray): For interface with pytorch, not used
        """
        with chainer.no_backprop_mode(), chainer.using_config('train', False):

            # Wrap by Variable
            xs = np2var(xs, use_cuda=self.use_cuda, backend='chainer')

            # Encode acoustic features
            enc_out, x_lens = self._encode(xs, x_lens)

            if beam_width == 1:
                best_hyps, _ = self._decode_infer_greedy(
                    enc_out, x_lens, max_decode_len)
            else:
                best_hyps = self._decode_infer_beam(
                    enc_out, x_lens, beam_width, max_decode_len)

        # NOTE: index 0 is reserved for <SOS>
        best_hyps -= 1

        perm_idx = np.arange(0, len(xs), 1)
        return best_hyps, perm_idx

    def _decode_infer_greedy(self, enc_out, x_lens, max_decode_len, is_sub_task=False):
        """Greedy decoding in the inference stage.
        Args:
            enc_out (chainer.Variable, float): A tensor of size
                `[B, T_in, decoder_num_units]`
            x_lens (np.ndarray): A tensor of size `[B]`
            max_decode_len (int): the length of output sequences
                to stop prediction when EOS token have not been emitted
            is_sub_task (bool, optional):
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B, T_out]`
            att_weights (np.ndarray): A tensor of size `[B, T_out, T_in]`
        """
        batch_size, max_time, decoder_num_units = enc_out.shape

        # Initialize decoder state, decoder output, attention_weights
        dec_state = self._init_decoder_state(enc_out, is_sub_task=is_sub_task)
        dec_out = self._zero_init(
            (batch_size, 1, decoder_num_units), dtype=np.float32)
        context_vec = self._zero_init(
            (batch_size, 1, decoder_num_units), dtype=np.float32)
        att_weights_step = self._zero_init(
            (batch_size, max_time), dtype=np.float32)

        # Start from <SOS>
        sos = self.sos_index_sub if is_sub_task else self.sos_index
        eos = self.eos_index_sub if is_sub_task else self.eos_index
        y = self._create_token(value=sos, batch_size=batch_size)

        best_hyps = []
        att_weights = []
        for _ in range(max_decode_len):

            if is_sub_task:
                y = self.embed_sub(y)
            else:
                y = self.embed(y)

            dec_out, dec_state, context_vec, att_weights_step = self._decode_step(
                enc_out=enc_out,
                x_lens=x_lens,
                y=y,
                dec_state=dec_state,
                dec_out=dec_out,
                context_vec=context_vec,
                att_weights_step=att_weights_step,
                is_sub_task=is_sub_task)

            concat = F.concat([dec_out, context_vec], axis=-1)
            if is_sub_task:
                # logits = self.fc_sub( F.tanh(self.proj_layer_sub(concat)))
                logits = self.fc_sub(self.proj_layer_sub(concat))
            else:
                # logits = self.fc( F.tanh(self.proj_layer(concat)))
                logits = self.fc(self.proj_layer(concat))

            # Pick up 1-best
            y = F.argmax(F.squeeze(logits, axis=1), axis=1)
            # logits: `[B, 1, num_classes]` -> `[B, num_classes]
            y = F.expand_dims(y, axis=1)
            best_hyps.append(y)
            att_weights.append(att_weights_step)

            # Break if <EOS> is outputed in all mini-batch
            if sum(y.data == eos)[0] == len(y):
                break

        # Concatenate in T_out dimension
        best_hyps = F.concat(best_hyps, axis=1)
        att_weights = F.concat(att_weights, axis=1)

        # Convert to numpy
        best_hyps = var2np(best_hyps, backend='chainer')
        att_weights = var2np(att_weights, backend='chainer')

        return best_hyps, att_weights

    def _decode_infer_beam(self, enc_out, x_lens,
                           beam_width, max_decode_len, is_sub_task=False):
        """Beam search decoding in the inference stage.
        Args:
            enc_out (chainer.Variable, float): A tensor of size
                `[B, T_in, decoder_num_units]`
            x_lens (np.ndarray): A tensor of size `[B]`
            beam_width (int): the size of beam
            max_decode_len (int, optional): the length of output sequences
                to stop prediction when EOS token have not been emitted
            is_sub_task (bool, optional):
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B, T_out]`
        """
        batch_size, _, decoder_num_units = enc_out.shape

        # Start from <SOS>
        sos = self.sos_index_sub if is_sub_task else self.sos_index
        eos = self.eos_index_sub if is_sub_task else self.eos_index

        best_hyps = []
        for b in range(batch_size):
            frame_num = int(x_lens[b])
            xp = cuda.get_array_module(enc_out)

            # Initialize decoder state, decoder output, attention_weights
            dec_state = self._init_decoder_state(
                enc_out[b:b + 1, :, :], is_sub_task=is_sub_task)
            dec_out = self._zero_init(
                (1, 1, decoder_num_units), dtype=np.float32)
            att_weights_step = self._zero_init(
                (1, frame_num), dtype=np.float32)

            complete = []
            beam = [{'hyp': [],
                     'score': LOG_1,
                     'dec_state': dec_state,
                     'dec_out': dec_out,
                     'att_weights_step': att_weights_step}]
            for t in range(max_decode_len):
                new_beam = []
                for i_beam in range(len(beam)):
                    y = beam[i_beam]['hyp'][-1] if t > 0 else sos
                    y = self._create_token(value=y, batch_size=1)

                    if is_sub_task:
                        y = self.embed_sub(y)
                    else:
                        y = self.embed(y)

                    # Compute context vector
                    context_vec = F.sum(enc_out[b: b + 1] * F.broadcast_to(
                        F.expand_dims(att_weights_step, axis=2),
                        (1, frame_num, enc_out.shape[-1])),
                        axis=1, keepdims=True)

                    dec_out, dec_state, context_vec, att_weights_step = self._decode_step(
                        enc_out=enc_out[b:b + 1, :frame_num],
                        x_lens=x_lens[b:b + 1],
                        y=y,
                        dec_state=beam[i_beam]['dec_state'],
                        dec_out=beam[i_beam]['dec_out'],
                        context_vec=context_vec,
                        att_weights_step=beam[i_beam]['att_weights_step'],
                        is_sub_task=is_sub_task)

                    concat = F.concat([dec_out, context_vec], axis=-1)
                    if is_sub_task:
                        # logits = self.fc_sub(
                        #     F.tanh(self.proj_layer_sub(concat)))
                        logits = self.fc_sub(self.proj_layer_sub(concat))
                    else:
                        # logits = self.fc(F.tanh(self.proj_layer(concat)))
                        logits = self.fc(self.proj_layer(concat))

                    # Path through the softmax layer & convert to log-scale
                    log_probs = F.log_softmax(F.squeeze(logits, axis=1))
                    # NOTE: `[1 (B), 1, num_classes]` -> `[1 (B), num_classes]`

                    # Pick up the top-k scores
                    indices_topk = xp.argsort(log_probs.data, axis=1)[
                        0, ::-1][:beam_width]
                    if xp != np:
                        indices_topk = indices_topk.get()

                    for i in indices_topk:
                        log_prob = log_probs.data[0, i]
                        if xp != np:
                            log_prob = log_prob.get()
                        new_hyp = beam[i_beam]['hyp'] + [i]

                        new_score = np.logaddexp(
                            beam[i_beam]['score'], log_prob)

                        new_beam.append({'hyp': new_hyp,
                                         'score': new_score,
                                         'dec_state': dec_state,
                                         'dec_out': dec_out,
                                         'att_weights_step': att_weights_step})

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
            xs (list of np.ndarray): A tensor of size `[B, T_in, input_size]`
            x_lens (np.ndarray): A tensor of size `[B]`
            beam_width (int, optional): the size of beam
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B]`
            perm_idx (np.ndarray): For interface with pytorch, not used
        """
        assert self.ctc_loss_weight > 0
        # TODO: add is_sub_task??

        with chainer.no_backprop_mode(), chainer.using_config('train', False):

            # Wrap by Variable
            xs = np2var(xs, use_cuda=self.use_cuda, backend='chainer')

            # Encode acoustic features
            enc_out, x_lens = self._encode(xs, x_lens)

            # Path through the softmax layer
            batch_size, max_time = enc_out.shape[:2]
            enc_out = enc_out.reshape(batch_size * max_time, -1)
            logits_ctc = self.fc_ctc(enc_out)
            logits_ctc = logits_ctc.reshape(batch_size, max_time, -1)

        if beam_width == 1:
            best_hyps = self._decode_ctc_greedy_np(
                var2np(logits_ctc, backend='chainer'), x_lens)
        else:
            best_hyps = self._decode_ctc_beam_np(
                var2np(F.log_softmax(logits_ctc), backend='chainer'),
                x_lens, beam_width=beam_width)

        # NOTE: index 0 is reserved for the blank
        best_hyps -= 1

        perm_idx = np.arange(0, len(xs), 1)
        return best_hyps, perm_idx
