#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention-based sequence-to-sequence model (chainer)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import copy

import chainer
from chainer import Variable
from chainer import functions as F
from models.chainer.ctc.ctc_loss_from_chainer import connectionist_temporal_classification

from models.chainer.base import ModelBase
from models.chainer.linear import LinearND, Embedding, Embedding_LS
from models.chainer.encoders.load_encoder import load
from models.chainer.attention.rnn_decoder import RNNDecoder
from models.chainer.attention.attention_layer import AttentionMechanism
from models.chainer.criterion import cross_entropy_label_smoothing
from models.pytorch.ctc.decoders.greedy_decoder import GreedyDecoder
from models.pytorch.ctc.decoders.beam_search_decoder import BeamSearchDecoder


class AttentionSeq2seq(ModelBase):
    """Attention-based sequence-to-sequence model.
    Args:
        input_size (int): the dimension of input features (freq * channel)
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
        parameter_init_distribution (string): uniform or normal or orthogonal
            or constant distribution
        parameter_init (float): Range of uniform distribution to initialize
            weight parameters
        recurrent_weight_orthogonal (bool): if True, recurrent weights are
            orthogonalized
        init_forget_gate_bias_with_one (bool): if True, initialize the forget
            gate bias with 1
        subsample_list (list): subsample in the corresponding layers (True)
            ex.) [False, True, True, False] means that subsample is conducted
                in the 2nd and 3rd layers.
        subsample_type (string): drop or concat
        bridge_layer (bool): if True, add the bridge layer between the encoder
            and decoder
        init_dec_state (bool): how to initialize decoder state
            zero => initialize with zero state
            mean => initialize with the mean of encoder outputs in all time steps
            final => initialize with tha final encoder state
            first => initialize with tha first encoder state
        sharpening_factor (float): a sharpening factor in the softmax layer
            for computing attention weights
        logits_temperature (float): a parameter for smoothing the softmax layer
            in outputing probabilities
        sigmoid_smoothing (bool): if True, replace softmax function in
            computing attention weights with sigmoid function for smoothing
        coverage_weight (float): the weight parameter for coverage computation
        ctc_loss_weight (float): A weight parameter for auxiliary CTC loss
        attention_conv_num_channels (int): the number of channles of conv outputs.
            This is used for location-based attention.
        attention_conv_width (int): the size of kernel.
            This must be the odd number.
        num_stack (int): the number of frames to stack
        splice (int): frames to splice. Default is 1 frame.
        input_channel (int): the number of channels of input features
        conv_channels (list): the number of channles in the convolution of the
            location-based attention
        conv_kernel_sizes (list): the size of kernels in the convolution of the
            location-based attention
        conv_strides (list): strides in the convolution of the location-based
            attention
        poolings (list): the size of poolings in the convolution of the
            location-based attention
        activation (string): The activation function of CNN layers.
            Choose from relu or prelu or hard_tanh or maxout
        batch_norm (bool):
        scheduled_sampling_prob (float):
        scheduled_sampling_max_step (float):
        label_smoothing_prob (float):
        weight_noise_std (flaot):
        encoder_residual (bool):
        encoder_dense_residual (bool):
        decoder_residual (bool):
        decoder_dense_residual (bool):
        decoding_order (string):
            luong or bahdanau or conditional
        bottleneck_dim (int): the dimension of the pre-softmax layer
        backward_loss_weight (int): A weight parameter for the loss of the backward decdoer,
            where the model predicts each token in the reverse order
        num_heads (int): the number of heads in the multi-head attention
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
                 bridge_layer=False,
                 init_dec_state='first',
                 sharpening_factor=1,
                 logits_temperature=1,
                 sigmoid_smoothing=False,
                 coverage_weight=0,
                 ctc_loss_weight=0,
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
                 bottleneck_dim=None,
                 backward_loss_weight=0,
                 num_heads=1):

        super(ModelBase, self).__init__()
        self.model_type = 'attention'

        # Setting for the encoder
        self.input_size = input_size
        self.num_stack = num_stack
        self.encoder_type = encoder_type
        self.encoder_num_units = encoder_num_units
        if encoder_bidirectional:
            self.encoder_num_units *= 2
        self.encoder_num_proj = encoder_num_proj
        self.encoder_num_layers = encoder_num_layers
        self.subsample_list = subsample_list

        # Setting for the decoder
        self.decoder_type = decoder_type
        self.decoder_num_units_0 = decoder_num_units
        self.decoder_num_layers_0 = decoder_num_layers
        self.embedding_dim = embedding_dim
        self.bottleneck_dim = decoder_num_units if bottleneck_dim is None else bottleneck_dim
        self.num_classes = num_classes + 1  # Add <EOS> class
        self.sos_0 = num_classes
        self.eos_0 = num_classes
        # NOTE: <SOS> and <EOS> have the same index
        self.decoding_order = decoding_order
        assert 0 <= backward_loss_weight <= 1
        self.fwd_weight_0 = 1 - backward_loss_weight
        self.bwd_weight_0 = backward_loss_weight

        # Setting for the decoder initialization
        if init_dec_state not in ['zero', 'mean', 'final', 'first']:
            raise ValueError(
                'init_dec_state must be "zero" or "mean" or "final" or "first".')
        self.init_dec_state_0_fwd = init_dec_state
        self.init_dec_state_0_bwd = init_dec_state
        if backward_loss_weight > 0:
            if init_dec_state == 'first':
                self.init_dec_state_0_bwd = 'final'
            elif init_dec_state == 'final':
                self.init_dec_state_0_bwd = 'first'
        if encoder_type != decoder_type:
            self.init_dec_state_0_fwd = 'zero'
            self.init_dec_state_0_bwd = 'zero'

        # Setting for the attention
        self.sharpening_factor = sharpening_factor
        self.logits_temperature = logits_temperature
        self.sigmoid_smoothing = sigmoid_smoothing
        self.coverage_weight = coverage_weight
        self.num_heads_0 = num_heads

        # Setting for regularization
        self.weight_noise_injection = False
        self.weight_noise_std = float(weight_noise_std)
        if scheduled_sampling_prob > 0 and scheduled_sampling_max_step == 0:
            raise ValueError
        self.ss_prob = scheduled_sampling_prob
        self._ss_prob = scheduled_sampling_prob
        self.ss_max_step = scheduled_sampling_max_step
        self._step = 0
        self.ls_prob = label_smoothing_prob

        # Setting for MTL
        self.ctc_loss_weight = ctc_loss_weight

        with self.init_scope():
            ##############################
            # Encoder
            ##############################
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
                    use_cuda=self.use_cuda,
                    activation=activation,
                    batch_norm=batch_norm)
                self.init_dec_state_0 = 'zero'
            else:
                raise NotImplementedError

            ##################################################
            # Bridge layer between the encoder and decoder
            ##################################################
            if encoder_type == 'cnn':
                self.bridge_0 = LinearND(
                    self.encoder.output_size, decoder_num_units,
                    dropout=dropout_encoder, use_cuda=self.use_cuda)
                self.encoder_num_units = decoder_num_units
                self.is_bridge = True
            elif bridge_layer:
                self.bridge_0 = LinearND(
                    self.encoder_num_units, decoder_num_units,
                    dropout=dropout_encoder, use_cuda=self.use_cuda)
                self.encoder_num_units = decoder_num_units
                self.is_bridge = True
            else:
                self.is_bridge = False

            directions = []
            if self.fwd_weight_0 > 0:
                directions.append('fwd')
            if self.bwd_weight_0 > 0:
                directions.append('bwd')
            for dir in directions:
                ##################################################
                # Initialization of the decoder
                ##################################################
                if getattr(self, 'init_dec_state_0_' + dir) != 'zero':
                    setattr(self, 'W_dec_init_0_' + dir, LinearND(
                        self.encoder_num_units, decoder_num_units,
                        use_cuda=self.use_cuda))

                ##############################
                # Decoder
                ##############################
                if decoding_order == 'conditional':
                    setattr(self, 'decoder_first_0_' + dir, RNNDecoder(
                        input_size=embedding_dim,
                        rnn_type=decoder_type,
                        num_units=decoder_num_units,
                        num_layers=1,
                        dropout=dropout_decoder,
                        use_cuda=self.use_cuda,
                        residual=False,
                        dense_residual=False))
                    setattr(self, 'decoder_second_0_' + dir, RNNDecoder(
                        input_size=self.encoder_num_units,
                        rnn_type=decoder_type,
                        num_units=decoder_num_units,
                        num_layers=1,
                        dropout=dropout_decoder,
                        use_cuda=self.use_cuda,
                        residual=False,
                        dense_residual=False))
                    # NOTE; the conditional decoder only supports the 1 layer
                else:
                    setattr(self, 'decoder_0_' + dir, RNNDecoder(
                        input_size=self.encoder_num_units + embedding_dim,
                        rnn_type=decoder_type,
                        num_units=decoder_num_units,
                        num_layers=decoder_num_layers,
                        dropout=dropout_decoder,
                        use_cuda=self.use_cuda,
                        residual=decoder_residual,
                        dense_residual=decoder_dense_residual))

                ##############################
                # Attention layer
                ##############################
                setattr(self, 'attend_0_' + dir, AttentionMechanism(
                    encoder_num_units=self.encoder_num_units,
                    decoder_num_units=decoder_num_units,
                    attention_type=attention_type,
                    attention_dim=attention_dim,
                    use_cuda=self.use_cuda,
                    sharpening_factor=sharpening_factor,
                    sigmoid_smoothing=sigmoid_smoothing,
                    out_channels=attention_conv_num_channels,
                    kernel_size=attention_conv_width,
                    num_heads=num_heads))

                ##############################
                # Output layer
                ##############################
                setattr(self, 'W_d_0_' + dir, LinearND(
                    decoder_num_units, self.bottleneck_dim,
                    dropout=dropout_decoder, use_cuda=self.use_cuda))
                setattr(self, 'W_c_0_' + dir, LinearND(
                    self.encoder_num_units, self.bottleneck_dim,
                    dropout=dropout_decoder, use_cuda=self.use_cuda))
                setattr(self, 'fc_0_' + dir,
                        LinearND(self.bottleneck_dim, self.num_classes,
                                 use_cuda=self.use_cuda))

            ##############################
            # Embedding
            ##############################
            if label_smoothing_prob > 0:
                self.embed_0 = Embedding_LS(
                    num_classes=self.num_classes,
                    embedding_dim=embedding_dim,
                    dropout=dropout_embedding,
                    label_smoothing_prob=label_smoothing_prob,
                    use_cuda=self.use_cuda)
            else:
                self.embed_0 = Embedding(
                    num_classes=self.num_classes,
                    embedding_dim=embedding_dim,
                    dropout=dropout_embedding,
                    # ignore_index=self.sos,
                    use_cuda=self.use_cuda)

            ##############################
            # CTC
            ##############################
            if ctc_loss_weight > 0:
                if self.is_bridge:
                    self.fc_ctc_0 = LinearND(
                        decoder_num_units, num_classes + 1,
                        use_cuda=self.use_cuda)
                else:
                    self.fc_ctc_0 = LinearND(
                        self.encoder_num_units, num_classes + 1,
                        use_cuda=self.use_cuda)

                self.blank_index = 0

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
            xs (list of np.ndarray): `[T_in, input_size]` * B
            ys (list of np.ndarray): `[T_out] * B`
            x_lens (list): `[B]`
            y_lens (list): `[B]`
            is_eval (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (chainer.Variable(float) or float): A tensor of size `[1]`
        """
        if is_eval:
            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                loss = self._forward(xs, ys, x_lens, y_lens).data
        else:
            # TODO: add Gaussian noise injection

            loss = self._forward(xs, ys, x_lens, y_lens)

            # Update the probability of scheduled sampling
            self._step += 1
            if self.ss_prob > 0:
                self._ss_prob = min(
                    self.ss_prob, self.ss_prob / self.ss_max_step * self._step)

        return loss

    def _forward(self, xs, ys, x_lens, y_lens):

        # Wrap by Variable
        xs = [Variable(self.xp.array(x, dtype=np.float32),
                       requires_grad=False) for x in xs]

        # Encode acoustic features
        xs, x_lens = self._encode(xs, x_lens)

        # Wrap by Variable
        sos = Variable(self.xp.array(
            [self.sos_0], dtype=np.int32), requires_grad=False)
        eos = Variable(self.xp.array(
            [self.eos_0], dtype=np.int32), requires_grad=False)
        ys = [Variable(self.xp.array(y, dtype=np.int32), requires_grad=False)
              for y in ys]

        ##################################################
        # Compute loss for the forward decoder
        ##################################################
        if self.fwd_weight_0 > 0:
            # NOTE: ys is padded with -1 here
            # ys_in_fwd is padded with <EOS> in order to convert to one-hot vector,
            # and added <SOS> before the first token
            # ys_out_fwd is padded with -1, and added <EOS> after the last token
            ys_in_fwd = [F.concat([sos, y], axis=0) for y in ys]
            ys_out_fwd = [F.concat([y, eos], axis=0) for y in ys]
            y_lens_fwd = self.np2var(y_lens)

            # Concatenate
            ys_in_fwd = F.pad_sequence(ys_in_fwd, padding=self.eos_0)
            ys_out_fwd = F.pad_sequence(ys_out_fwd, padding=-1)

            # Compute XE loss
            loss = self.compute_xe_loss(
                xs, ys_in_fwd, ys_out_fwd, x_lens, y_lens_fwd,
                task=0, dir='fwd') * self.fwd_weight_0
        else:
            loss = 0

        ##################################################
        # Compute loss for the backward decoder
        ##################################################
        if self.bwd_weight_0 > 0:
            # Reverse the order
            if self.bwd_weight_0 > 0:
                ys_tmp = [y[::-1] for y in ys]

            # Wrap by Variable
            ys_in_bwd = [F.concat([eos, y], axis=0) for y in ys_tmp]
            ys_out_bwd = [F.concat([y, sos], axis=0) for y in ys_tmp]
            y_lens_bwd = self.np2var(y_lens)

            # Concatenate
            ys_in_bwd = F.pad_sequence(ys_in_bwd, padding=self.eos_0)
            ys_out_bwd = F.pad_sequence(ys_out_bwd, padding=-1)

            # Compute XE loss
            loss += self.compute_xe_loss(
                xs, ys_in_bwd, ys_out_bwd, x_lens, y_lens_bwd,
                task=0, dir='bwd') * self.bwd_weight_0

        ##################################################
        # Auxiliary CTC loss
        ##################################################
        if self.ctc_loss_weight > 0:
            # Wrap by Variable
            ys_ctc = self.np2var(ys)
            y_lens_ctc = self.np2var(y_lens)

            loss += self.compute_ctc_loss(
                xs,
                # ys_in[:, 1:] + 1 if self.blank_index == 0 else ys_in[:, 1:],
                ys_ctc + 1,
                self.np2var(x_lens),  # Variable
                y_lens_ctc) * self.ctc_loss_weight
            # NOTE: exclude <SOS>

        return loss

    def compute_xe_loss(self, enc_out, ys_in, ys_out, x_lens, y_lens,
                        task, dir):
        """Compute XE loss.
        Args:
            enc_out (chainer.Variable, float): A tensor of size
                `[B, T_in, encoder_num_units]`
            ys_in (chainer.Variable, long): A tensor of size
                `[B, T_out]`, which includes <SOS>
            ys_out (chainer.Variable, long): A tensor of size
                `[B, T_out]`, which includes <EOS>
            x_lens (chainer.Variable, int): A tensor of size `[B]`
            y_lens (chainer.Variable, int): A tensor of size `[B]`
            task (int): the index of a task
            dir (str): fwd or bwd
        Returns:
            loss (torch.autograd.Variable, float): A tensor of size `[1]`
        """
        # Teacher-forcing
        logits, aw = self._decode_train(
            enc_out, x_lens, ys_in, task, dir)

        # Output smoothing
        if self.logits_temperature != 1:
            logits /= self.logits_temperature

        # Compute XE sequence loss
        loss = F.softmax_cross_entropy(
            x=logits.reshape((-1, logits.shape[2])),
            t=F.flatten(ys_out),
            normalize=False, cache_score=True, class_weight=None,
            ignore_label=-1, reduce='no')
        # NOTE: len(loss) = batch_size * max_time
        loss = F.sum(loss, axis=0) / len(enc_out)

        # Label smoothing (with uniform distribution)
        if self.ls_prob > 0:
            loss_ls = cross_entropy_label_smoothing(
                logits,
                y_lens=y_lens + 1,  # Add <EOS>
                label_smoothing_prob=self.ls_prob,
                distribution='uniform',
                size_average=False) / len(enc_out)
            loss = loss * (1 - self.ls_prob) + loss_ls

        # Add coverage term
        if self.coverage_weight != 0:
            raise NotImplementedError

        return loss

    def compute_ctc_loss(self, enc_out, ys, x_lens, y_lens, task=0):
        """Compute CTC loss.
        Args:
            enc_out (chainer.Variable, float): A tensor of size
                `[B, T_in, decoder_num_units]`
            ys (chainer.Variable, int): A tensor of size `[B, T_out]`
            x_lens (chainer.Variable, int): A tensor of size `[B]`
            y_lens (chainer.Variable, int): A tensor of size `[B]`
            task (int): the index of a task
        Returns:
            loss (chainer.Variable, float): A tensor of size `[1]`
        """
        # Path through the fully-connected layer
        logits = getattr(self, 'fc_ctc_' + str(task))(enc_out)

        # Compute CTC loss
        loss = connectionist_temporal_classification(
            x=F.separate(logits, axis=1),  # list of Variable
            t=ys,
            blank_symbol=0,
            input_length=x_lens,
            label_length=y_lens,
            reduce='no')
        loss = F.sum(loss, axis=0)

        # Label smoothing (with uniform distribution)
        # if self.ls_prob > 0:
        #     # XE
        #     loss_ls = cross_entropy_label_smoothing(
        #         logits,
        #         y_lens=x_lens,  # NOTE: CTC is frame-synchronous
        #         label_smoothing_prob=self.ls_prob,
        #         distribution='uniform',
        #         size_average=False)
        #     loss = loss * (1 - self.ls_prob) + \
        #         loss_ls

        loss /= len(enc_out)

        return loss

    def _encode(self, xs, x_lens, is_multi_task=False):
        """Encode acoustic features.
        Args:
            xs (list of chainer.Variable(float)):
                A list of tensors of size `[T_in, input_size]`
            x_lens (np.ndarray): A tensor of size `[B]`
            is_multi_task (bool):
        Returns:
            xs (chainer.Variable, float): A tensor of size
                `[B, T_in, encoder_num_units]`
            x_lens (np.ndarray): A tensor of size `[B]`
            OPTION:
                xs_sub (chainer.Variable, float): A tensor of size
                    `[B, T_in, encoder_num_units]`
                x_lens_sub (np.ndarray): A tensor of size `[B]`
        """
        if is_multi_task:
            if self.encoder_type == 'cnn':
                xs, x_lens = self.encoder(xs, x_lens)
                xs_sub = xs
                x_lens_sub = x_lens
            else:
                xs, x_lens, xs_sub, x_lens_sub = self.encoder(xs, x_lens)
        else:
            xs, x_lens = self.encoder(xs, x_lens)

        # Concatenate
        xs = F.pad_sequence(xs, padding=0)

        if is_multi_task:
            # Concatenate
            xs_sub = F.pad_sequence(xs_sub, padding=0)

        # Bridge between the encoder and decoder in the main task
        if self.is_bridge:
            xs = self.bridge_0(xs)
        if is_multi_task and self.is_bridge_sub:
            xs_sub = self.bridge_1(xs_sub)

        if is_multi_task:
            return xs, x_lens, xs_sub, x_lens_sub
        else:
            return xs, x_lens

    def _compute_coverage(self, aw):
        batch_size, max_time_outputs, max_time_inputs = aw.shape
        raise NotImplementedError

    def _decode_train(self, enc_out, x_lens, ys, task, dir):
        """Decoding in the training stage.
        Args:
            enc_out (chainer.Variable, float): A tensor of size
                `[B, T_in, encoder_num_units]`
            x_lens (np.ndarray): A tensor of size `[B]`
            ys (chainer.Variable, int): A tensor of size `[B, T_out]`
            task (int): the index of a task
            dir (str): fwd or bwd
        Returns:
            logits (chainer.Variable, float): A tensor of size `[B, T_out, num_classes]`
            # aw (chainer.Variable, float): A tensor of size `[B, T_out, T_in, num_heads]`
            aw (chainer.Variable, float): A tensor of size `[B, T_out, T_in]`
        """
        batch_size, max_time = enc_out.shape[:2]

        # Initialize decoder state, decoder output, attention_weights
        dec_state, dec_out = self._init_decoder_state(
            enc_out, task, dir)
        # aw_step = self._create_zero_var(
        #     (batch_size, max_time, getattr(self, 'num_heads_' + str(task))))
        aw_step = self._create_zero_var((batch_size, max_time))
        context_vec = self._create_zero_var((batch_size, 1, enc_out.shape[-1]))

        # Pre-computation of embedding
        ys_emb = [getattr(self, 'embed_' + str(task))(ys[:, t:t + 1])
                  for t in range(ys.shape[1])]
        ys_emb = F.concat(ys_emb, axis=1)

        # Pre-computation of encoder-side features computing scores
        enc_out_a = []
        for h in range(getattr(self, 'num_heads_' + str(task))):
            enc_out_a += [getattr(getattr(self, 'attend_' +
                                          str(task) + '_' + dir), 'W_enc_head' + str(h))(enc_out)]
        enc_out_a = F.concat(enc_out_a, axis=-1)

        logits, aw = [], []
        for t in range(ys.shape[1]):
            # Sample for scheduled sampling
            if self.ss_prob > 0 and t > 0 and self._step > 0 and random.random(
            ) < self._ss_prob:
                y = getattr(self, 'embed_' + str(task))(
                    F.argmax(logits[-1], axis=2))
            else:
                y = ys_emb[:, t:t + 1]

            if self.decoding_order == 'bahdanau':
                if t > 0:
                    # Recurrency
                    dec_in = F.concat([y, context_vec], axis=-1)
                    dec_out, dec_state = getattr(
                        self, 'decoder_' + str(task) + '_' + dir)(dec_in, dec_state)

                # Score
                context_vec, aw_step = getattr(self, 'attend_' + str(task) + '_' + dir)(
                    enc_out, enc_out_a, x_lens, dec_out, aw_step)

            elif self.decoding_order == 'luong':
                # Recurrency
                dec_in = F.concat([y, context_vec], axis=-1)
                dec_out, dec_state = getattr(
                    self, 'decoder_' + str(task) + '_' + dir)(dec_in, dec_state)

                # Score
                context_vec, aw_step = getattr(self, 'attend_' + str(task) + '_' + dir)(
                    enc_out, enc_out_a, x_lens, dec_out, aw_step)

            elif self.decoding_order == 'conditional':
                # Recurrency of the first decoder
                _dec_out, _dec_state = getattr(self, 'decoder_first_' + str(task) + '_' + dir)(
                    y, dec_state)

                # Score
                context_vec, aw_step = getattr(self, 'attend_' + str(task) + '_' + dir)(
                    enc_out, enc_out_a, x_lens, _dec_out, aw_step)

                # Recurrency of the second decoder
                dec_out, dec_state = getattr(self, 'decoder_second_' + str(task) + '_' + dir)(
                    context_vec, _dec_state)

            else:
                raise ValueError(self.decoding_order)

            # Generate
            logits_step = getattr(self, 'fc_' + str(task) + '_' + dir)(F.tanh(
                getattr(self, 'W_d_' + str(task) + '_' + dir)(dec_out) +
                getattr(self, 'W_c_' + str(task) + '_' + dir)(context_vec)))

            logits.append(logits_step)
            aw.append(aw_step)

        # Concatenate in T_out-dimension
        logits = F.concat(logits, axis=1)
        # aw = F.concat(aw, axis=-1).reshape(
        #     batch_size, -1, max_time, getattr(self, 'num_heads_' + str(task)))
        aw = F.concat(aw, axis=-1).reshape(batch_size, -1, max_time)

        return logits, aw

    def _init_decoder_state(self, enc_out, task, dir):
        """Initialize decoder state.
        Args:
            enc_out (chainer.Variable, float): A tensor of size
                `[B, T_in, encoder_num_units]`
            task (int): the index of a task
            dir (str): fwd or bwd
        Returns:
            dec_state (list or tuple of list):
            dec_out (chainer.Variable): A tensor of size
                '[B, 1, decoder_num_units]'
        """
        zero_state = self._create_zero_var(
            (enc_out.shape[0], getattr(self, 'decoder_num_units_' + str(task))))

        if getattr(self, 'init_dec_state_' + str(task) + '_' + dir) == 'zero':
            if self.decoder_type == 'stateless_lstm':
                hx_list = [None] * getattr(
                    self, 'decoder_num_layers_' + str(task))
                cx_list = [None] * getattr(
                    self, 'decoder_num_layers_' + str(task))
            elif self.decoder_type == 'lstm':
                hx_list = [zero_state] * getattr(
                    self, 'decoder_num_layers_' + str(task))
                cx_list = [zero_state] * getattr(
                    self, 'decoder_num_layers_' + str(task))
            else:
                hx_list = [zero_state] * getattr(
                    self, 'decoder_num_layers_' + str(task))

            dec_out = self._create_zero_var((enc_out.shape[0], 1, getattr(
                self, 'decoder_num_units_' + str(task))))
        else:
            if getattr(self, 'init_dec_state_' + str(task) + '_' + dir) == 'mean':
                # Initialize with mean of all encoder outputs
                h_0 = F.mean(enc_out, axis=1, keepdims=False)
            elif getattr(self, 'init_dec_state_' + str(task) + '_' + dir) == 'final':
                # Initialize with the final encoder output
                h_0 = enc_out[:, -1, :]
            elif getattr(self, 'init_dec_state_' + str(task) + '_' + dir) == 'first':
                # Initialize with the first encoder output
                h_0 = enc_out[:, 0, :]
            # NOTE: h_0: `[B, encoder_num_units]`

            # Path through the linear layer
            h_0 = F.tanh(
                getattr(self, 'W_dec_init_' + str(task) + '_' + dir)(h_0))

            hx_list = [h_0] * getattr(self, 'decoder_num_layers_' + str(task))
            # NOTE: all layers are initialized with the same values

            if self.decoder_type == 'stateless_lstm':
                cx_list = [None] * getattr(
                    self, 'decoder_num_layers_' + str(task))
            elif self.decoder_type == 'lstm':
                cx_list = [zero_state] * getattr(
                    self, 'decoder_num_layers_' + str(task))

            dec_out = F.expand_dims(h_0, axis=1)

        if self.decoder_type in ['gru', 'rnn']:
            dec_state = hx_list
        else:
            dec_state = (hx_list, cx_list)

        return dec_state, dec_out

    def decode(self, xs, x_lens, beam_width, max_decode_len, min_decode_len=0,
               length_penalty=0, coverage_penalty=0, task_index=0,
               resolving_unk=False):
        """Decoding in the inference stage.
        Args:
            xs (list of np.ndarray): A tensor of size `[B, T_in, input_size]`
            xs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            x_lens (np.ndarray): A tensor of size `[B]`
            beam_width (int): the size of beam
            max_decode_len (int): the maximum sequence length of tokens
            min_decode_len (int): the minimum sequence length of tokens
            length_penalty (float): length penalty in beam search decoding
            coverage_penalty (float): coverage penalty in beam search decoding
            task_index (int): not used (to make compatible)
            resolving_unk (bool): not used (to make compatible)
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B]`
            # aw (np.ndarray): A tensor of size `[B, T_out, T_in, num_heads]`
            aw (np.ndarray): A tensor of size `[B, T_out, T_in]`
            perm_idx (np.ndarray): For interface with pytorch, not used
        """
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            # Wrap by Variable
            xs = [Variable(self.xp.array(x, dtype=np.float32),
                           requires_grad=False) for x in xs]

            # Encode acoustic features
            enc_out, x_lens = self._encode(xs, x_lens)

            dir = 'fwd'if self.fwd_weight_0 >= self.bwd_weight_0 else 'bwd'

            if beam_width == 1:
                best_hyps, aw = self._decode_infer_greedy(
                    enc_out, x_lens, max_decode_len, task=0, dir=dir)
            else:
                best_hyps, aw, = self._decode_infer_beam(
                    enc_out, x_lens, beam_width, max_decode_len, min_decode_len,
                    length_penalty, coverage_penalty, task=0, dir=dir)

        # TODO: fix this
        # if beam_width == 1:
        #     aw = aw[:, :, :, 0]

        perm_idx = np.arange(0, len(xs), 1)

        return best_hyps, aw, perm_idx

    def _decode_infer_greedy(self, enc_out, x_lens, max_decode_len, task, dir):
        """Greedy decoding in the inference stage.
        Args:
            enc_out (chainer.Variable, float): A tensor of size
                `[B, T_in, encoder_num_units]`
            x_lens (np.ndarray): A tensor of size `[B]`
            max_decode_len (int): the maximum sequence length of tokens
            task (int): the index of a task
            dir (str): fwd or bwd
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B, T_out]`
            # aw (np.ndarray): A tensor of size `[B, T_out, T_in, num_heads]`
            aw (np.ndarray): A tensor of size `[B, T_out, T_in]`
        """
        if dir == 'bwd':
            assert getattr(self, 'bwd_weight_' + str(task)) > 0

        batch_size, max_time = enc_out.shape[:2]

        # Initialization
        dec_state, dec_out = self._init_decoder_state(enc_out, task, dir)
        # aw_step = self._create_zero_var(
        #     (batch_size, max_time, getattr(self, 'num_heads_' + str(task))))
        aw_step = self._create_zero_var((batch_size, max_time))
        context_vec = self._create_zero_var((batch_size, 1, enc_out.shape[-1]))

        # Start from <SOS>
        sos = getattr(self, 'sos_' + str(task))
        eos = getattr(self, 'eos_' + str(task))
        y = self._create_var((batch_size, 1), fill_value=sos, dtype=np.int32)

        # Pre-computation of encoder-side features computing scores
        enc_out_a = []
        for h in range(getattr(self, 'num_heads_' + str(task))):
            enc_out_a += [getattr(getattr(self, 'attend_' +
                                          str(task) + '_' + dir), 'W_enc_head' + str(h))(enc_out)]
        enc_out_a = F.concat(enc_out_a, axis=-1)

        best_hyps, aw = [], []
        y_lens = np.zeros((batch_size,), dtype=np.int32)
        eos_flag = [False] * batch_size
        for t in range(max_decode_len):
            y = getattr(self, 'embed_' + str(task))(y)

            if self.decoding_order == 'bahdanau':
                if t > 0:
                    # Recurrency
                    dec_in = F.concat([y, context_vec], axis=-1)
                    dec_out, dec_state = getattr(
                        self, 'decoder_' + str(task) + '_' + dir)(dec_in, dec_state)

                # Score
                context_vec, aw_step = getattr(self, 'attend_' + str(task) + '_' + dir)(
                    enc_out, enc_out_a, x_lens, dec_out, aw_step)

            elif self.decoding_order == 'luong':
                # Recurrency
                dec_in = F.concat([y, context_vec], axis=-1)
                dec_out, dec_state = getattr(
                    self, 'decoder_' + str(task) + '_' + dir)(dec_in, dec_state)

                # Score
                context_vec, aw_step = getattr(self, 'attend_' + str(task) + '_' + dir)(
                    enc_out, enc_out_a, x_lens, dec_out, aw_step)

            elif self.decoding_order == 'conditional':
                # Recurrency of the first decoder
                _dec_out, _dec_state = getattr(self, 'decoder_first_' + str(task) + '_' + dir)(
                    y, dec_state)

                # Score
                context_vec, aw_step = getattr(self, 'attend_' + str(task) + '_' + dir)(
                    enc_out, enc_out_a, x_lens, _dec_out, aw_step)

                # Recurrency of the second decoder
                dec_out, dec_state = getattr(self, 'decoder_second_' + str(task) + '_' + dir)(
                    context_vec, _dec_state)

            # Generate
            logits_step = getattr(self, 'fc_' + str(task) + '_' + dir)(F.tanh(
                getattr(self, 'W_d_' + str(task) + '_' + dir)(dec_out) +
                getattr(self, 'W_c_' + str(task) + '_' + dir)(context_vec)))

            # Pick up 1-best
            y = F.expand_dims(
                F.argmax(F.squeeze(logits_step, axis=1), axis=1), axis=1)
            best_hyps.append(y)

            aw.append(aw_step)

            # Count lengths of hypotheses
            if dir == 'bwd':
                for b in range(batch_size):
                    if not eos_flag[b]:
                        if y.data[b] == eos:
                            eos_flag[b] = True
                        else:
                            y_lens[b] += 1
                        # NOTE: exclude <EOS>

            # Break if <EOS> is outputed in all mini-batch
            if sum(y.data == eos)[0] == len(y):
                break

        # Concatenate in T_out dimension
        best_hyps = F.concat(best_hyps, axis=1)
        # aw = F.concat(aw, axis=-1).reshape(
        #     batch_size, -1, max_time, getattr(self, 'num_heads_' + str(task)))
        aw = F.concat(aw, axis=-1).reshape(batch_size, -1, max_time)

        # Convert to numpy
        best_hyps = self.var2np(best_hyps)
        aw = self.var2np(aw)

        # Reverse the order
        if dir == 'bwd':
            for b in range(batch_size):
                best_hyps[b, :y_lens[b]] = best_hyps[b, :y_lens[b]][::-1]

        return best_hyps, aw

    def _decode_infer_beam(self, enc_out, x_lens, beam_width,
                           max_decode_len, min_decode_len,
                           length_penalty, coverage_penalty, task, dir):
        """Beam search decoding in the inference stage.
        Args:
            enc_out (chainer.Variable, float): A tensor of size
                `[B, T_in, encoder_num_units]`
            x_lens (torch.IntTensor): A tensor of size `[B]`
            beam_width (int): the size of beam
            max_decode_len (int): the maximum sequence length of tokens
            min_decode_len (int): the minimum sequence length of tokens
            length_penalty (float):
            coverage_penalty (float):
            task (int): the index of a task
            dir (str): fwd or bwd
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B]`
            aw (list): attention weights of the best hypothesis
        """
        if dir == 'bwd':
            assert getattr(self, 'bwd_weight_' + str(task)) > 0

        # Start from <SOS>
        sos = getattr(self, 'sos_' + str(task))
        eos = getattr(self, 'eos_' + str(task))

        min_decode_len_ratio = 0.05

        # Pre-computation of encoder-side features computing scores
        enc_out_a = []
        for h in range(getattr(self, 'num_heads_' + str(task))):
            enc_out_a += [getattr(getattr(self, 'attend_' +
                                          str(task) + '_' + dir), 'W_enc_head' + str(h))(enc_out)]
            enc_out_a = F.concat(enc_out_a, axis=-1)

        best_hyps, aw = [], []
        y_lens = np.zeros((enc_out.shape[0],), dtype=np.int32)
        for b in range(enc_out.shape[0]):
            # Initialization per utterance
            dec_state, dec_out = self._init_decoder_state(
                enc_out[b:b + 1, :, :], task, dir)
            # aw_step = self._create_zero_var(
            #     (1, int(x_lens[b]), getattr(self, 'num_heads_' + str(task))))
            aw_step = self._create_zero_var((1, int(x_lens[b])))
            context_vec = self._create_zero_var((1,  1, enc_out.shape[-1]))

            complete = []
            beam = [{'hyp': [sos],
                     'score': 0,  # log 1
                     'dec_state': dec_state,
                     'dec_out': dec_out,
                     'context_vec': context_vec,
                     'aw_steps': [aw_step]}]
            for t in range(max_decode_len):
                new_beam = []
                for i_beam in range(len(beam)):
                    y = self._create_var(
                        (1, 1), fill_value=beam[i_beam]['hyp'][-1], dtype='long')
                    y = getattr(self, 'embed_' + str(task))(y)

                    if self.decoding_order == 'bahdanau':
                        if t == 0:
                            pass
                        else:
                            # Recurrency
                            dec_in = F.concat([y, context_vec], axis=-1)
                            dec_out, dec_state = getattr(
                                self, 'decoder_' + str(task) + '_' + dir)(dec_in, beam[i_beam]['dec_state'])

                        # Score
                        context_vec, aw_step = getattr(self, 'attend_' + str(task) + '_' + dir)(
                            enc_out[b:b + 1],
                            enc_out_a[b:b + 1],
                            x_lens[b:b + 1],
                            beam[i_beam]['dec_out'], beam[i_beam]['aw_steps'][-1])

                    elif self.decoding_order == 'luong':
                        # Recurrency
                        dec_in = F.concat([y, context_vec], axis=-1)
                        dec_out, dec_state = getattr(
                            self, 'decoder_' + str(task) + '_' + dir)(dec_in, beam[i_beam]['dec_state'])

                        # Score
                        context_vec, aw_step = getattr(self, 'attend_' + str(task) + '_' + dir)(
                            enc_out[b:b + 1],
                            enc_out_a[b:b + 1],
                            x_lens[b:b + 1],
                            beam[i_beam]['dec_out'], beam[i_beam]['aw_steps'][-1])

                    elif self.decoding_order == 'conditional':
                        # Recurrency of the first decoder
                        _dec_out, _dec_state = getattr(self, 'decoder_first_' + str(task) + '_' + dir)(
                            y, beam[i_beam]['dec_state'])

                        # Score
                        context_vec, aw_step = getattr(self, 'attend_' + str(task) + '_' + dir)(
                            enc_out[b:b + 1],
                            enc_out_a[b:b + 1],
                            x_lens[b:b + 1],
                            _dec_out, beam[i_beam]['aw_steps'][-1])

                        # Recurrency of the second decoder
                        dec_out, dec_state = getattr(self, 'decoder_second_' + str(task) + '_' + dir)(
                            context_vec, _dec_state)

                    # Generate
                    logits_step = getattr(self, 'fc_' + str(task) + '_' + dir)(F.tanh(
                        getattr(self, 'W_d_' + str(task) + '_' + dir)(dec_out) +
                        getattr(self, 'W_c_' + str(task) + '_' + dir)(context_vec)))

                    # Path through the softmax layer & convert to log-scale
                    log_probs = F.log_softmax(F.squeeze(logits_step, axis=1))
                    # log_probs = F.squeeze(logits_step, axis=1)

                    # Pick up the top-k scores
                    indices_topk = self.xp.argsort(log_probs.data, axis=1)[
                        0, ::-1][:beam_width]
                    if self.xp != np:
                        indices_topk = indices_topk.get()

                    for idx in indices_topk:
                        # Exclude short hypotheses
                        if idx == eos and len(beam[i_beam]['hyp']) < min_decode_len:
                            continue
                        # if idx == eos and len(beam[i_beam]['hyp']) < x_lens[b] * min_decode_len_ratio:
                        #     continue

                        log_prob = log_probs.data[0, idx]
                        if self.xp != np:
                            log_prob = log_prob.get()

                        # Add length penalty
                        score = beam[i_beam]['score'] + \
                            log_prob + length_penalty

                        # Add coverage penalty
                        if coverage_penalty > 0:
                            raise NotImplementedError

                            threshold = 0.5
                            aw_steps = F.concat(F.squeeze(
                                beam[i_beam]['aw_steps'], axis=0).sum(0), axis=1)

                            # Google NMT
                            # cov_sum = torch.where(
                            #     aw_steps < threshold, aw_steps, torch.ones_like(aw_steps) * threshold).sum(0)
                            # score += torch.log(cov_sum) * coverage_penalty

                            # Toward better decoding
                            cov_sum = torch.where(
                                aw_steps > threshold, aw_steps, torch.zeros_like(aw_steps)).sum(0)
                            score += cov_sum * coverage_penalty

                        new_beam.append({'hyp': beam[i_beam]['hyp'] + [idx],
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
                    if cand['hyp'][-1] == eos:
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
            y_lens[b] = len(complete[0]['hyp'][1:])

        # Concatenate in T_out dimension
        for j in range(len(aw)):
            for k in range(len(aw[j])):
                # aw[j][k] = aw[j][k][:, :, 0]  # TODO: fix for MHA
                aw[j][k] = aw[j][k][:, :]
            aw[j] = self.var2np(
                F.concat(aw[j], axis=-1).reshape(1, -1, aw[j][0].shape[1]))

        # Reverse the order
        if dir == 'bwd':
            for b in range(enc_out.shape[0]):
                best_hyps[b][:y_lens[b]] = best_hyps[b][:y_lens[b]][::-1]

        return np.array(best_hyps), aw

    def decode_ctc(self, xs, x_lens, beam_width=1, task_index=0):
        """Decoding by the CTC layer in the inference stage.
            This is only used for Joint CTC-Attention model.
        Args:
            xs (list of np.ndarray): A tensor of size `[B, T_in, input_size]`
            x_lens (np.ndarray): A tensor of size `[B]`
            beam_width (int): the size of beam
            task_index (int): the index of a task
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B]`
            perm_idx (np.ndarray): For interface with pytorch, not used
        """
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            # Wrap by Variable
            xs = [Variable(self.xp.array(x, dtype=np.float32),
                           requires_grad=False) for x in xs]

            # Encode acoustic features
            if task_index == 0:
                enc_out, x_lens = self._encode(xs, x_lens)
            elif task_index == 1:
                _, _, enc_out, x_lens = self._encode(
                    xs, x_lens, is_multi_task=True)
            else:
                raise NotImplementedError

            # Path through the softmax layer
            batch_size, max_time = enc_out.shape[:2]
            enc_out = enc_out.reshape(batch_size * max_time, -1)
            logits_ctc = getattr(self, 'fc_ctc_' + str(task_index))(enc_out)
            logits_ctc = logits_ctc.reshape(batch_size, max_time, -1)

        if beam_width == 1:
            best_hyps = self._decode_ctc_greedy_np(
                self.var2np(logits_ctc), x_lens)
        else:
            best_hyps = self._decode_ctc_beam_np(
                self.var2np(F.log_softmax(logits_ctc)),
                x_lens, beam_width=beam_width)

        # NOTE: index 0 is reserved for the blank
        best_hyps -= 1

        perm_idx = np.arange(0, len(xs), 1)

        return best_hyps, perm_idx
