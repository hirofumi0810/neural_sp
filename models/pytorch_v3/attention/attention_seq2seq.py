#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention-based sequence-to-sequence model (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import copy
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from models.pytorch_v3.base import ModelBase
from models.pytorch_v3.linear import LinearND, Embedding
from models.pytorch_v3.encoders.load_encoder import load
from models.pytorch_v3.attention.rnn_decoder import RNNDecoder
from models.pytorch_v3.attention.attention_layer import AttentionMechanism, MultiheadAttentionMechanism
from models.pytorch_v3.ctc.ctc import my_warpctc
from models.pytorch_v3.criterion import cross_entropy_label_smoothing
from models.pytorch_v3.ctc.decoders.greedy_decoder import GreedyDecoder
from models.pytorch_v3.ctc.decoders.beam_search_decoder import BeamSearchDecoder
from models.pytorch_v3.utils import np2var, var2np, pad_list
from utils.io.inputs.frame_stacking import stack_frame
from utils.io.inputs.splicing import do_splice
from models.pytorch_v3.lm.rnnlm import RNNLM


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
        num_skip (int): the number of frames to skip
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
        label_smoothing_type (string): uniform or unigram
        weight_noise_std (flaot):
        encoder_residual (bool):
        encoder_dense_residual (bool):
        decoder_residual (bool):
        decoder_dense_residual (bool):
        decoding_order (string): bahdanau or luong or conditional
        bottleneck_dim (int): the dimension of the pre-softmax layer
        backward_loss_weight (int): A weight parameter for the loss of the backward decdoer,
            where the model predicts each token in the reverse order
        num_heads (int): the number of heads in the multi-head attention
        rnnlm_fusion_type (string): False or cold_fusion or logits_fusion
        rnnlm_config (dict): configuration of the pre-trained RNNLM
        rnnlm_weight (float): the weight for XE loss of RNNLM
        concat_embedding (int): if True, concat embeddings of ASR and RMMLM
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
                 init_dec_state='zero',
                 sharpening_factor=1,
                 logits_temperature=1,
                 sigmoid_smoothing=False,
                 coverage_weight=0,
                 ctc_loss_weight=0,
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
                 bottleneck_dim=None,
                 backward_loss_weight=0,
                 num_heads=1,
                 rnnlm_fusion_type=None,
                 rnnlm_config=None,
                 rnnlm_weight=0,
                 concat_embedding=False):

        super(ModelBase, self).__init__()
        self.model_type = 'attention'

        # Setting for the encoder
        self.input_size = input_size
        self.num_stack = num_stack
        self.num_skip = num_skip
        self.splice = splice
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
        self.pad_index = -1024
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
        self.logits_temp = logits_temperature
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
        self.ls_type = label_smoothing_type

        # Setting for MTL
        self.ctc_loss_weight = ctc_loss_weight

        # Setting for the RNNLM fusion
        self.rnnlm_fusion_type = rnnlm_fusion_type
        self.rnnlm_0_fwd = None
        self.rnnlm_0_bwd = None
        self.rnnlm_weight_0 = rnnlm_weight
        self.concat_embedding = concat_embedding

        # RNNLM fusion
        if rnnlm_fusion_type:
            assert rnnlm_config is not None
            self.rnnlm_0_fwd = RNNLM(
                embedding_dim=rnnlm_config['embedding_dim'],
                rnn_type=rnnlm_config['rnn_type'],
                bidirectional=rnnlm_config['bidirectional'],
                num_units=rnnlm_config['num_units'],
                num_layers=rnnlm_config['num_layers'],
                dropout_embedding=rnnlm_config['dropout_embedding'],
                dropout_hidden=rnnlm_config['dropout_hidden'],
                dropout_output=rnnlm_config['dropout_output'],
                num_classes=rnnlm_config['num_classes'],
                parameter_init_distribution=rnnlm_config['parameter_init_distribution'],
                parameter_init=rnnlm_config['parameter_init'],
                recurrent_weight_orthogonal=rnnlm_config['recurrent_weight_orthogonal'],
                init_forget_gate_bias_with_one=rnnlm_config['init_forget_gate_bias_with_one'],
                tie_weights=rnnlm_config['tie_weights'],
                backward=rnnlm_config['backward'])

            self.W_rnnlm_logits_0_fwd = LinearND(
                self.rnnlm_0_fwd.num_classes, decoder_num_units,
                dropout=dropout_decoder)
            self.W_rnnlm_gate_0_fwd = LinearND(
                decoder_num_units * 2, self.bottleneck_dim,
                dropout=dropout_decoder)

            # Fix RNNLM parameters
            if rnnlm_weight == 0:
                for param in self.rnnlm_0_fwd.parameters():
                    param.requires_grad = False
        # TODO: backward RNNLM

        # Encoder
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
                dense_residual=encoder_dense_residual,
                nin=0)
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
        else:
            raise NotImplementedError

        # Bridge layer between the encoder and decoder
        if encoder_type == 'cnn':
            self.bridge_0 = LinearND(
                self.encoder.output_size, decoder_num_units,
                dropout=dropout_encoder)
            self.encoder_num_units = decoder_num_units
            self.is_bridge = True
        elif bridge_layer:
            self.bridge_0 = LinearND(self.encoder_num_units, decoder_num_units,
                                     dropout=dropout_encoder)
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
            # Initialization of the decoder
            if getattr(self, 'init_dec_state_0_' + dir) != 'zero':
                setattr(self, 'W_dec_init_0_' + dir, LinearND(
                    self.encoder_num_units, decoder_num_units))

            # Decoder
            embedding_size = embedding_dim
            if rnnlm_fusion_type and concat_embedding:
                embedding_size += self.rnnlm_0_fwd.embedding_dim
            if decoding_order == 'conditional':
                setattr(self, 'decoder_first_0_' + dir, RNNDecoder(
                    input_size=embedding_size,
                    rnn_type=decoder_type,
                    num_units=decoder_num_units,
                    num_layers=1,
                    dropout=dropout_decoder,
                    residual=False,
                    dense_residual=False))
                setattr(self, 'decoder_second_0_' + dir, RNNDecoder(
                    input_size=self.encoder_num_units,
                    rnn_type=decoder_type,
                    num_units=decoder_num_units,
                    num_layers=1,
                    dropout=dropout_decoder,
                    residual=False,
                    dense_residual=False))
                # NOTE; the conditional decoder only supports the 1 layer
            else:
                setattr(self, 'decoder_0_' + dir, RNNDecoder(
                    input_size=self.encoder_num_units + embedding_size,
                    rnn_type=decoder_type,
                    num_units=decoder_num_units,
                    num_layers=decoder_num_layers,
                    dropout=dropout_decoder,
                    residual=decoder_residual,
                    dense_residual=decoder_dense_residual))

            # Attention layer
            if num_heads > 1:
                setattr(self, 'attend_0_' + dir, MultiheadAttentionMechanism(
                    encoder_num_units=self.encoder_num_units,
                    decoder_num_units=decoder_num_units,
                    attention_type=attention_type,
                    attention_dim=attention_dim,
                    sharpening_factor=sharpening_factor,
                    sigmoid_smoothing=sigmoid_smoothing,
                    out_channels=attention_conv_num_channels,
                    kernel_size=attention_conv_width,
                    num_heads=num_heads))
            else:
                setattr(self, 'attend_0_' + dir, AttentionMechanism(
                    encoder_num_units=self.encoder_num_units,
                    decoder_num_units=decoder_num_units,
                    attention_type=attention_type,
                    attention_dim=attention_dim,
                    sharpening_factor=sharpening_factor,
                    sigmoid_smoothing=sigmoid_smoothing,
                    out_channels=attention_conv_num_channels,
                    kernel_size=attention_conv_width))

            # Output layer
            setattr(self, 'W_d_0_' + dir, LinearND(
                decoder_num_units, self.bottleneck_dim,
                dropout=dropout_decoder))
            setattr(self, 'W_c_0_' + dir, LinearND(
                self.encoder_num_units, self.bottleneck_dim,
                dropout=dropout_decoder))
            setattr(self, 'fc_0_' + dir,
                    LinearND(self.bottleneck_dim, self.num_classes))

        # Embedding
        self.embed_0 = Embedding(num_classes=self.num_classes,
                                 embedding_dim=embedding_dim,
                                 dropout=dropout_embedding,
                                 ignore_index=self.eos_0)

        # CTC
        if ctc_loss_weight > 0:
            if self.is_bridge:
                self.fc_ctc_0 = LinearND(
                    decoder_num_units, num_classes + 1)
            else:
                self.fc_ctc_0 = LinearND(
                    self.encoder_num_units, num_classes + 1)

            # Set CTC decoders
            self._decode_ctc_greedy_np = GreedyDecoder(blank_index=0)
            self._decode_ctc_beam_np = BeamSearchDecoder(blank_index=0)
            # TODO: set space index

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

    def forward(self, xs, ys, is_eval=False):
        """Forward computation.
        Args:
            xs (list): A list of length `[B]`, which contains arrays of size `[T, input_size]`
            ys (list): A list of length `[B]`, which contains arrays of size `[L]`
            is_eval (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (torch.autograd.Variable(float)): A tensor of size `[1]`
            acc (float): Token-level accuracy in teacher-forcing
        """
        if is_eval:
            self.eval()
        else:
            self.train()

            # Gaussian noise injection
            if self.weight_noise_injection:
                self.inject_weight_noise(mean=0, std=self.weight_noise_std)

        # Sort by lenghts in the descending order
        if is_eval and self.encoder_type != 'cnn':
            perm_idx = sorted(list(range(0, len(xs), 1)),
                              key=lambda i: xs[i].shape[0], reverse=True)
            xs = [xs[i] for i in perm_idx]
            ys = [ys[i] for i in perm_idx]
            # NOTE: must be descending order for pack_padded_sequence
            # NOTE: assumed that xs is already sorted in the training stage

        # Frame stacking
        if self.num_stack > 1:
            xs = [stack_frame(x, self.num_stack, self.num_skip)
                  for x in xs]

        # Splicing
        if self.splice > 1:
            xs = [do_splice(x, self.splice, self.num_stack) for x in xs]

        # Wrap by Variable
        x_lens = [len(x) for x in xs]
        xs = [np2var(x, self.device_id).float() for x in xs]
        if self.fwd_weight_0 > 0 or self.ctc_loss_weight > 0:
            ys_fwd = [np2var(np.fromiter(y, dtype=np.int64), self.device_id).long()
                      for y in ys]
        if self.bwd_weight_0 > 0:
            ys_bwd = [np2var(np.fromiter(y[::-1], dtype=np.int64), self.device_id).long()
                      for y in ys]
            # NOTE: reverse the order

        # Encode acoustic features
        xs, x_lens = self._encode(xs, x_lens)

        # Compute XE loss for the forward decoder
        if self.fwd_weight_0 > 0:
            loss, acc = self.compute_xe_loss(
                xs, ys_fwd, x_lens, task=0, dir='fwd')
            loss *= self.fwd_weight_0
            acc *= self.fwd_weight_0
        else:
            loss = Variable(xs.data.new(1,).fill_(0.))
            acc = 0.

        # Compute XE loss for the backward decoder
        if self.bwd_weight_0 > 0:
            loss_bwd, acc_bwd = self.compute_xe_loss(
                xs, ys_bwd, x_lens, task=0, dir='bwd')
            loss += loss_bwd * self.bwd_weight_0
            acc += acc_bwd * self.bwd_weight_0

        # Auxiliary CTC loss
        if self.ctc_loss_weight > 0:
            loss += self.compute_ctc_loss(
                xs, ys_fwd, x_lens) * self.ctc_loss_weight

        if not is_eval:
            # Update the probability of scheduled sampling
            self._step += 1
            if self.ss_prob > 0:
                self._ss_prob = min(
                    self.ss_prob, self.ss_prob / self.ss_max_step * self._step)

        return loss, acc

    def compute_xe_loss(self, enc_out, ys, x_lens, task, dir):
        """Compute XE loss.
        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, encoder_num_units]`
            ys (list): A list of length `[B]`, which contains Variables of size `[L]`
            x_lens (list): A list of length `[B]`
            task (int): the index of a task
            dir (str): fwd or bwd
        Returns:
            loss (torch.autograd.Variable, float): A tensor of size `[1]`
            acc (float): Token-level accuracy in teacher-forcing
        """
        sos = Variable(enc_out.data.new(1,).fill_(
            getattr(self, 'sos_' + str(task))).long())
        eos = Variable(enc_out.data.new(1,).fill_(
            getattr(self, 'eos_' + str(task))).long())

        # Append <SOS> and <EOS>
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # Convert list to Variable
        ys_in = pad_list(ys_in, getattr(self, 'eos_' + str(task)))
        ys_out = pad_list(ys_out, self.pad_index)

        # NOTE: change RNNLM to the evaluation mode in case of cold fusion
        if self.rnnlm_fusion_type and getattr(self, 'rnnlm_' + str(task) + '_' + dir) == 0 and getattr(self, 'rnnlm_' + str(task) + '_' + dir) is not None:
            getattr(self, 'rnnlm_' + str(task) + '_' + dir).eval()

        # Teacher-forcing
        logits, aw, logits_rnnlm = self._decode_train(
            enc_out, x_lens, ys_in, task, dir)

        # Output smoothing
        if self.logits_temp != 1:
            logits /= self.logits_temp

        # Compute XE sequence loss
        if self.ls_prob > 0:
            # Label smoothing
            y_lens = [y.size(0) + 1 for y in ys]   # Add <EOS>
            loss = cross_entropy_label_smoothing(
                logits, ys=ys_out, y_lens=y_lens,
                label_smoothing_prob=self.ls_prob,
                label_smoothing_type=self.ls_type, size_average=True)
        else:
            loss = F.cross_entropy(
                input=logits.view((-1, logits.size(2))),
                target=ys_out.view(-1),  # long
                ignore_index=self.pad_index, size_average=False) / len(enc_out)

        # Compute XE loss for RNNLM
        if self.rnnlm_fusion_type and getattr(self, 'rnnlm_weight_' + str(task)) > 0:
            if dir == 'fwd':
                loss_rnnlm = F.cross_entropy(
                    input=logits_rnnlm.view((-1, logits_rnnlm.size(2))),
                    target=ys_out.contiguous().view(-1),
                    ignore_index=self.pad_index, size_average=True)
                loss += loss_rnnlm * getattr(self, 'rnnlm_weight_' + str(task))

        # Add coverage term
        if self.coverage_weight > 0:
            raise NotImplementedError

        # Compute token-level accuracy in teacher-forcing
        pad_pred = logits.data.view(ys_out.size(
            0), ys_out.size(1), logits.size(-1)).max(2)[1]
        mask = ys_out.data != self.pad_index
        numerator = torch.sum(pad_pred.masked_select(
            mask) == ys_out.data.masked_select(mask))
        denominator = torch.sum(mask)
        acc = float(numerator) / float(denominator)

        return loss, acc

    def compute_ctc_loss(self, enc_out, ys, x_lens, task=0):
        """Compute CTC loss.
        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, encoder_num_units]`
            ys (torch.autograd.Variable, long): A tensor of size `[B, L]`,
                which does not include <SOS> nor <EOS>
            x_lens (list): A list of length `[B]`
            task (int): the index of a task
        Returns:
            loss (torch.autograd.Variable, float): A tensor of size `[1]`
        """
        # Wrap by Variable
        x_lens = np2var(np.fromiter(x_lens, dtype=np.int32), -1).int()
        y_lens = np2var(np.fromiter([y.size(0)
                                     for y in ys], dtype=np.int32), -1).int()
        # NOTE: do not copy to GPUs

        # Concatenate all elements in ys for warpctc_pytorch
        ys_ctc = torch.cat(ys, dim=0).cpu().int() + 1
        # NOTE: index 0 is reserved for blank in warpctc_pytorch

        # Path through the fully-connected layer
        logits = getattr(self, 'fc_ctc_' + str(task))(enc_out)

        # Compute CTC loss
        loss = my_warpctc(logits.transpose(0, 1),  # time-major
                          ys_ctc,  # int
                          x_lens,  # int
                          y_lens,  # int
                          size_average=False) / len(enc_out)

        if self.device_id >= 0:
            loss = loss.cuda(self.device_id)

        return loss

    def _encode(self, xs, x_lens, is_multi_task=False):
        """Encode acoustic features.
        Args:
            xs (list): A list of length `[B]`, which contains Variables of size `[T, input_size]`
            x_lens (list): A list of length `[B]`
            is_multi_task (bool): Set True in the case of MTL
        Returns:
            xs (torch.autograd.Variable, float): A tensor of size
                `[B, T, encoder_num_units]`
            x_lens (list): A tensor of size `[B]`
            OPTION:
                xs_sub (torch.autograd.Variable, float): A tensor of size
                    `[B, T, encoder_num_units]`
                x_lens_sub (list): A tensor of size `[B]`
        """
        # Convert list to Variables
        xs = pad_list(xs)

        if is_multi_task:
            if self.encoder_type == 'cnn':
                xs, x_lens = self.encoder(xs, x_lens)
                xs_sub = xs.clone()
                x_lens_sub = copy.deepcopy(x_lens)
            else:
                xs, x_lens, xs_sub, x_lens_sub = self.encoder(
                    xs, x_lens, volatile=not self.training)
        else:
            if self.encoder_type == 'cnn':
                xs, x_lens = self.encoder(xs, x_lens)
            else:
                xs, x_lens = self.encoder(
                    xs, x_lens, volatile=not self.training)

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
        batch_size, max_time_out, max_time_in = aw.size()
        raise NotImplementedError

    def _decode_train(self, enc_out, x_lens, ys, task, dir):
        """Decoding in the training stage.
        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, encoder_num_units]`
            x_lens (list): A list of length `[B]`
            ys (torch.autograd.Variable, long): A tensor of size `[B, L]`,
                which should be padded with <EOS>.
            task (int): the index of a task
            dir (str): fwd or bwd
        Returns:
            logits (torch.autograd.Variable, float): A tensor of size
                `[B, L, num_classes]`
            aw (torch.autograd.Variable, float): A tensor of size
                `[B, L, T, num_heads]`
            logits_rnnlm (torch.autograd.Variable, float): A tensor of size
                `[B, L, num_classes]`
        """
        batch_size, max_time = enc_out.size()[:2]
        taskdir = '_' + str(task) + '_' + dir

        # Initialization
        dec_state, dec_out = self._init_dec_state(enc_out, x_lens, task, dir)
        getattr(self, 'attend_' + str(task) + '_' + dir).reset()
        aw_step = None
        if self.decoding_order == 'luong':
            context_vec = Variable(enc_out.data.new(
                batch_size, 1, enc_out.size(-1)).fill_(0.))
        rnnlm_state = None

        # Pre-computation of embedding
        ys_emb = getattr(self, 'embed_' + str(task))(ys)
        if self.rnnlm_fusion_type:
            ys_rnnlm_emb = [getattr(self, 'rnnlm' + taskdir).embed(ys[:, t:t + 1])
                            for t in range(ys.size(1))]
            ys_rnnlm_emb = torch.cat(ys_rnnlm_emb, dim=1)

        logits, aw, logits_rnnlm = [], [], []
        for t in range(ys.size(1)):
            # Sample for scheduled sampling
            is_sample = self.ss_prob > 0 and t > 0 and self._step > 0 and random.random() < self._ss_prob
            if is_sample:
                y = getattr(self, 'embed_' + str(task))(
                    torch.max(logits[-1], dim=2)[1]).detach()
            else:
                y = ys_emb[:, t:t + 1]

            # Update RNNLM states
            if self.rnnlm_fusion_type:
                if is_sample:
                    y_rnnlm = getattr(self, 'rnnlm' + taskdir).embed(
                        torch.max(logits[-1], dim=2)[1]).detach()
                else:
                    y_rnnlm = ys_rnnlm_emb[:, t:t + 1]
                rnnlm_out, rnnlm_state = getattr(self, 'rnnlm' + taskdir).rnn(
                    y_rnnlm, hx=rnnlm_state)
                logits_step_rnnlm = getattr(self, 'rnnlm' + taskdir).output(
                    rnnlm_out)
                logits_rnnlm += [logits_step_rnnlm]

            if self.decoding_order == 'bahdanau':
                if t > 0:
                    # Recurrency
                    if self.rnnlm_fusion_type and self.concat_embedding:
                        y = torch.cat([y, y_rnnlm], dim=-1)
                    dec_in = torch.cat([y, context_vec], dim=-1)
                    dec_out, dec_state = getattr(self, 'decoder' + taskdir)(
                        dec_in, dec_state)

                # Score
                context_vec, aw_step = getattr(self, 'attend' + taskdir)(
                    enc_out, x_lens, dec_out, aw_step)

            elif self.decoding_order == 'luong':
                # Recurrency
                if self.rnnlm_fusion_type and self.concat_embedding:
                    y = torch.cat([y, y_rnnlm], dim=-1)
                dec_in = torch.cat([y, context_vec], dim=-1)
                dec_out, dec_state = getattr(self, 'decoder' + taskdir)(
                    dec_in, dec_state)

                # Score
                context_vec, aw_step = getattr(self, 'attend' + taskdir)(
                    enc_out, x_lens, dec_out, aw_step)

            elif self.decoding_order == 'conditional':
                # Recurrency of the first decoder
                if self.rnnlm_fusion_type and self.concat_embedding:
                    y = torch.cat([y, y_rnnlm], dim=-1)
                _dec_out, _dec_state = getattr(self, 'decoder_first' + taskdir)(
                    y, dec_state)

                # Score
                context_vec, aw_step = getattr(self, 'attend' + taskdir)(
                    enc_out, x_lens, _dec_out, aw_step)

                # Recurrency of the second decoder
                dec_out, dec_state = getattr(self, 'decoder_second' + taskdir)(
                    context_vec, _dec_state)

            else:
                raise ValueError(self.decoding_order)

            # Generate
            logits_step = getattr(self, 'W_d' + taskdir)(dec_out) + \
                getattr(self, 'W_c' + taskdir)(context_vec)

            # RNNLM fusion
            if self.rnnlm_fusion_type == 'cold_fusion':
                logits_step_rnnlm = getattr(self, 'W_rnnlm_logits' + taskdir)(
                    logits_step_rnnlm)
                gate = F.sigmoid(getattr(self, 'W_rnnlm_gate' + taskdir)(
                    torch.cat([dec_out, logits_step_rnnlm], dim=-1)))
                logits_step += gate * logits_step_rnnlm
                non_linear = F.relu
            else:
                non_linear = F.tanh
            logits_step = getattr(self, 'fc' + taskdir)(
                non_linear(logits_step))
            if self.rnnlm_fusion_type == 'logits_fusion':
                logits_step += logits_step_rnnlm

            logits += [logits_step]
            if self.coverage_weight > 0:
                aw += [aw_step]

        logits = torch.cat(logits, dim=1)
        if self.coverage_weight > 0:
            aw = torch.stack(aw, dim=1)
        if self.rnnlm_fusion_type:
            logits_rnnlm = torch.cat(logits_rnnlm, dim=1)

        return logits, aw, logits_rnnlm

    def _init_dec_state(self, enc_out, x_lens, task, dir):
        """Initialize decoder state.
        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, encoder_num_units]`
            x_lens (list): A list of length `[B]`
            task (int): the index of a task
            dir (str): fwd or bwd
        Returns:
            dec_state (list or tuple of list):
            dec_out (torch.autograd.Variable, float): A tensor of size
                `[B, 1, decoder_num_units]`
        """
        taskdir = '_' + str(task) + '_' + dir

        zero_state = Variable(enc_out.data.new(
            enc_out.size(0), getattr(self, 'decoder_num_units_' + str(task))).fill_(0.),
            volatile=not self.training)

        dec_out = Variable(enc_out.data.new(
            enc_out.size(0), 1, getattr(self, 'decoder_num_units_' + str(task))).fill_(0.),
            volatile=not self.training)

        if getattr(self, 'init_dec_state' + taskdir) == 'zero':
            if self.decoder_type == 'lstm':
                hx_list = [zero_state] * \
                    getattr(self, 'decoder_num_layers_' + str(task))
                cx_list = [zero_state] * \
                    getattr(self, 'decoder_num_layers_' + str(task))
            else:
                hx_list = [zero_state] * \
                    getattr(self, 'decoder_num_layers_' + str(task))

        else:
            # TODO: consider x_lens

            if getattr(self, 'init_dec_state' + taskdir) == 'mean':
                # Initialize with mean of all encoder outputs
                h_0 = enc_out.mean(dim=1, keepdim=False)
            elif getattr(self, 'init_dec_state' + taskdir) == 'final':
                # Initialize with the final encoder output
                h_0 = enc_out[:, -1, :]
            elif getattr(self, 'init_dec_state' + taskdir) == 'first':
                # Initialize with the first encoder output
                h_0 = enc_out[:, 0, :]
            # NOTE: h_0: `[B, encoder_num_units]`

            # Path through the linear layer
            h_0 = F.tanh(getattr(self, 'W_dec_init_' +
                                 str(task) + '_' + dir)(h_0))

            hx_list = [h_0] * \
                getattr(self, 'decoder_num_layers_' + str(task))
            # NOTE: all layers are initialized with the same values

            if self.decoder_type == 'lstm':
                cx_list = [zero_state] * \
                    getattr(self, 'decoder_num_layers_' + str(task))

            dec_out = h_0.unsqueeze(1)

        if self.decoder_type == 'lstm':
            dec_state = (hx_list, cx_list)
        else:
            dec_state = hx_list

        return dec_state, dec_out

    def decode(self, xs, beam_width, max_decode_len, min_decode_len=0,
               min_decode_len_ratio=0, length_penalty=0, coverage_penalty=0,
               rnnlm_weight=0, task_index=0, resolving_unk=False):
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
            task_index (int): not used (to make compatible)
            resolving_unk (bool): not used (to make compatible)
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw (list): A list of length `[B]`, which contains arrays of size `[L, T]`
            perm_idx (list): A list of length `[B]`
        """
        self.eval()

        # Sort by lenghts in the descending order
        if self.encoder_type != 'cnn':
            perm_idx = sorted(list(range(0, len(xs), 1)),
                              key=lambda i: xs[i].shape[0], reverse=True)
            xs = [xs[i] for i in perm_idx]
            # NOTE: must be descending order for pack_padded_sequence
        else:
            perm_idx = list(range(0, len(xs), 1))

        # Frame stacking
        if self.num_stack > 1:
            xs = [stack_frame(x, self.num_stack, self.num_skip)
                  for x in xs]

        # Splicing
        if self.splice > 1:
            xs = [do_splice(x, self.splice, self.num_stack) for x in xs]

        # Wrap by Variable
        xs = [np2var(x, self.device_id, volatile=True).float() for x in xs]
        x_lens = [len(x) for x in xs]

        # Encode acoustic features
        enc_out, x_lens = self._encode(xs, x_lens)

        dir = 'fwd' if self.fwd_weight_0 >= self.bwd_weight_0 else 'bwd'

        if beam_width == 1:
            best_hyps, aw = self._decode_infer_greedy(
                enc_out, x_lens, max_decode_len, task=0, dir=dir)
        else:
            best_hyps, aw = self._decode_infer_beam(
                enc_out, x_lens, beam_width, max_decode_len, min_decode_len,
                min_decode_len_ratio, length_penalty, coverage_penalty,
                rnnlm_weight, task=0, dir=dir)

        return best_hyps, aw, perm_idx

    def _decode_infer_greedy(self, enc_out, x_lens, max_decode_len, task, dir):
        """Greedy decoding in the inference stage.
        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, encoder_num_units]`
            x_lens (list): A list of length `[B]`
            max_decode_len (int): the maximum sequence length of tokens
            task (int): the index of a task
            dir (str): fwd or bwd
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw (list): A list of length `[B]`, which contains arrays of size `[L, T]`
        """
        if dir == 'bwd':
            assert getattr(self, 'bwd_weight_' + str(task)) > 0

        batch_size, max_time = enc_out.size()[:2]
        taskdir = '_' + str(task) + '_' + dir

        # Initialization
        dec_state, dec_out = self._init_dec_state(enc_out, x_lens, task, dir)
        getattr(self, 'attend_' + str(task) + '_' + dir).reset()
        aw_step = None
        if self.decoding_order == 'luong':
            context_vec = Variable(enc_out.data.new(
                batch_size, 1, enc_out.size(-1)).fill_(0.), volatile=True)
        rnnlm_state = None

        # Start from <SOS>
        sos = getattr(self, 'sos_' + str(task))
        eos = getattr(self, 'eos_' + str(task))
        y = Variable(enc_out.data.new(
            batch_size, 1).fill_(sos).long(), volatile=True)

        _best_hyps, _aw = [], []
        y_lens = np.zeros((batch_size,), dtype=np.int32)
        eos_flag = [False] * batch_size
        for t in range(max_decode_len + 1):
            # Update RNNLM states
            if self.rnnlm_fusion_type:
                y_rnnlm = getattr(self, 'rnnlm' + taskdir).embed(y)
                rnnlm_out, rnnlm_state = getattr(self, 'rnnlm' + taskdir).rnn(
                    y_rnnlm, hx=rnnlm_state)
                logits_step_rnnlm = getattr(self, 'rnnlm' + taskdir).output(
                    rnnlm_out)

            y = getattr(self, 'embed_' + str(task))(y)

            if self.decoding_order == 'bahdanau':
                if t > 0:
                    # Recurrency
                    if self.rnnlm_fusion_type and self.concat_embedding:
                        y = torch.cat([y, y_rnnlm], dim=-1)
                    dec_in = torch.cat([y, context_vec], dim=-1)
                    dec_out, dec_state = getattr(self, 'decoder' + taskdir)(
                        dec_in, dec_state)

                # Score
                context_vec, aw_step = getattr(self, 'attend' + taskdir)(
                    enc_out, x_lens, dec_out, aw_step)

            elif self.decoding_order == 'luong':
                # Recurrency
                if self.rnnlm_fusion_type and self.concat_embedding:
                    y = torch.cat([y, y_rnnlm], dim=-1)
                dec_in = torch.cat([y, context_vec], dim=-1)
                dec_out, dec_state = getattr(self, 'decoder' + taskdir)(
                    dec_in, dec_state)

                # Score
                context_vec, aw_step = getattr(self, 'attend' + taskdir)(
                    enc_out, x_lens, dec_out, aw_step)

            elif self.decoding_order == 'conditional':
                # Recurrency of the first decoder
                if self.rnnlm_fusion_type and self.concat_embedding:
                    y = torch.cat([y, y_rnnlm], dim=-1)
                _dec_out, _dec_state = getattr(self, 'decoder_first' + taskdir)(
                    y, dec_state)

                # Score
                context_vec, aw_step = getattr(self, 'attend' + taskdir)(
                    enc_out, x_lens, _dec_out, aw_step)

                # Recurrency of the second decoder
                dec_out, dec_state = getattr(self, 'decoder_second' + taskdir)(
                    context_vec, _dec_state)

            else:
                raise ValueError(self.decoding_order)

            # Generate
            logits_step = getattr(self, 'W_d' + taskdir)(dec_out) + \
                getattr(self, 'W_c' + taskdir)(context_vec)

            # RNNLM fusion
            if self.rnnlm_fusion_type == 'cold_fusion':
                rnnlm_feat = getattr(self, 'W_rnnlm_logits' + taskdir)(
                    logits_step_rnnlm)
                gate = F.sigmoid(getattr(self, 'W_rnnlm_gate' + taskdir)(
                    torch.cat([dec_out, rnnlm_feat], dim=-1)))
                logits_step += gate * rnnlm_feat
                non_linear = F.relu
            else:
                non_linear = F.tanh
            logits_step = getattr(self, 'fc' + taskdir)(
                non_linear(logits_step))
            if self.rnnlm_fusion_type == 'logits_fusion':
                logits_step += logits_step_rnnlm

            # Pick up 1-best
            y = torch.max(logits_step.squeeze(1), dim=1)[1].unsqueeze(1)
            _best_hyps += [y]
            _aw += [aw_step]

            # Count lengths of hypotheses
            for b in range(batch_size):
                if not eos_flag[b]:
                    if y.data.cpu().numpy()[b] == eos:
                        eos_flag[b] = True
                    y_lens[b] += 1
                    # NOTE: include <EOS>

            # Break if <EOS> is outputed in all mini-batch
            if sum(eos_flag) == batch_size:
                break

        # Concatenate in L dimension
        _best_hyps = torch.cat(_best_hyps, dim=1)
        _aw = torch.stack(_aw, dim=1)

        # Convert to numpy
        _best_hyps = var2np(_best_hyps)
        _aw = var2np(_aw)

        if getattr(self, 'num_heads_' + str(task)) > 1:
            _aw = _aw[:, :, :, 0]
            # TODO: fix for MHA

        # Truncate by <EOS>
        best_hyps, aw = [], []
        for b in range(batch_size):
            if dir == 'bwd':
                # Reverse the order
                best_hyps += [_best_hyps[b, :y_lens[b]][::-1]]
                aw += [_aw[b, :y_lens[b]][::-1]]
            else:
                best_hyps += [_best_hyps[b, :y_lens[b]]]
                aw += [_aw[b, :y_lens[b]]]

        return best_hyps, aw

    def _decode_infer_beam(self, enc_out, x_lens, beam_width,
                           max_decode_len, min_decode_len, min_decode_len_ratio,
                           length_penalty, coverage_penalty, rnnlm_weight,
                           task, dir):
        """Beam search decoding in the inference stage.
        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, encoder_num_units]`
            x_lens (list): A list of length `[B]`
            beam_width (int): the size of beam
            max_decode_len (int): the maximum sequence length of tokens
            min_decode_len (int): the minimum sequence length of tokens
            min_decode_len_ratio (float):
            length_penalty (float): length penalty
            coverage_penalty (float): coverage penalty
            rnnlm_weight (float): the weight of RNNLM score
            task (int): the index of a task
            dir (str): fwd or bwd
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw (list): A list of length `[B]`, which contains arrays of size `[L, T]`
        """
        if dir == 'bwd':
            assert getattr(self, 'bwd_weight_' + str(task)) > 0
        taskdir = '_' + str(task) + '_' + dir

        if (rnnlm_weight > 0 or self.rnnlm_fusion_type) and getattr(self, 'rnnlm' + taskdir) is not None:
            assert not getattr(self, 'rnnlm' + taskdir).training

        # Start from <SOS>
        sos = getattr(self, 'sos_' + str(task))
        eos = getattr(self, 'eos_' + str(task))

        best_hyps, aw = [], []
        y_lens = np.zeros((enc_out.size(0),), dtype=np.int32)
        for b in range(enc_out.size(0)):
            # Initialization per utterance
            dec_state, dec_out = self._init_dec_state(
                enc_out[b:b + 1], x_lens[b], task, dir)
            context_vec = Variable(enc_out.data.new(
                1, 1, enc_out.size(-1)).fill_(0.), volatile=True)
            getattr(self, 'attend_' + str(task) + '_' + dir).reset()

            complete = []
            beam = [{'hyp': [sos],
                     'score': 0,  # log 1
                     'dec_state': dec_state,
                     'dec_out': dec_out,
                     'context_vec': context_vec,
                     'aw_steps': [None],
                     'rnnlm_state': None,
                     'previous_coverage': 0}]
            for t in range(max_decode_len + 1):
                new_beam = []
                for i_beam in range(len(beam)):
                    # Update RNNLM states
                    if (rnnlm_weight > 0 or self.rnnlm_fusion_type) and getattr(self, 'rnnlm' + taskdir) is not None:
                        y_rnnlm = Variable(enc_out.data.new(
                            1, 1).fill_(beam[i_beam]['hyp'][-1]).long(), volatile=True)
                        y_rnnlm = getattr(self, 'rnnlm' + taskdir).embed(
                            y_rnnlm)
                        rnnlm_out, rnnlm_state = getattr(self, 'rnnlm' + taskdir).rnn(
                            y_rnnlm, hx=beam[i_beam]['rnnlm_state'])
                        logits_step_rnnlm = getattr(self, 'rnnlm' + taskdir).output(
                            rnnlm_out)

                    y = Variable(enc_out.data.new(
                        1, 1).fill_(beam[i_beam]['hyp'][-1]).long(), volatile=True)
                    y = getattr(self, 'embed_' + str(task))(y)

                    if self.decoding_order == 'bahdanau':
                        if t == 0:
                            dec_out = beam[i_beam]['dec_out']
                        else:
                            # Recurrency
                            if self.rnnlm_fusion_type and self.concat_embedding:
                                y = torch.cat([y, y_rnnlm], dim=-1)
                            dec_in = torch.cat(
                                [y, beam[i_beam]['context_vec']], dim=-1)
                            dec_out, dec_state = getattr(self, 'decoder' + taskdir)(
                                dec_in, beam[i_beam]['dec_state'])

                        # Score
                        context_vec, aw_step = getattr(self, 'attend' + taskdir)(
                            enc_out[b:b + 1, :x_lens[b]], x_lens[b:b + 1],
                            dec_out, beam[i_beam]['aw_steps'][-1])

                    elif self.decoding_order == 'luong':
                        # Recurrency
                        if self.rnnlm_fusion_type and self.concat_embedding:
                            y = torch.cat([y, y_rnnlm], dim=-1)
                        dec_in = torch.cat(
                            [y, beam[i_beam]['context_vec']], dim=-1)
                        dec_out, dec_state = getattr(self, 'decoder' + taskdir)(
                            dec_in, beam[i_beam]['dec_state'])

                        # Score
                        context_vec, aw_step = getattr(self, 'attend' + taskdir)(
                            enc_out[b:b + 1, :x_lens[b]], x_lens[b:b + 1],
                            dec_out, beam[i_beam]['aw_steps'][-1])

                    elif self.decoding_order == 'conditional':
                        # Recurrency of the first decoder
                        if self.rnnlm_fusion_type and self.concat_embedding:
                            y = torch.cat([y, y_rnnlm], dim=-1)
                        _dec_out, _dec_state = getattr(self, 'decoder_first' + taskdir)(
                            y, beam[i_beam]['dec_state'])

                        # Score
                        context_vec, aw_step = getattr(self, 'attend' + taskdir)(
                            enc_out[b:b + 1, :x_lens[b]], x_lens[b:b + 1],
                            _dec_out, beam[i_beam]['aw_steps'][-1])

                        # Recurrency of the second decoder
                        dec_out, dec_state = getattr(self, 'decoder_second' + taskdir)(
                            context_vec, _dec_state)

                    else:
                        raise ValueError(self.decoding_order)

                    # Generate
                    logits_step = getattr(self, 'W_d' + taskdir)(dec_out) +\
                        getattr(self, 'W_c' + taskdir)(context_vec)

                    # RNNLM fusion
                    if self.rnnlm_fusion_type == 'cold_fusion':
                        rnnlm_feat = getattr(self, 'W_rnnlm_logits' + taskdir)(
                            logits_step_rnnlm)
                        gate = F.sigmoid(getattr(self, 'W_rnnlm_gate' + taskdir)(
                            torch.cat([dec_out, rnnlm_feat], dim=-1)))
                        logits_step += gate * rnnlm_feat
                        non_linear = F.relu
                    else:
                        non_linear = F.tanh
                    logits_step = getattr(self, 'fc' + taskdir)(
                        non_linear(logits_step))
                    if self.rnnlm_fusion_type == 'logits_fusion':
                        logits_step += logits_step_rnnlm

                    # Path through the softmax layer & convert to log-scale
                    log_probs = F.log_softmax(logits_step.squeeze(1), dim=1)
                    # log_probs = logits_step.squeeze(1)
                    # NOTE: `[1 (B), 1, num_classes]` -> `[1 (B), num_classes]`

                    # Pick up the top-k scores
                    log_probs_topk, indices_topk = torch.topk(
                        log_probs, k=beam_width, dim=1, largest=True, sorted=True)

                    for k in range(beam_width):
                        # Exclude short hypotheses
                        if indices_topk[0, k].data[0] == eos and len(beam[i_beam]['hyp']) < min_decode_len:
                            continue
                        if indices_topk[0, k].data[0] == eos and len(beam[i_beam]['hyp']) < x_lens[b] * min_decode_len_ratio:
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

                            if getattr(self, 'num_heads_' + str(task)) > 1:
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
                        if rnnlm_weight > 0 and getattr(self, 'rnnlm' + taskdir) is not None:
                            rnnlm_log_probs = F.log_softmax(
                                logits_step_rnnlm.squeeze(1), dim=1)
                            assert log_probs.size() == rnnlm_log_probs.size()
                            score += rnnlm_log_probs.data[0,
                                                          indices_topk.data[0, k]] * rnnlm_weight
                        else:
                            rnnlm_state = None

                        new_beam.append(
                            {'hyp': beam[i_beam]['hyp'] + [indices_topk.data[0, k]],
                             'score': score,
                             'dec_state': copy.deepcopy(dec_state),
                             'dec_out': dec_out,
                             'context_vec': context_vec,
                             'aw_steps': beam[i_beam]['aw_steps'] + [aw_step],
                             'rnnlm_state': rnnlm_state,
                             'previous_coverage': cov_sum})

                new_beam = sorted(
                    new_beam, key=lambda x: x['score'], reverse=True)

                # Remove complete hypotheses
                not_complete = []
                for cand in new_beam[:beam_width]:
                    if cand['hyp'][-1] == eos:
                        complete += [cand]
                    else:
                        not_complete += [cand]

                if len(complete) >= beam_width:
                    complete = complete[:beam_width]
                    break

                beam = not_complete[:beam_width]

            if len(complete) == 0:
                complete = beam

            complete = sorted(complete, key=lambda x: x['score'], reverse=True)
            best_hyps += [np.array(complete[0]['hyp'][1:])]
            aw += [complete[0]['aw_steps'][1:]]
            y_lens[b] = len(complete[0]['hyp'][1:])

        # Concatenate in L dimension
        for b in range(len(aw)):
            aw[b] = var2np(torch.stack(aw[b], dim=1).squeeze(0))
            if getattr(self, 'num_heads_' + str(task)) > 1:
                aw[b] = aw[b][:, :, 0]
                # TODO: fix for MHA

        # Reverse the order
        if dir == 'bwd':
            for b in range(enc_out.size(0)):
                best_hyps[b] = best_hyps[b][::-1]
                aw[b] = aw[b][::-1]

        return best_hyps, aw

    def decode_ctc(self, xs, beam_width=1, task_index=0):
        """Decoding by the CTC layer in the inference stage.
            This is only used for Joint CTC-Attention model.
        Args:
            xs (list): A list of length `[B]`, which contains arrays of size `[T, input_size]`
            beam_width (int): the size of beam
            task_index (int): the index of a task
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            perm_idx (list): A list of length `[B]`
        """
        self.eval()

        # Sort by lenghts in the descending order
        if self.encoder_type != 'cnn':
            perm_idx = sorted(list(range(0, len(xs), 1)),
                              key=lambda i: xs[i].shape[0], reverse=True)
            xs = [xs[i] for i in perm_idx]
            # NOTE: must be descending order for pack_padded_sequence
        else:
            perm_idx = list(range(0, len(xs), 1))

        # Frame stacking
        if self.num_stack > 1:
            xs = [stack_frame(x, self.num_stack, self.num_skip)
                  for x in xs]

        # Splicing
        if self.splice > 1:
            xs = [do_splice(x, self.splice, self.num_stack) for x in xs]

        # Wrap by Variable
        xs = [np2var(x, self.device_id, volatile=True).float() for x in xs]
        x_lens = [len(x) for x in xs]

        # Encode acoustic features
        if task_index == 0:
            enc_out, x_lens = self._encode(xs, x_lens)
        elif task_index == 1:
            _, _, enc_out, x_lens = self._encode(
                xs, x_lens, is_multi_task=True)
        else:
            raise NotImplementedError

        # Path through the softmax layer
        batch_size, max_time = enc_out.size()[:2]
        enc_out = enc_out.view(batch_size * max_time, -1).contiguous()
        logits_ctc = getattr(self, 'fc_ctc_' + str(task_index))(enc_out)
        logits_ctc = logits_ctc.view(batch_size, max_time, -1)

        if beam_width == 1:
            best_hyps = self._decode_ctc_greedy_np(var2np(logits_ctc), x_lens)
        else:
            best_hyps = self._decode_ctc_beam_np(
                var2np(F.log_softmax(logits_ctc, dim=-1)),
                x_lens, beam_width=beam_width)

        # NOTE: index 0 is reserved for blank in warpctc_pytorch
        best_hyps -= 1

        return best_hyps, perm_idx
