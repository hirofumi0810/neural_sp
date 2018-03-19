#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention-based sequence-to-sequence model (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import warpctc_pytorch
except:
    pass
    # raise ImportError('Install warpctc_pytorch.')

import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _assert_no_grad

from models.pytorch.base import ModelBase
from models.pytorch.criterion import cross_entropy_label_smoothing
from models.pytorch.linear import LinearND, Embedding, Embedding_LS
from models.pytorch.encoders.load_encoder import load
# from models.pytorch.attention.rnn_decoder import RNNDecoder
from models.pytorch.attention.rnn_decoder_nstep import RNNDecoder
from models.pytorch.attention.attention_layer import AttentionMechanism
from models.pytorch.ctc.ctc import _concatenate_labels
from models.pytorch.ctc.decoders.greedy_decoder import GreedyDecoder
from models.pytorch.ctc.decoders.beam_search_decoder import BeamSearchDecoder

LOG_1 = 0


# [Referrence]
# https://github.com/espnet/espnet/blob/master/src/nets/e2e_asr_attctc_th.py

# class _ChainerLikeCTC(warpctc_pytorch._CTC):
#     @staticmethod
#     def forward(ctx, acts, labels, act_lens, label_lens):
#         is_cuda = True if acts.is_cuda else False
#         acts = acts.contiguous()  # `[T, B, num_classes]`
#         loss_func = warpctc_pytorch.gpu_ctc if is_cuda else warpctc_pytorch.cpu_ctc
#         grads = torch.zeros(acts.size()).type_as(acts)
#         minibatch_size = acts.size(1)
#         costs = torch.zeros(minibatch_size).cpu()
#         loss_func(acts,
#                   grads,
#                   labels,
#                   label_lens,
#                   act_lens,
#                   minibatch_size,
#                   costs)
#         # modified only here from original
#         # costs = torch.FloatTensor([costs.sum()]) / acts.size(1)
#         costs = torch.FloatTensor([costs.sum()])
#         # NOTE: normalize by batch_size later
#         ctx.grads = Variable(grads)
#         ctx.grads /= ctx.grads.size(1)
#
#         return costs


def chainer_like_ctc_loss(acts, labels, act_lens, label_lens):
    """Chainer like CTC Loss
    acts: Tensor of (seqLength x batch x outputDim) containing output from network
    labels: 1 dimensional Tensor containing all the targets of the batch in one sequence
    act_lens: Tensor of size (batch) containing size of each output sequence from the network
    act_lens: Tensor of (batch) containing label length of each example
    """
    # assert len(labels.size()) == 1  # labels must be 1 dimensional
    # _assert_no_grad(labels)
    # _assert_no_grad(act_lens)
    # _assert_no_grad(label_lens)
    return _ChainerLikeCTC.apply(acts, labels, act_lens, label_lens)


# warpctc = warpctc_pytorch.CTCLoss()
# chainer_like_warpctc = chainer_like_ctc_loss


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
            first => initialize with tha first encoder state
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
        decoding_order (string, optional): attend_spell or spell_attend or conditional
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
                 decoder_dense_residual=False,
                 decoding_order='spell_attend'):

        super(ModelBase, self).__init__()
        self.model_type = 'attention'

        # TODO: clip_activation

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
        self.attention_type = attention_type
        self.attention_dim = attention_dim
        self.decoder_type = decoder_type
        self.decoder_num_units = decoder_num_units
        self.decoder_num_layers = decoder_num_layers
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes + 1  # Add <EOS> class
        self.sos = num_classes
        self.eos = num_classes
        # NOTE: <SOS> and <EOS> have the same index

        # Setting for the attention
        if init_dec_state not in ['zero', 'mean', 'final', 'first']:
            raise ValueError(
                'init_dec_state must be "zero" or "mean" or "final" or "first".')
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

        # Setting for MTL
        self.ctc_loss_weight = ctc_loss_weight

        self.decoding_order = decoding_order

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
                activation=activation,
                batch_norm=batch_norm)
            self.init_dec_state = 'zero'
        else:
            raise NotImplementedError

        if encoder_type != decoder_type:
            self.init_dec_state = 'zero'

        if self.init_dec_state != 'zero':
            self.W_dec_init = LinearND(
                self.encoder_num_units, decoder_num_units)

        ##############################
        # Decoder
        ##############################
        if decoding_order == 'conditional':
            self.decoder_first = RNNDecoder(
                input_size=embedding_dim,
                rnn_type=decoder_type,
                num_units=decoder_num_units,
                num_layers=decoder_num_layers,
                dropout=dropout_decoder,
                residual=decoder_residual,
                dense_residual=decoder_dense_residual)
            self.decoder_second = RNNDecoder(
                input_size=self.encoder_num_units,
                rnn_type=decoder_type,
                num_units=decoder_num_units,
                num_layers=decoder_num_layers,
                dropout=dropout_decoder,
                residual=decoder_residual,
                dense_residual=decoder_dense_residual)
        else:
            self.decoder = RNNDecoder(
                input_size=self.encoder_num_units + embedding_dim,
                rnn_type=decoder_type,
                num_units=decoder_num_units,
                num_layers=decoder_num_layers,
                dropout=dropout_decoder,
                residual=decoder_residual,
                dense_residual=decoder_dense_residual)

        ##############################
        # Attention layer
        ##############################
        self.attend = AttentionMechanism(
            encoder_num_units=self.encoder_num_units,
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
        self.is_bridge = False
        if encoder_type == 'cnn':
            self.bridge = LinearND(
                self.encoder.output_size, decoder_num_units,
                dropout=dropout_encoder)
            self.is_bridge = True

        ##############################
        # Embedding
        ##############################
        if label_smoothing_prob > 0:
            self.embed = Embedding_LS(num_classes=self.num_classes,
                                      embedding_dim=embedding_dim,
                                      dropout=dropout_embedding,
                                      label_smoothing_prob=label_smoothing_prob)
        else:
            self.embed = Embedding(num_classes=self.num_classes,
                                   embedding_dim=embedding_dim,
                                   dropout=dropout_embedding)

        ##############################
        # Output layer
        ##############################
        self.W_d = LinearND(decoder_num_units, decoder_num_units,
                            dropout=dropout_decoder)
        self.W_c = LinearND(self.encoder_num_units, decoder_num_units,
                            dropout=dropout_decoder)
        # self.W_y = LinearND(embedding_dim, decoder_num_units,
        #                     dropout=dropout_decoder)
        self.fc = LinearND(decoder_num_units, self.num_classes)

        ##############################
        # CTC
        ##############################
        if ctc_loss_weight > 0:
            if self.is_bridge:
                self.fc_ctc = LinearND(
                    decoder_num_units, num_classes + 1)
            else:
                self.fc_ctc = LinearND(
                    self.encoder_num_units, num_classes + 1)

            # Set CTC decoders
            self._decode_ctc_greedy_np = GreedyDecoder(blank_index=0)
            self._decode_ctc_beam_np = BeamSearchDecoder(blank_index=0)
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

    def forward(self, xs, ys, x_lens, y_lens, is_eval=False):
        """Forward computation.
        Args:
            xs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            ys (np.ndarray): A tensor of size `[B, T_out]`, which should be padded with -1.
            x_lens (np.ndarray): A tensor of size `[B]`
            y_lens (np.ndarray): A tensor of size `[B]`
            is_eval (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            loss (torch.autograd.Variable(float) or float): A tensor of size `[1]`
        """
        if is_eval:
            self.eval()
        else:
            self.train()

            # Gaussian noise injection
            if self.weight_noise_injection:
                self.inject_weight_noise(mean=0, std=self.weight_noise_std)

        # NOTE: ys is padded with -1 here
        # ys_in is padded with <EOS> in order to convert to one-hot vector,
        # and added <SOS> before the first token
        # ys_out is padded with -1, and added <EOS> after the last token
        ys_in = self._create_var((ys.shape[0], ys.shape[1] + 1),
                                 fill_value=self.eos, dtype='long')
        ys_out = self._create_var((ys.shape[0], ys.shape[1] + 1),
                                  fill_value=-1, dtype='long')
        for b in range(len(xs)):
            ys_in.data[b, 0] = self.sos
            ys_in.data[b, 1:y_lens[b] + 1] = torch.from_numpy(
                ys[b, :y_lens[b]])

            ys_out.data[b, :y_lens[b]] = torch.from_numpy(ys[b, :y_lens[b]])
            ys_out.data[b, y_lens[b]] = self.eos

        if self.use_cuda:
            ys_in = ys_in.cuda()
            ys_out = ys_out.cuda()

        # Wrap by Variable
        xs = self.np2var(xs)
        x_lens = self.np2var(x_lens, dtype='int')
        y_lens = self.np2var(y_lens, dtype='int')

        # Encode acoustic features
        xs, x_lens, perm_idx = self._encode(xs, x_lens)

        # Permutate indices
        if perm_idx is not None:
            ys_in = ys_in[perm_idx]
            ys_out = ys_out[perm_idx]
            y_lens = y_lens[perm_idx]

        # Compute XE loss
        loss = self.compute_xe_loss(
            xs, ys_in, ys_out, x_lens, y_lens, size_average=True)

        # Auxiliary CTC loss (optional)
        if self.ctc_loss_weight > 0:
            ctc_loss = self.compute_ctc_loss(
                xs, ys_in[:, 1:] + 1,
                x_lens, y_lens, size_average=True)
            # NOTE: exclude <SOS>
            # NOTE: index 0 is reserved for blank in warpctc_pytorch
            loss = loss * (1 - self.ctc_loss_weight) + \
                ctc_loss * self.ctc_loss_weight

        if is_eval:
            loss = loss.data[0]
        else:
            # Update the probability of scheduled sampling
            self._step += 1
            if self.sample_prob > 0:
                self._sample_prob = min(
                    self.sample_prob,
                    self.sample_prob / self.sample_ramp_max_step * self._step)

        return loss

    def compute_xe_loss(self, enc_out, ys_in, ys_out, x_lens, y_lens,
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
            is_sub_task (bool, optional):
            size_average (bool, optional):
        Returns:
            loss (torch.autograd.Variable, float): A tensor of size `[1]`
        """
        # Teacher-forcing
        logits, aw = self._decode_train(
            enc_out, x_lens, ys_in, is_sub_task=is_sub_task)

        # Output smoothing
        if self.logits_temperature != 1:
            logits /= self.logits_temperature

        # Compute XE sequence loss
        loss = F.cross_entropy(
            input=logits.view((-1, logits.size(2))),
            target=ys_out.view(-1),
            ignore_index=-1, size_average=False) / len(enc_out)
        # NOTE: ys_in are padded by <SOS>

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

    def compute_ctc_loss(self, enc_out, ys, x_lens, y_lens,
                         is_sub_task=False, size_average=False):
        """Compute CTC loss.
        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T_in, encoder_num_units]`
            ys (torch.autograd.Variable, long): A tensor of size `[B, T_out]`,
                which includes <SOS> nor <EOS>
            x_lens (torch.autograd.Variable, int): A tensor of size `[B]`
            y_lens (torch.autograd.Variable, int): A tensor of size `[B]`,
                which includes <SOS> nor <EOS>
            is_sub_task (bool, optional):
            size_average (bool, optional):
        Returns:
            loss (torch.autograd.Variable, float): A tensor of size `[1]`
        """
        # Concatenate all _ys for warpctc_pytorch
        # `[B, T_out]` -> `[1,]`
        concatenated_labels = _concatenate_labels(ys, y_lens)

        # Path through the fully-connected layer
        if is_sub_task:
            logits = self.fc_ctc_sub(enc_out)
        else:
            logits = self.fc_ctc(enc_out)

        # Compute CTC loss
        # loss = warpctc(logits.transpose(0, 1),  # time-major
        #                concatenated_labels.cpu(),
        #                x_lens.cpu(),
        #                y_lens.cpu())

        loss = chainer_like_warpctc(logits.transpose(0, 1),  # time-major
                                    concatenated_labels.cpu(),
                                    x_lens.cpu(),
                                    y_lens.cpu())

        if self.use_cuda:
            loss = loss.cuda()

        # Label smoothing (with uniform distribution)
        # if self.label_smoothing_prob > 0:
        #     # XE
        #     loss_ls = cross_entropy_label_smoothing(
        #         logits,
        #         y_lens=x_lens,  # NOTE: CTC is frame-synchronous
        #         label_smoothing_prob=self.label_smoothing_prob,
        #         distribution='uniform',
        #         size_average=False)
        #     loss = loss * (1 - self.label_smoothing_prob) + \
        #         loss_ls
        #     # print(loss_ls)

        if size_average:
            loss /= len(enc_out)

        return loss

    def _encode(self, xs, x_lens, is_multi_task=False):
        """Encode acoustic features.
        Args:
            xs (torch.autograd.Variable, float): A tensor of size
                `[B, T_in, input_size]`
            x_lens (torch.autograd.Variable, int): A tensor of size `[B]`
            is_multi_task (bool, optional):
        Returns:
            xs (torch.autograd.Variable, float): A tensor of size
                `[B, T_in, encoder_num_units]`
            x_lens (torch.autograd.Variable, int): A tensor of size `[B]`
            OPTION:
                xs_sub (torch.autograd.Variable, float): A tensor of size
                    `[B, T_in, encoder_num_units]`
                x_lens_sub (torch.autograd.Variable, int): A tensor of size `[B]`
            perm_idx (torch.autograd.Variable, long): A tensor of size `[B]`
        """
        if is_multi_task:
            xs, x_lens, xs_sub, x_lens_sub, perm_idx = self.encoder(
                xs, x_lens, volatile=not self.training)
        else:
            if self.encoder_type == 'cnn':
                xs, x_lens = self.encoder(xs, x_lens)
                perm_idx = None
            else:
                xs, x_lens, perm_idx = self.encoder(
                    xs, x_lens, volatile=not self.training)
        # NOTE: xs: `[B, T_in, encoder_num_units * encoder_num_directions]`
        # xs_sub: `[B, T_in, encoder_num_units * encoder_num_directions]`

        # Bridge between the encoder and decoder in the main task
        if self.is_bridge:
            xs = self.bridge(xs)
            # NOTE: this is used for CNN encoder

        if is_multi_task:
            return xs, x_lens, xs_sub, x_lens_sub, perm_idx
        else:
            return xs, x_lens, perm_idx

    def _compute_coverage(self, aw):
        batch_size, max_time_outputs, max_time_inputs = aw.size()
        raise NotImplementedError

    def _decode_train(self, enc_out, x_lens, ys, is_sub_task=False):
        """Decoding in the training stage.
        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T_in, encoder_num_units]`
            x_lens (torch.autograd.Variable, int): A tensor of size `[B]`
            ys (torch.autograd.Variable, long): A tensor of size `[B, T_out]`,
                which should be padded with <EOS>.
            is_sub_task (bool, optional):
        Returns:
            logits (torch.autograd.Variable, float): A tensor of size
                `[B, T_out, num_classes]`
            aw (torch.autograd.Variable, float): A tensor of size
                `[B, T_out, T_in]`
        """
        batch_size = enc_out.size(0)

        # Initialize decoder state, decoder output, attention_weights
        dec_state = self._init_decoder_state(enc_out, is_sub_task=is_sub_task)
        dec_out = self._create_var((batch_size, 1, self.decoder_num_units))
        aw_step = self._create_var((batch_size, enc_out.size(1)))

        if self.decoding_order == 'spell_attend':
            context_vec = self._create_var(
                (batch_size, 1, self.encoder_num_units))

        logits = []
        aw = []
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
                if self.decoding_order == 'attend_spell':
                    # Compute attention distributions
                    context_vec, aw_step = self.attend_sub(
                        enc_out, x_lens, dec_out, aw_step)

                    # Update decoder states
                    y = self.embed_sub(y)
                    dec_in = torch.cat([y, context_vec], dim=-1)
                    dec_out, dec_state = self.decoder_sub(dec_in, dec_state)

                elif self.decoding_order == 'spell_attend':
                    # Update decoder states
                    y = self.embed_sub(y)
                    dec_in = torch.cat([y, context_vec], dim=-1)
                    dec_out, dec_state = self.decoder_sub(dec_in, dec_state)

                    # Compute attention distributions
                    context_vec, aw_step = self.attend_sub(
                        enc_out, x_lens, dec_out, aw_step)

                elif self.decoding_order == 'conditional':
                    # Update decoder states of the first decoder
                    _dec_out, _dec_state = self.decoder_first_sub(
                        self.embed_sub(y), dec_state)

                    # Compute attention distributions
                    context_vec, aw_step = self.attend_sub(
                        enc_out, x_lens, _dec_out, aw_step)

                    # Update decoder states of the second decoder
                    dec_out, dec_state = self.decoder_second_sub(
                        context_vec, _dec_state)

                logits_step = self.fc_sub(F.tanh(
                    self.W_d_sub(dec_out) + self.W_c_sub(context_vec)))
            else:
                if self.decoding_order == 'attend_spell':
                    # Compute attention distributions
                    context_vec, aw_step = self.attend(
                        enc_out, x_lens, dec_out, aw_step)

                    # Update decoder states
                    y = self.embed(y)
                    dec_in = torch.cat([y, context_vec], dim=-1)
                    dec_out, dec_state = self.decoder(dec_in, dec_state)

                elif self.decoding_order == 'spell_attend':
                    # Update decoder states
                    y = self.embed(y)
                    dec_in = torch.cat([y, context_vec], dim=-1)
                    dec_out, dec_state = self.decoder(dec_in, dec_state)

                    # Compute attention distributions
                    context_vec, aw_step = self.attend(
                        enc_out, x_lens, dec_out, aw_step)

                elif self.decoding_order == 'conditional':
                    # Update decoder states of the first decoder
                    _dec_out, _dec_state = self.decoder_first(
                        self.embed(y), dec_state)

                    # Compute attention distributions
                    context_vec, aw_step = self.attend(
                        enc_out, x_lens, _dec_out, aw_step)

                    # Update decoder states of the second decoder
                    dec_out, dec_state = self.decoder_second(
                        context_vec, _dec_state)

                logits_step = self.fc(F.tanh(
                    self.W_d(dec_out) + self.W_c(context_vec)))

            logits.append(logits_step)
            aw.append(aw_step)

        # Concatenate in T_out-dimension
        logits = torch.cat(logits, dim=1)
        aw = torch.stack(aw, dim=1)
        # NOTE; aw in the training stage may be used for computing the
        # coverage, so do not convert to numpy yet.

        return logits, aw

    def _create_var(self, size, fill_value=0, dtype='float', volatile=False):
        """Initialize a variable with zero.
        Args:
            size (tuple):
            fill_value (int or float, optional):
            dtype (string): long or float
            volatile (bool, optional):
        Returns:
            var (torch.autograd.Variable, float):
        """
        var = Variable(torch.zeros(size).fill_(fill_value))
        if dtype == 'long':
            var = var.long()
        if volatile:
            var.volatile = True
        if self.use_cuda:
            var = var.cuda()
        return var

    # def _init_decoder_state(self, enc_out, is_sub_task=False):
    #     """Initialize decoder state.
    #     Args:
    #         enc_out (torch.autograd.Variable, float): A tensor of size
    #             `[B, T_in, encoder_num_units]`
    #         is_sub_task (bool, optional):
    #     Returns:
    #         dec_state (list or tuple of list):
    #     """
    #     zero_state = self._create_var((enc_out.size(0), self.decoder_num_units),
    #                                   volatile=not self.training)
    #
    #     if self.decoder_type == 'lstm':
    #         if is_sub_task:
    #             hx_list = [zero_state] * self.decoder_num_layers_sub
    #             cx_list = [zero_state] * self.decoder_num_layers_sub
    #         else:
    #             hx_list = [zero_state] * self.decoder_num_layers
    #             cx_list = [zero_state] * self.decoder_num_layers
    #     else:
    #         if is_sub_task:
    #             hx_list = [zero_state] * self.decoder_num_layers_sub
    #         else:
    #             hx_list = [zero_state] * self.decoder_num_layers
    #
    #     if self.init_dec_state != 'zero' and self.encoder_type == self.decoder_type:
    #         if self.init_dec_state == 'mean':
    #             # Initialize with mean of all encoder outputs
    #             h_0 = enc_out.mean(dim=1, keepdim=False)
    #         elif self.init_dec_state == 'final':
    #             # Initialize with the final encoder output
    #             h_0 = enc_out[:, -1, :]
    #         elif self.init_dec_state == 'first':
    #             # Initialize with the first encoder output
    #             h_0 = enc_out[:, 0, :]
    #
    #         # Path through the linear layer
    #         if is_sub_task:
    #             hx_list[0] = F.tanh(self.W_dec_init_sub(h_0))
    #         else:
    #             hx_list[0] = F.tanh(self.W_dec_init(h_0))
    #
    #     if self.decoder_type == 'lstm':
    #         dec_state = (hx_list, cx_list)
    #     else:
    #         dec_state = hx_list
    #
    #     return dec_state

    def _init_decoder_state(self, enc_out, is_sub_task=False):
        """Initialize decoder state (Nstep ver.).
        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T_in, encoder_num_units]`
            is_sub_task (bool, optional):
        Returns:
            dec_state (torch.autograd.Variable(float) or tuple): A tensor of size
                `[1, B, decoder_num_units]`
        """
        batch_size = enc_out.size(0)

        if self.init_dec_state == 'zero' or self.encoder_type != self.decoder_type:
            # Initialize with zero state
            h_0 = self._create_var((1, batch_size, self.decoder_num_units),
                                   volatile=not self.training)
        else:
            if self.init_dec_state == 'mean':
                # Initialize with mean of all encoder outputs
                h_0 = enc_out.mean(dim=1, keepdim=True)
            elif self.init_dec_state == 'final':
                # Initialize with the final encoder output
                h_0 = enc_out[:, -1, :].unsqueeze(1)
            elif self.init_dec_state == 'first':
                # Initialize with the first encoder output
                h_0 = enc_out[:, 0, :].unsqueeze(1)

            # Path through the linear layer
            if is_sub_task:
                h_0 = F.tanh(self.W_dec_init_sub(h_0))
            else:
                h_0 = F.tanh(self.W_dec_init(h_0))

            # Convert to time-major
            h_0 = h_0.transpose(0, 1).contiguous()

        if self.decoder_type == 'lstm':
            # Initialize memory cell with zero state
            c_0 = self._create_var((1, batch_size, self.decoder_num_units),
                                   volatile=not self.training)
            dec_state = (h_0, c_0)
        else:
            dec_state = h_0

        return dec_state

    def attention_weights(self, xs, x_lens, max_decode_len, is_sub_task=False):
        """Get attention weights for visualization.
        Args:
            xs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            x_lens (np.ndarray): A tensor of size `[B]`
            max_decode_len (int): the length of output sequences
                to stop prediction when EOS token have not been emitted
            is_sub_task (bool, optional):
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B, T_out]`
            aw (np.ndarray): A tensor of size `[B, T_out, T_in]`
            perm_idx (np.ndarray): A tensor of size `[B]`
        """
        if is_sub_task and self.sub_loss_weight == 0:
            raise ValueError

        # Change to evaluation mode
        self.eval()

        # Wrap by Variable
        xs = self.np2var(xs)
        x_lens = self.np2var(x_lens, dtype='int')

        # Encode acoustic features
        if hasattr(self, 'main_loss_weight'):
            if is_sub_task:
                _, _, enc_out, _, perm_idx = self._encode(
                    xs, x_lens, is_multi_task=True)
            else:
                enc_out, _, _, _, perm_idx = self._encode(
                    xs, x_lens, is_multi_task=True)
        else:
            enc_out, _, perm_idx = self._encode(xs, x_lens)

        # Permutate indices
        if perm_idx is not None:
            perm_idx = self.var2np(perm_idx)

        # NOTE: assume beam_width == 1
        best_hyps, aw = self._decode_infer_greedy(
            enc_out, x_lens, max_decode_len, is_sub_task=is_sub_task)

        # Permutate indices to the original order
        if perm_idx is not None:
            best_hyps = best_hyps[perm_idx]
            aw = aw[perm_idx]

        return best_hyps, aw, perm_idx

    def decode(self, xs, x_lens, beam_width, max_decode_len,
               length_penalty=0, coverage_penalty=0):
        """Decoding in the inference stage.
        Args:
            xs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            x_lens (np.ndarray): A tensor of size `[B]`
            beam_width (int): the size of beam
            max_decode_len (int): the length of output sequences
                to stop prediction when EOS token have not been emitted
            length_penalty (float, optional):
            coverage_penalty (float, optional):
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B]`
            perm_idx (np.ndarray): A tensor of size `[B]`
        """
        # Change to evaluation mode
        self.eval()

        # Wrap by Variable
        xs = self.np2var(xs)
        x_lens = self.np2var(x_lens, dtype='int')

        # Encode acoustic features
        enc_out, x_lens, perm_idx = self._encode(xs, x_lens)

        if beam_width == 1:
            best_hyps, _ = self._decode_infer_greedy(
                enc_out, x_lens, max_decode_len)
        else:
            best_hyps = self._decode_infer_beam(
                enc_out, x_lens, beam_width, max_decode_len,
                length_penalty, coverage_penalty)

        # Permutate indices to the original order
        if perm_idx is None:
            perm_idx = np.arange(0, len(xs), 1)
        else:
            perm_idx = self.var2np(perm_idx)

        return best_hyps, perm_idx

    def _decode_infer_greedy(self, enc_out, x_lens, max_decode_len,
                             is_sub_task=False):
        """Greedy decoding in the inference stage.
        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T_in, encoder_num_units]`
            x_lens (torch.autograd.Variable, int): A tensor of size `[B]`
            max_decode_len (int): the length of output sequences
                to stop prediction when EOS token have not been emitted
            is_sub_task (bool, optional):
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B, T_out]`
            aw (np.ndarray): A tensor of size `[B, T_out, T_in]`
        """
        batch_size = enc_out.size(0)

        # Initialize decoder state
        dec_state = self._init_decoder_state(enc_out, is_sub_task=is_sub_task)
        dec_out = self._create_var(
            (batch_size, 1, self.decoder_num_units), volatile=True)
        aw_step = self._create_var((
            batch_size, enc_out.size(1)), volatile=True)

        if self.decoding_order == 'spell_attend':
            context_vec = self._create_var(
                (batch_size, 1, self.encoder_num_units), volatile=True)

        # Start from <SOS>
        sos = self.sos_sub if is_sub_task else self.sos
        eos = self.eos_sub if is_sub_task else self.eos
        y = self._create_var((batch_size, 1), fill_value=sos, dtype='long')

        best_hyps = []
        aw = []
        for _ in range(max_decode_len):

            if is_sub_task:
                if self.decoding_order == 'attend_spell':
                    # Compute attention distributions
                    context_vec, aw_step = self.attend_sub(
                        enc_out, x_lens, dec_out, aw_step)

                    # Update decoder states
                    y = self.embed_sub(y)
                    dec_in = torch.cat([y, context_vec], dim=-1)
                    dec_out, dec_state = self.decoder_sub(dec_in, dec_state)

                elif self.decoding_order == 'spell_attend':
                    # Update decoder states
                    y = self.embed_sub(y)
                    dec_in = torch.cat([y, context_vec], dim=-1)
                    dec_out, dec_state = self.decoder_sub(dec_in, dec_state)

                    # Compute attention distributions
                    context_vec, aw_step = self.attend_sub(
                        enc_out, x_lens, dec_out, aw_step)

                elif self.decoding_order == 'conditional':
                    # Update decoder states of the first decoder
                    _dec_out, _dec_state = self.decoder_first_sub(
                        self.embed_sub(y), dec_state)

                    # Compute attention distributions
                    context_vec, aw_step = self.attend_sub(
                        enc_out, x_lens, _dec_out, aw_step)

                    # Update decoder states of the second decoder
                    dec_out, dec_state = self.decoder_second_sub(
                        context_vec, _dec_state)

                logits_step = self.fc_sub(F.tanh(
                    self.W_d_sub(dec_out) + self.W_c_sub(context_vec)))
            else:
                if self.decoding_order == 'attend_spell':
                    # Compute attention distributions
                    context_vec, aw_step = self.attend(
                        enc_out, x_lens, dec_out, aw_step)

                    # Update decoder states
                    y = self.embed(y)
                    dec_in = torch.cat([y, context_vec], dim=-1)
                    dec_out, dec_state = self.decoder(dec_in, dec_state)

                elif self.decoding_order == 'spell_attend':
                    # Update decoder states
                    y = self.embed(y)
                    dec_in = torch.cat([y, context_vec], dim=-1)
                    dec_out, dec_state = self.decoder(dec_in, dec_state)

                    # Compute attention distributions
                    context_vec, aw_step = self.attend(
                        enc_out, x_lens, dec_out, aw_step)

                elif self.decoding_order == 'conditional':
                    # Update decoder states of the first decoder
                    _dec_out, _dec_state = self.decoder_first(
                        self.embed(y), dec_state)

                    # Compute attention distributions
                    context_vec, aw_step = self.attend(
                        enc_out, x_lens, _dec_out, aw_step)

                    # Update decoder states of the second decoder
                    dec_out, dec_state = self.decoder_second(
                        context_vec, _dec_state)

                logits_step = self.fc(F.tanh(
                    self.W_d(dec_out) + self.W_c(context_vec)))

            # Pick up 1-best
            y = torch.max(logits_step.squeeze(1), dim=1)[1].unsqueeze(1)
            # logits_step: `[B, 1, num_classes]` -> `[B, num_classes]`
            best_hyps.append(y)
            aw.append(aw_step)

            # Break if <EOS> is outputed in all mini-batch
            if torch.sum(y.data == eos) == y.numel():
                break

        # Concatenate in T_out dimension
        best_hyps = torch.cat(best_hyps, dim=1)
        aw = torch.stack(aw, dim=1)

        # Convert to numpy
        best_hyps = self.var2np(best_hyps)
        aw = self.var2np(aw)

        return best_hyps, aw

    def _decode_infer_beam(self, enc_out, x_lens, beam_width, max_decode_len,
                           length_penalty, coverage_penalty, is_sub_task=False):
        """Beam search decoding in the inference stage.
        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T_in, encoder_num_units]`
            x_lens (torch.autograd.Variable, int): A tensor of size `[B]`
            beam_width (int): the size of beam
            max_decode_len (int, optional): the length of output sequences
                to stop prediction when EOS token have not been emitted
            length_penalty (float):
            coverage_penalty (float):
            is_sub_task (bool, optional):
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B]`
        """
        # Start from <SOS>
        sos = self.sos_sub if is_sub_task else self.sos
        eos = self.eos_sub if is_sub_task else self.eos

        best_hyps = []
        for b in range(enc_out.size(0)):
            frame_num = x_lens[b].data[0]

            # Initialization per utterance
            dec_state = self._init_decoder_state(
                enc_out[b:b + 1, :, :], is_sub_task=is_sub_task)
            dec_out = self._create_var(
                (1, 1, self.decoder_num_units), volatile=True)
            aw_step = self._create_var((1, frame_num), volatile=True)

            complete = []
            beam = [{'hyp': [sos],
                     'score': LOG_1,
                     'dec_state': dec_state,
                     'dec_out': dec_out,
                     'aw_step': aw_step}]
            for t in range(max_decode_len):
                new_beam = []
                for i_beam in range(len(beam)):
                    y = self._create_var(
                        (1, 1), fill_value=beam[i_beam]['hyp'][-1],
                        dtype='long')

                    if self.decoding_order == 'spell_attend':
                        # Compute context vector
                        context_vec = torch.sum(
                            enc_out[b: b + 1, :frame_num] *
                            beam[i_beam]['aw_step'].unsqueeze(2),
                            dim=1, keepdim=True)

                    if is_sub_task:
                        if self.decoding_order == 'attend_spell':
                            # Compute attention distributions
                            context_vec, aw_step = self.attend_sub(
                                enc_out[b:b + 1, :frame_num],
                                x_lens[b:b + 1],
                                dec_out, beam[i_beam]['aw_step'])

                            # Update decoder states
                            y = self.embed_sub(y)
                            dec_in = torch.cat([y, context_vec], dim=-1)
                            dec_out, dec_state = self.decoder_sub(
                                dec_in, beam[i_beam]['dec_state'])

                        elif self.decoding_order == 'spell_attend':
                            # Update decoder states
                            y = self.embed_sub(y)
                            dec_in = torch.cat([y, context_vec], dim=-1)
                            dec_out, dec_state = self.decoder_sub(
                                dec_in, beam[i_beam]['dec_state'])

                            # Compute attention distributions
                            context_vec, aw_step = self.attend_sub(
                                enc_out[b:b + 1, :frame_num],
                                x_lens[b:b + 1],
                                dec_out, beam[i_beam]['aw_step'])

                        elif self.decoding_order == 'conditional':
                            # Update decoder states of the first decoder
                            _dec_out, _dec_state = self.decoder_first_sub(
                                self.embed_sub(y), beam[i_beam]['dec_state'])

                            # Compute attention distributions
                            context_vec, aw_step = self.attend_sub(
                                enc_out[b:b + 1, :frame_num],
                                x_lens[b:b + 1],
                                _dec_out, beam[i_beam]['aw_step'])

                            # Update decoder states of the second decoder
                            dec_out, dec_state = self.decoder_second_sub(
                                context_vec, _dec_state)

                        logits_step = self.fc_sub(F.tanh(
                            self.W_d_sub(dec_out) + self.W_c_sub(context_vec)))
                    else:
                        if self.decoding_order == 'attend_spell':
                            # Compute attention distributions
                            context_vec, aw_step = self.attend(
                                enc_out[b:b + 1, :frame_num],
                                x_lens[b:b + 1],
                                dec_out, beam[i_beam]['aw_step'])

                            # Update decoder states
                            y = self.embed(y)
                            dec_in = torch.cat([y, context_vec], dim=-1)
                            dec_out, dec_state = self.decoder(
                                dec_in, beam[i_beam]['dec_state'])

                        elif self.decoding_order == 'spell_attend':
                            # Update decoder states
                            y = self.embed(y)
                            dec_in = torch.cat([y, context_vec], dim=-1)
                            dec_out, dec_state = self.decoder(
                                dec_in, beam[i_beam]['dec_state'])

                            # Compute attention distributions
                            context_vec, aw_step = self.attend(
                                enc_out[b:b + 1, :frame_num],
                                x_lens[b:b + 1],
                                dec_out, beam[i_beam]['aw_step'])

                        elif self.decoding_order == 'conditional':
                            # Update decoder states of the first decoder
                            _dec_out, _dec_state = self.decoder_first(
                                self.embed(y), beam[i_beam]['dec_state'])

                            # Compute attention distributions
                            context_vec, aw_step = self.attend(
                                enc_out[b:b + 1, :frame_num],
                                x_lens[b:b + 1],
                                _dec_out, beam[i_beam]['aw_step'])

                            # Update decoder states of the second decoder
                            dec_out, dec_state = self.decoder_second(
                                context_vec, _dec_state)

                        logits_step = self.fc(F.tanh(
                            self.W_d(dec_out) + self.W_c(context_vec)))

                    # Path through the softmax layer & convert to log-scale
                    log_probs = F.log_softmax(logits_step.squeeze(1), dim=-1)
                    # NOTE: `[1 (B), 1, num_classes]` -> `[1 (B), num_classes]`

                    # Pick up the top-k scores
                    log_probs_topk, indices_topk = torch.topk(
                        log_probs, k=beam_width, dim=-1,
                        largest=True, sorted=True)

                    for c, log_prob in zip(indices_topk.data[0], log_probs_topk.data[0]):
                        # numpy
                        new_score = np.logaddexp(
                            beam[i_beam]['score'], log_prob) + length_penalty

                        # torch
                        # new_score = _logsumexp([beam[i_beam]['score'], log_prob], dim=0)

                        new_beam.append(
                            {'hyp': beam[i_beam]['hyp'] + [c],
                             'score': new_score,
                             'dec_state': dec_state,
                             'dec_out': dec_out,
                             'aw_step': aw_step})

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
            best_hyps.append(np.array(complete[0]['hyp'][1:]))
            # NOTE: Exclude <SOS>

        return np.array(best_hyps)

    def decode_ctc(self, xs, x_lens, beam_width=1, is_sub_task=False):
        """Decoding by the CTC layer in the inference stage.
            This is only used for Joint CTC-Attention model.
        Args:
            xs (np.ndarray): A tensor of size `[B, T_in, input_size]`
            x_lens (np.ndarray): A tensor of size `[B]`
            beam_width (int, optional): the size of beam
            is_sub_task (bool, optional):
        Returns:
            best_hyps (np.ndarray): A tensor of size `[B]`
            perm_idx (np.ndarray): A tensor of size `[B]`
        """
        # assert self.ctc_loss_weight > 0
        # TODO: add is_sub_task

        # Change to evaluation mode
        self.eval()

        # Wrap by Variable
        xs = self.np2var(xs)
        x_lens = self.np2var(x_lens, dtype='int')

        # Encode acoustic features
        if is_sub_task:
            _, _, enc_out, x_lens, perm_idx = self._encode(
                xs, x_lens, is_multi_task=True)
        else:
            enc_out, x_lens, perm_idx = self._encode(xs, x_lens)

        # Path through the softmax layer
        batch_size, max_time = enc_out.size()[:2]
        enc_out = enc_out.view(batch_size * max_time, -1).contiguous()
        if is_sub_task:
            logits_ctc = self.fc_ctc_sub(enc_out)
        else:
            logits_ctc = self.fc_ctc(enc_out)
        logits_ctc = logits_ctc.view(batch_size, max_time, -1)

        if beam_width == 1:
            best_hyps = self._decode_ctc_greedy_np(
                self.var2np(logits_ctc), self.var2np(x_lens))
        else:
            best_hyps = self._decode_ctc_beam_np(
                self.var2np(F.log_softmax(logits_ctc, dim=-1)),
                self.var2np(x_lens), beam_width=beam_width)

        # NOTE: index 0 is reserved for blank in warpctc_pytorch
        best_hyps -= 1

        # Permutate indices to the original order
        if perm_idx is None:
            perm_idx = np.arange(0, len(xs), 1)
        else:
            perm_idx = self.var2np(perm_idx)

        return best_hyps, perm_idx


def _logsumexp(x, dim=None):
    """Pytorch implementation of logsumexp.
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
