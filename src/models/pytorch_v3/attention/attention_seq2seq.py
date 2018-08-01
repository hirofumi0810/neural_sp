#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention-based sequence-to-sequence model (pytorch)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import six

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from src.models.pytorch_v3.attention.decoder import Decoder
from src.models.pytorch_v3.base import ModelBase
from src.models.pytorch_v3.linear import LinearND


from src.models.pytorch_v3.encoders.load_encoder import load
from src.models.pytorch_v3.attention.attention_layer import AttentionMechanism
from src.models.pytorch_v3.attention.attention_layer import MultiheadAttentionMechanism
from src.models.pytorch_v3.criterion import cross_entropy_label_smoothing
from src.models.pytorch_v3.ctc.decoders.greedy_decoder import GreedyDecoder
from src.models.pytorch_v3.ctc.decoders.beam_search_decoder import BeamSearchDecoder
from src.models.pytorch_v3.linear import Embedding
from src.models.pytorch_v3.lm.rnnlm import RNNLM
from src.models.pytorch_v3.utils import np2var, var2np, pad_list
from src.utils.io.inputs.frame_stacking import stack_frame
from src.utils.io.inputs.splicing import do_splice


class AttentionSeq2seq(ModelBase):
    """Attention-based sequence-to-sequence model.

    Args:
        enc_in_type (string): speech or text
            speech means ASR or speech translation, and text means NMT or P2W and so on...
        enc_in_size (int): the dimension of input features (freq * channel)
        n_stack (int): the number of frames to stack
        n_skip (int): the number of frames to skip
        n_splice (int): frames to splice. Default is 1 frame.
        conv_in_channel (int): the number of channels of input features
        conv_n_channels (int): the number of channles of conv outputs.
        conv_width (int): the size of kernel. This must be the odd number.
        conv_channels (list): the number of channles in the convolution
        conv_kernel_sizes (list): the size of kernels in the convolution
        conv_strides (list): strides in the convolution
        conv_poolings (list): the size of poolings in the convolution
        conv_batch_norm (bool):
        enc_type (string): the type of the encoder. Set lstm or gru or rnn.
        enc_bidirectional (bool): if True, create a bidirectional encoder
        enc_n_units (int): the number of units in each layer of the encoder
        enc_n_projs (int): the number of nodes in the projection layer of the encoder
        enc_n_layers (int): the number of layers of the encoder
        enc_residual (bool):
        subsample_list (list): subsample in the corresponding layers (True)
            ex.) [False, True, True, False] means that subsample is conducted in the 2nd and 3rd layers.
        subsample_type (string): drop or concat

        att_type (string): the type of attention
        att_dim: (int) the dimension of the attention layer
        att_conv_n_channels (int):
        att_conv_width (int):
        att_n_heads (int): the number of heads in the multi-head attention
        sharpening_factor (float): a sharpening factor in the softmax layer
            for computing attention weights
        sigmoid_smoothing (bool): if True, replace softmax function in
            computing attention weights with sigmoid function for smoothing
        bridge_layer (bool): if True, add the bridge layer between the encoder and decoder

        dec_type (string): lstm or gru
        dec_n_units (int): the number of units in each layer of the decoder
        dec_n_layers (int): the number of layers of the decoder
        dec_residual (bool):
        emb_dim (int): the dimension of the embedding in target spaces.
            0 means that decoder inputs are represented by one-hot vectors.
        bottle_dim (int): the dimension of the pre-softmax layer
        generate_feat (string): s or sc
        n_classes (int): the number of nodes in softmax layer (excluding <SOS> and <EOS> classes)
                param_init_dist (string): uniform or normal or orthogonal or constant distribution
        logits_temp (float): a parameter for smoothing the softmax layer in outputing probabilities
        param_init (float): Range of uniform distribution to initialize weight parameters
        param_init_dist (string):
        recurrent_weight_orthogonal (bool): if True, recurrent weights are orthogonalized
        init_forget_gate_bias_with_one (bool): if True, initialize the forget gate bias with 1
        dropout_in (float): the probability to drop nodes in input-hidden connection
        dropout_enc (float): the probability to drop nodes in hidden-hidden connection of the encoder
        dropout_dec (float): the probability to drop nodes of the decoder
        dropout_emb (float): the probability to drop nodes of the embedding layer
        ss_prob (float):
        lsm_prob (float):
        lsm_type (string): uniform or unigram
        weight_noise_std (flaot):
        coverage_weight (float): the weight parameter for coverage computation
        ctc_loss_weight (float): A weight parameter for auxiliary CTC loss
        backward_loss_weight (int): A weight parameter for the loss of the backward decdoer,
            where the model predicts each token in the reverse order
        lm_fusion (string):
            False:
            cold_fusion_prob:
            cold_fusion_hidden:
            logits_fusion:
        lm_config (dict): configuration of the pre-trained RNNLM
        lm_loss_weight (float): the weight for XE loss of RNNLM
        lm_init (bool):
        finetune_gate (bool):
        n_classes_in (int):

    """

    def __init__(self,
                 enc_in_type,
                 enc_in_size,
                 n_stack,
                 n_skip,
                 n_splice,
                 conv_in_channel,
                 conv_channels,
                 conv_kernel_sizes,
                 conv_strides,
                 conv_poolings,
                 conv_batch_norm,
                 enc_type,
                 enc_bidirectional,
                 enc_n_units,
                 enc_n_projs,
                 enc_n_layers,
                 enc_residual,
                 subsample_list,
                 subsample_type,
                 att_type,
                 att_dim,
                 att_conv_n_channels,
                 att_conv_width,
                 att_n_heads,
                 sharpening_factor,
                 sigmoid_smoothing,
                 bridge_layer,
                 dec_type,
                 dec_n_units,
                 dec_n_layers,
                 dec_residual,
                 emb_dim,
                 bottle_dim,
                 generate_feat,
                 n_classes,
                 logits_temp,
                 param_init,
                 param_init_dist,
                 recurrent_weight_orthogonal,
                 init_forget_gate_bias_with_one,
                 dropout_in,
                 dropout_enc,
                 dropout_dec,
                 dropout_emb,
                 ss_prob,
                 lsm_prob,
                 lsm_type,
                 weight_noise_std,
                 coverage_weight,
                 ctc_loss_weight,
                 backward_loss_weight,
                 lm_fusion,
                 lm_config,
                 lm_loss_weight,
                 lm_init,
                 finetune_gate,
                 n_classes_in):

        super(ModelBase, self).__init__()
        self.model_type = 'attention'

        # Setting for the encoder
        self.enc_in_type = enc_in_type
        assert enc_in_type in ['speech', 'text']
        self.n_stack = n_stack
        self.n_skip = n_skip
        self.n_splice = n_splice
        self.enc_type = enc_type
        self.enc_n_units = enc_n_units
        if enc_bidirectional:
            self.enc_n_units *= 2
        if enc_n_projs:
            self.enc_n_units = enc_n_projs

        # Setting for the attention
        self.sharpening_factor = sharpening_factor
        self.sigmoid_smoothing = sigmoid_smoothing
        self.coverage_weight = coverage_weight
        self.att_n_heads_0 = att_n_heads
        self.share_att = False

        # Setting for the decoder
        self.n_classes = n_classes + 1  # Add <EOS> class
        self.sos_0 = n_classes
        self.eos_0 = n_classes
        self.pad_index = -1024
        # NOTE: <SOS> and <EOS> have the same index
        assert 0 <= backward_loss_weight <= 1
        self.fwd_weight_0 = 1 - backward_loss_weight
        self.bwd_weight_0 = backward_loss_weight

        self.logits_temp = logits_temp

        # Setting for regularization
        self.weight_noise_injection = False
        self.weight_noise_std = float(weight_noise_std)
        self.ss_prob = ss_prob
        self.lsm_prob = lsm_prob
        self.ls_type = lsm_type

        # Setting for MTL
        self.ctc_loss_weight = ctc_loss_weight
        if ctc_loss_weight > 0:
            from src.models.pytorch_v3.ctc.ctc import my_warpctc
            self.warp_ctc = my_warpctc

        # Setting for the RNNLM fusion
        if lm_fusion and lm_config is not None:
            self.lm_fusion_type_0 = lm_fusion
        else:
            self.lm_fusion_type_0 = False
        self.rnnlm_0_fwd = None
        self.rnnlm_0_bwd = None
        self.rnnlm_weight_0 = lm_loss_weight
        if lm_init and lm_config is not None:
            self.rnnlm_init_0 = lm_init
        else:
            self.rnnlm_init_0 = False

        # Setting for text input
        self.n_classes_in = n_classes_in + 1

        # RNNLM fusion
        if self.lm_fusion_type_0 or self.rnnlm_init_0:
            self.rnnlm_0_fwd = RNNLM(
                emb_dim=lm_config['emb_dim'],
                rnn_type=lm_config['rnn_type'],
                bidirectional=lm_config['bidirectional'],
                n_units=lm_config['n_units'],
                n_layers=lm_config['n_layers'],
                dropout_emb=lm_config['dropout_emb'],
                dropout_hidden=lm_config['dropout_hidden'],
                dropout_out=lm_config['dropout_out'],
                n_classes=lm_config['n_classes'],
                param_init_dist=lm_config['param_init_dist'],
                param_init=lm_config['param_init'],
                recurrent_weight_orthogonal=lm_config['recurrent_weight_orthogonal'],
                init_forget_gate_bias_with_one=lm_config['init_forget_gate_bias_with_one'],
                lsm_prob=lm_config['lsm_prob'],
                tie_weights=lm_config['tie_weights'],
                residual_connection=lm_config['residual_connection'],
                backward=lm_config['backward'])

            # TODO(hirofumi): cold fusion for backward RNNLM

        # Encoder
        if enc_type in ['lstm', 'gru', 'rnn']:
            self.enc = load(enc_type=enc_type)(
                input_size=enc_in_size,
                rnn_type=enc_type,
                bidirectional=enc_bidirectional,
                n_units=enc_n_units,
                n_proj=enc_n_projs,
                n_layers=enc_n_layers,
                dropout_in=dropout_in,
                dropout_hidden=dropout_enc,
                subsample_list=subsample_list,
                subsample_type=subsample_type,
                batch_first=True,
                merge_bidirectional=False,
                pack_sequence=True,
                n_stack=n_stack,
                n_splice=n_splice,
                conv_in_channel=conv_in_channel,
                conv_channels=conv_channels,
                conv_kernel_sizes=conv_kernel_sizes,
                conv_strides=conv_strides,
                conv_poolings=conv_poolings,
                batch_norm=conv_batch_norm,
                residual=enc_residual,
                nin=0)
        elif enc_type == 'cnn':
            assert n_stack == 1 and n_splice == 1
            self.enc = load(enc_type='cnn')(
                input_size=enc_in_size,
                conv_in_channel=conv_in_channel,
                conv_channels=conv_channels,
                conv_kernel_sizes=conv_kernel_sizes,
                conv_strides=conv_strides,
                conv_poolings=conv_poolings,
                dropout_in=dropout_in,
                dropout_hidden=dropout_enc,
                batch_norm=conv_batch_norm)
        else:
            raise NotImplementedError

        # Bridge layer between the encoder and decoder
        if enc_type == 'cnn':
            self.bridge_0 = LinearND(self.enc.output_size, dec_n_units)
            self.enc_n_units = dec_n_units
            self.is_bridge = True
        elif bridge_layer:
            self.bridge_0 = LinearND(self.enc_n_units, dec_n_units)
            self.enc_n_units = dec_n_units
            self.is_bridge = True
        else:
            self.is_bridge = False

        directions = []
        if self.fwd_weight_0 > 0:
            directions.append('fwd')
        if self.bwd_weight_0 > 0:
            directions.append('bwd')
        for dir in directions:
            # Attention layer
            if att_n_heads > 1:
                att = MultiheadAttentionMechanism(
                    enc_n_units=self.enc_n_units,
                    dec_n_units=dec_n_units,
                    att_type=att_type,
                    att_dim=att_dim,
                    sharpening_factor=sharpening_factor,
                    sigmoid_smoothing=sigmoid_smoothing,
                    out_channels=att_conv_n_channels,
                    kernel_size=att_conv_width,
                    n_heads=att_n_heads)
            else:
                att = AttentionMechanism(
                    enc_n_units=self.enc_n_units,
                    dec_n_units=dec_n_units,
                    att_type=att_type,
                    att_dim=att_dim,
                    sharpening_factor=sharpening_factor,
                    sigmoid_smoothing=sigmoid_smoothing,
                    out_channels=att_conv_n_channels,
                    kernel_size=att_conv_width)

            # Decoder
            setattr(self, 'dec_0_' + dir, Decoder(
                score_fn=att,
                sos=self.sos_0,
                eos=self.eos_0,
                enc_n_units=self.enc_n_units,
                rnn_type=dec_type,
                n_units=dec_n_units,
                n_layers=dec_n_layers,
                emb_dim=emb_dim,
                bottle_dim=bottle_dim,
                generate_feat=generate_feat,
                n_classes=self.n_classes,
                dropout_dec=dropout_dec,
                dropout_emb=dropout_emb,
                residual=dec_residual,
                cov_weight=coverage_weight,
                lm_fusion=lm_fusion,
            ))

        if enc_in_type == 'text':
            self.embed_in = Embedding(num_classes=n_classes_in,
                                      embedding_dim=enc_in_type,
                                      dropout=dropout_emb,
                                      ignore_index=n_classes_in - 1)

        # CTC
        if ctc_loss_weight > 0:
            if self.is_bridge:
                self.fc_ctc_0 = LinearND(dec_n_units, n_classes + 1)
            else:
                self.fc_ctc_0 = LinearND(self.enc_n_units, n_classes + 1)

            # Set CTC decoders
            self._decode_ctc_greedy = GreedyDecoder(blank_index=0)
            self._decode_ctc_beam = BeamSearchDecoder(blank_index=0)
            # TODO(hirofumi): set space index

        # Fix all parameters except for gate
        if self.lm_fusion_type_0 and finetune_gate:
            assert self.lm_fusion_type_0 in ['cold_fusion_prob', 'cold_fusion_hidden']
            fix_params = ['fc_dec', 'fc_cv', 'fc_bottle',
                          'fc_cf_lm_logits', 'fc_cf_gate', 'fc_cf_gated_lm']

            for n, p in self.named_parameters():
                if n.split('.')[0] not in fix_params:
                    p.requires_grad = False

        # Initialize weight matrices
        self.init_weights(param_init, distribution=param_init_dist,
                          ignore_keys=['bias'])

        # Initialize all biases with 0
        self.init_weights(0, distribution='constant', keys=['bias'])

        # Recurrent weights are orthogonalized
        if recurrent_weight_orthogonal:
            # encoder
            if enc_type != 'cnn':
                self.init_weights(param_init, distribution='orthogonal',
                                  keys=[enc_type, 'weight'], ignore_keys=['bias'])
            # decoder
            self.init_weights(param_init, distribution='orthogonal',
                              keys=[dec_type, 'weight'], ignore_keys=['bias'])

        # Initialize bias in forget gate with 1
        if init_forget_gate_bias_with_one:
            self.init_forget_gate_bias_with_one()

        # Initialize bias in gating with -1
        if self.lm_fusion_type_0 in ['cold_fusion_prob', 'cold_fusion_hidden']:
            self.init_weights(-1, distribution='constant', keys=['fc_cf_gate.fc.bias'])

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
        if is_eval and self.enc_type != 'cnn' or self.enc_in_type == 'text':
            perm_idx = sorted(list(six.moves.range(0, len(xs), 1)),
                              key=lambda i: len(xs[i]), reverse=True)
            xs = [xs[i] for i in perm_idx]
            ys = [ys[i] for i in perm_idx]
            # NOTE: must be descending order for pack_padded_sequence
            # NOTE: assumed that xs is already sorted in the training stage

        if self.fwd_weight_0 > 0 or self.ctc_loss_weight > 0:
            ys_fwd = [np2var(np.fromiter(y, dtype=np.int64), self.device_id).long()
                      for y in ys]
        if self.bwd_weight_0 > 0:
            ys_bwd = [np2var(np.fromiter(y[::-1], dtype=np.int64), self.device_id).long()
                      for y in ys]
            # NOTE: reverse the order

        # Encode input features
        xs, x_lens = self._encode(xs)

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

        return loss, acc

    def compute_xe_loss(self, enc_out, ys, x_lens):
        """Compute XE loss.

        Args:
            enc_out (torch.autograd.Variable, float): A tensor of size
                `[B, T, enc_n_units]`
            ys (list): A list of length `[B]`, which contains Variables of size `[L]`
            x_lens (list): A list of length `[B]`
            task (int): the index of a task
            dir (str): fwd or bwd
        Returns:
            loss (torch.autograd.Variable, float): A tensor of size `[1]`
            acc (float): Token-level accuracy in teacher-forcing

        """
        sos = Variable(enc_out.data.new(1,).fill_(getattr(self, 'sos_' + str(task))).long())
        eos = Variable(enc_out.data.new(1,).fill_(getattr(self, 'eos_' + str(task))).long())

        # Append <SOS> and <EOS>
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # Convert list to Variable
        ys_in = pad_list(ys_in, getattr(self, 'eos_' + str(task)))
        ys_out = pad_list(ys_out, self.pad_index)

        # NOTE: change RNNLM to the evaluation mode in case of cold fusion
        if getattr(self, 'rnnlm_fusion_type_' + str(task)) and getattr(self, 'rnnlm_' + str(task) + '_' + dir) == 0:
            getattr(self, 'rnnlm_' + str(task) + '_' + dir).eval()

        # Teacher-forcing
        logits, aw, logits_rnnlm = self._decode_train(enc_out, x_lens, ys_i)

        # Output smoothing
        if self.logits_temp != 1:
            logits /= self.logits_temp

        # Compute XE sequence loss
        if self.lsm_prob > 0:
            # Label smoothing
            y_lens = [y.size(0) + 1 for y in ys]   # Add <EOS>
            loss = cross_entropy_label_smoothing(
                logits, ys=ys_out, y_lens=y_lens,
                lsm_prob=self.lsm_prob,
                lsm_type=self.ls_type, size_average=True)
        else:
            loss = F.cross_entropy(
                input=logits.view((-1, logits.size(2))),
                target=ys_out.view(-1),  # long
                ignore_index=self.pad_index, size_average=False) / len(enc_out)

        # Compute XE loss for RNNLM
        if getattr(self, 'rnnlm_fusion_type_' + str(task)) and getattr(self, 'rnnlm_weight_' + str(task)) > 0:
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
                `[B, T, enc_n_units]`
            ys (torch.autograd.Variable, long): A tensor of size `[B, L]`,
                which does not include <SOS> nor <EOS>
            x_lens (list): A list of length `[B]`
            task (int): the index of a task
        Returns:
            loss (torch.autograd.Variable, float): A tensor of size `[1]`

        """
        # Wrap by Variable
        x_lens = np2var(np.fromiter(x_lens, dtype=np.int32), -1).int()
        y_lens = np2var(np.fromiter([y.size(0) for y in ys], dtype=np.int32), -1).int()
        # NOTE: do not copy to GPUs

        # Concatenate all elements in ys for warpctc_pytorch
        ys_ctc = torch.cat(ys, dim=0).cpu().int() + 1
        # NOTE: index 0 is reserved for blank in warpctc_pytorch

        # Path through the fully-connected layer
        logits = getattr(self, 'fc_ctc_' + str(task))(enc_out)

        # Compute CTC loss
        loss = self.warp_ctc(logits.transpose(0, 1),  # time-major
                             ys_ctc,  # int
                             x_lens,  # int
                             y_lens,  # int
                             size_average=False) / len(enc_out)

        if self.device_id >= 0:
            loss = loss.cuda(self.device_id)

        return loss

    def _encode(self, xs, is_multi_task=False):
        """Encode acoustic features.

        Args:
            xs (list): A list of length `[B]`, which contains Variables of size `[T, input_size]`
            is_multi_task (bool): Set True in the case of MTL
        Returns:
            xs (torch.autograd.Variable, float): A tensor of size
                `[B, T, enc_n_units]`
            x_lens (list): A tensor of size `[B]`
            OPTION:
                xs_sub (torch.autograd.Variable, float): A tensor of size
                    `[B, T, enc_n_units]`
                x_lens_sub (list): A tensor of size `[B]`

        """
        if self.enc_in_type == 'speech':
            # Frame stacking
            if self.n_stack > 1:
                xs = [stack_frame(x, self.n_stack, self.n_skip)
                      for x in xs]

            # Splicing
            if self.n_splice > 1:
                xs = [do_splice(x, self.n_splice, self.n_stack) for x in xs]

            # Wrap by Variable
            x_lens = [len(x) for x in xs]
            xs = [np2var(x, self.device_id).float() for x in xs]
            xs = pad_list(xs)

        elif self.enc_in_type == 'text':
            # Wrap by Variable
            x_lens = [len(x) for x in xs]
            xs = [np2var(np.fromiter(x, dtype=np.int64), self.device_id).long()
                  for x in xs]
            xs = pad_list(xs, self.n_classes_in - 1)
            xs = self.embed_in(xs)

        if is_multi_task:
            if self.enc_type == 'cnn':
                xs, x_lens = self.enc(xs, x_lens)
                xs_sub = xs.clone()
                x_lens_sub = copy.deepcopy(x_lens)
            else:
                xs, x_lens, xs_sub, x_lens_sub = self.enc(
                    xs, x_lens, volatile=not self.training)
        else:
            if self.enc_type == 'cnn':
                xs, x_lens = self.enc(xs, x_lens)
            else:
                xs, x_lens = self.enc(
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

    def decode(self, xs, beam_width,
               min_len_ratio=0, length_penalty=0, coverage_penalty=0, coverage_threshold=0,
               lm_loss_weight=0, task_index=0, resolving_unk=False,
               exclude_eos=True):
        """Decoding in the inference stage.

        Args:
            xs (list): A list of length `[B]`, which contains arrays of size `[T, input_size]`
            beam_width (int): the size of beam
            min_len_ratio (float):
            min_len_ratio (float):
            length_penalty (float): length penalty
            coverage_penalty (float): coverage penalty
            coverage_threshold (float): threshold for coverage penalty
            lm_loss_weight (float): the weight of RNNLM score
            task_index (int): not used (to make compatible)
            resolving_unk (bool): not used (to make compatible)
            exclude_eos (bool): if True, exclude <EOS> from best_hyps
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw (list): A list of length `[B]`, which contains arrays of size `[L, T]`
            perm_idx (list): A list of length `[B]`

        """
        self.eval()

        # Sort by lenghts in the descending order
        if self.enc_type != 'cnn' or self.enc_in_type == 'text':
            perm_idx = sorted(list(six.moves.range(0, len(xs), 1)),
                              key=lambda i: len(xs[i]), reverse=True)
            xs = [xs[i] for i in perm_idx]
            # NOTE: must be descending order for pack_padded_sequence
        else:
            perm_idx = list(six.moves.range(0, len(xs), 1))

        # Encode input features
        enc_out, x_lens = self._encode(xs)

        dir = 'fwd' if self.fwd_weight_0 >= self.bwd_weight_0 else 'bwd'

        if beam_width == 1:
            best_hyps, aw = self._decode_infer_greedy(
                enc_out, x_lens, max_decode_len, 0, dir, exclude_eos)
        else:
            best_hyps, aw = self.beam_search(
                enc_out, x_lens, beam_width, max_decode_len, min_decode_len,
                min_len_ratio, length_penalty, coverage_penalty, coverage_threshold,
                lm_loss_weight, 0, dir, exclude_eos)

        return best_hyps, aw, perm_idx

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
        if self.enc_type != 'cnn' or self.enc_in_type == 'text':
            perm_idx = sorted(list(six.moves.range(0, len(xs), 1)),
                              key=lambda i: len(xs[i]), reverse=True)
            xs = [xs[i] for i in perm_idx]
            # NOTE: must be descending order for pack_padded_sequence
        else:
            perm_idx = list(six.moves.range(0, len(xs), 1))

        # Encode input features
        if task_index == 0:
            enc_out, x_lens = self._encode(xs)
        elif task_index == 1:
            _, _, enc_out, x_lens = self._encode(xs, is_multi_task=True)
        else:
            raise NotImplementedError

        # Path through the softmax layer
        batch_size, max_time = enc_out.size()[:2]
        enc_out = enc_out.view(batch_size * max_time, -1).contiguous()
        logits_ctc = getattr(self, 'fc_ctc_' + str(task_index))(enc_out)
        logits_ctc = logits_ctc.view(batch_size, max_time, -1)

        if beam_width == 1:
            best_hyps = self._decode_ctc_greedy(var2np(logits_ctc), x_lens)
        else:
            best_hyps = self._decode_ctc_beam(
                var2np(F.log_softmax(logits_ctc, dim=-1)),
                x_lens, beam_width=beam_width)

        # NOTE: index 0 is reserved for blank in warpctc_pytorch
        best_hyps -= 1

        return best_hyps, perm_idx
