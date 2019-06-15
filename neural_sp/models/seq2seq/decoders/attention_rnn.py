#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""RNN decoder (including CTC loss calculation)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.criterion import cross_entropy_lsm
from neural_sp.models.criterion import focal_loss
from neural_sp.models.criterion import distillation
from neural_sp.models.lm.rnnlm import RNNLM
from neural_sp.models.modules.embedding import Embedding
from neural_sp.models.modules.linear import LinearND
from neural_sp.models.modules.multihead_attention import MultiheadAttentionMechanism
from neural_sp.models.modules.singlehead_attention import AttentionMechanism
from neural_sp.models.modules.zoneout import zoneout_wrapper
from neural_sp.models.seq2seq.decoders.ctc import CTC
from neural_sp.models.seq2seq.decoders.ctc import CTCPrefixScore
from neural_sp.models.seq2seq.decoders.decoder_base import DecoderBase
from neural_sp.models.torch_utils import compute_accuracy
from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list
from neural_sp.models.torch_utils import make_pad_mask
from neural_sp.models.torch_utils import tensor2np

random.seed(1)


class RNNDecoder(DecoderBase):
    """RNN decoder.

    Args:
        eos (int): index for <eos> (shared with <sos>)
        unk (int): index for <unk>
        pad (int): index for <pad>
        blank (int): index for <blank>
        enc_n_units (int):
        attn_type (str):
        rnn_type (str): lstm or gru
        n_units (int): number of units in each RNN layer
        n_projs (int): number of units in each projection layer
        n_layers (int): number of RNN layers
        loop_type (str): normal or lmdecoder
        residual (bool):
        bottleneck_dim (int): dimension of the bottleneck layer before the softmax layer for label generation
        emb_dim (int): dimension of the embedding in target spaces.
        vocab (int): number of nodes in softmax layer
        tie_embedding (bool):
        attn_dim (int):
        attn_sharpening_factor (float):
        attn_sigmoid_smoothing (bool):
        attn_conv_out_channels (int):
        attn_conv_kernel_size (int):
        attn_n_heads (int): number of attention heads
        dropout (float): dropout probability for the RNN layer
        dropout_emb (float): dropout probability for the embedding layer
        dropout_att (float): dropout probability for attention distributions
        zoneout (float): zoneout probability for the RNN layer
        lsm_prob (float): label smoothing probability
        ss_prob (float): scheduled sampling probability
        ss_type (str): constant or saturation
        fl_weight (float):
        fl_gamma (float):
        ctc_weight (float):
        ctc_lsm_prob (float): label smoothing probability for CTC
        ctc_fc_list (list):
        input_feeding (bool):
        backward (bool): decode in the backward order
        lm_fusion (RNNLM):
        lm_fusion_type (str): type of LM fusion
        contextualize (str): hierarchical_encoder or ...
        lm_init (RNNLM):
        lmobj_weight (float):
        share_lm_softmax (bool):
        global_weight (float):
        mtl_per_batch (bool):
        adaptive_softmax (bool):
        param_init (float):
        replace_sos (bool):

    """

    def __init__(self,
                 eos,
                 unk,
                 pad,
                 blank,
                 enc_n_units,
                 attn_type,
                 rnn_type,
                 n_units,
                 n_projs,
                 n_layers,
                 residual,
                 loop_type,
                 bottleneck_dim,
                 emb_dim,
                 vocab,
                 tie_embedding=False,
                 attn_dim=0,
                 attn_sharpening_factor=0.0,
                 attn_sigmoid_smoothing=False,
                 attn_conv_out_channels=0,
                 attn_conv_kernel_size=0,
                 attn_n_heads=0,
                 dropout=0.0,
                 dropout_emb=0.0,
                 dropout_att=0.0,
                 zoneout=0,
                 lsm_prob=0.0,
                 ss_prob=0.0,
                 ss_type='constant',
                 fl_weight=0.0,
                 fl_gamma=2.0,
                 ctc_weight=0.0,
                 ctc_lsm_prob=0.0,
                 ctc_fc_list=[],
                 input_feeding=False,
                 backward=False,
                 lm_fusion=None,
                 lm_fusion_type='cold',
                 contextualize='',
                 lm_init=None,
                 lmobj_weight=0.0,
                 share_lm_softmax=False,
                 global_weight=1.0,
                 mtl_per_batch=False,
                 adaptive_softmax=False,
                 param_init=0.1,
                 replace_sos=False):

        super(RNNDecoder, self).__init__()
        logger = logging.getLogger('training')

        self.eos = eos
        self.unk = unk
        self.pad = pad
        self.blank = blank
        self.vocab = vocab
        self.rnn_type = rnn_type
        assert rnn_type in ['lstm', 'gru']
        self.enc_n_units = enc_n_units
        self.dec_n_units = n_units
        self.n_projs = n_projs
        self.n_layers = n_layers
        self.loop_type = loop_type
        if loop_type == 'lmdecoder':
            assert n_layers >= 2
        self.residual = residual
        self.ss_prob = ss_prob
        self.ss_type = ss_type
        if ss_type == 'constant':
            self._ss_prob = ss_prob
        elif ss_type == 'saturation':
            self._ss_prob = 0  # start from 0
        self.lsm_prob = lsm_prob
        self.fl_weight = fl_weight
        self.fl_gamma = fl_gamma
        self.ctc_weight = ctc_weight
        self.input_feeding = input_feeding
        if input_feeding:
            assert loop_type == 'normal'
        self.bwd = backward
        self.lm_fusion_type = lm_fusion_type
        self.contextualize = contextualize
        self.lmobj_weight = lmobj_weight
        if lmobj_weight > 0:
            assert loop_type == 'lmdecoder'
            assert not input_feeding
        self.share_lm_softmax = share_lm_softmax
        self.global_weight = global_weight
        self.mtl_per_batch = mtl_per_batch
        self.replace_sos = replace_sos

        # for cache
        self.fifo_cache_ids = []
        self.fifo_cache_sp_key = None
        self.fifo_cache_lm_key = None
        self.dict_cache_sp = {}
        self.static_cache = {}
        self.static_cache_utt_ids = []
        self.dict_cache_lm = {}
        self.prev_spk = ''
        self.total_step = 0
        self.dstates_final = None
        self.lmstate_final = None

        if ctc_weight > 0:
            self.ctc = CTC(eos=eos,
                           blank=blank,
                           enc_n_units=enc_n_units,
                           vocab=vocab,
                           dropout=dropout,
                           lsm_prob=ctc_lsm_prob,
                           fc_list=ctc_fc_list,
                           param_init=param_init)

        if ctc_weight < global_weight:
            # Attention layer
            if attn_n_heads > 1:
                self.score = MultiheadAttentionMechanism(
                    key_dim=self.enc_n_units,
                    query_dim=n_units if n_projs == 0 else n_projs,
                    attn_type=attn_type,
                    attn_dim=attn_dim,
                    n_heads=attn_n_heads,
                    dropout=dropout_att)
            else:
                self.score = AttentionMechanism(
                    key_dim=self.enc_n_units,
                    query_dim=n_units if n_projs == 0 else n_projs,
                    attn_type=attn_type,
                    attn_dim=attn_dim,
                    sharpening_factor=attn_sharpening_factor,
                    sigmoid_smoothing=attn_sigmoid_smoothing,
                    conv_out_channels=attn_conv_out_channels,
                    conv_kernel_size=attn_conv_kernel_size,
                    dropout=dropout_att)

            # for MTL with LM objective
            if lmobj_weight > 0 and loop_type == 'lmdecoder':
                if share_lm_softmax:
                    self.output_lmobj = self.output  # share paramters
                else:
                    self.output_lmobj = LinearND(n_units, vocab)

            # Decoder
            self.rnn = nn.ModuleList()
            if self.n_projs > 0:
                self.proj = nn.ModuleList([LinearND(n_units, n_projs) for _ in range(n_layers)])
            self.dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(n_layers)])
            cell = nn.LSTMCell if rnn_type == 'lstm' else nn.GRUCell
            # 1st layer
            if loop_type == 'normal':
                dec_idim = n_units if input_feeding else enc_n_units
                dec_idim += emb_dim
            elif loop_type == 'lmdecoder':
                dec_idim = emb_dim
            self.rnn += [zoneout_wrapper(cell(dec_idim, n_units), zoneout, zoneout)]
            dec_idim = n_units
            if self.n_projs > 0:
                dec_idim = n_projs
            # 2nd layer
            if n_layers >= 2:
                if loop_type == 'lmdecoder':
                    dec_idim += enc_n_units
                self.rnn += [zoneout_wrapper(cell(dec_idim, n_units))]
                if self.n_projs > 0:
                    dec_idim = n_projs
            # 3rd~ layers
            if n_layers >= 3:
                for l in range(n_layers - 2):
                    self.rnn += [zoneout_wrapper(cell(dec_idim, n_units), zoneout, zoneout)]
                    if self.n_projs > 0:
                        dec_idim = n_projs

            # LM fusion
            if lm_fusion is not None:
                self.linear_dec_feat = LinearND(dec_idim + enc_n_units, n_units)
                if lm_fusion_type in ['cold', 'deep']:
                    self.linear_lm_feat = LinearND(lm_fusion.n_units, n_units)
                    self.linear_lm_gate = LinearND(n_units * 2, n_units)
                elif lm_fusion_type == 'cold_prob':
                    self.linear_lm_feat = LinearND(lm_fusion.vocab, n_units)
                    self.linear_lm_gate = LinearND(n_units * 2, n_units)
                else:
                    raise ValueError(lm_fusion_type)
                self.output_bn = LinearND(n_units * 2, bottleneck_dim)

                # fix LM parameters
                for p in lm_fusion.parameters():
                    p.requires_grad = False
            elif contextualize:
                raise NotImplementedError
            else:
                self.output_bn = LinearND(dec_idim + enc_n_units, bottleneck_dim)

            self.embed = Embedding(vocab, emb_dim,
                                   dropout=dropout_emb,
                                   ignore_index=pad)

            if adaptive_softmax:
                assert self.vocab >= 25000
                self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
                    bottleneck_dim, vocab,
                    cutoffs=[self.vocab // 10, 3 * self.vocab // 10],
                    # cutoffs=[self.vocab // 25, 3 * self.vocab // 5],
                    div_value=4.0)
                self.output = None
            else:
                self.adaptive_softmax = None
                self.output = LinearND(bottleneck_dim, vocab)
                # NOTE: include bias even when tying weights

                # Optionally tie weights as in:
                # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
                # https://arxiv.org/abs/1608.05859
                # and
                # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
                # https://arxiv.org/abs/1611.01462
                if tie_embedding:
                    if emb_dim != bottleneck_dim:
                        raise ValueError('When using the tied flag, n_units must be equal to emb_dim.')
                    self.output.fc.weight = self.embed.embed.weight

        # Initialize parameters
        self.reset_parameters(param_init)

        # resister the external LM
        self.lm = lm_fusion

        # decoder initialization with pre-trained LM
        if lm_init is not None:
            assert lm_init.vocab == vocab
            assert lm_init.n_units == n_units
            assert lm_init.emb_dim == emb_dim
            logger.info('===== Initialize the decoder with pre-trained RNNLM')
            assert lm_init.n_projs == 0  # TODO(hirofumi): fix later

            if loop_type == 'lmdecoder':
                assert n_layers < lm_init.n_layers
                assert lm_init.n_layers == 1
                assert lm_init.n_units_null_context == 0

            # RNN
            if lm_init.fast_impl:
                for n, p in lm_init.rnn.named_parameters():
                    l = int(n.split('_')[2].replace('l', ''))
                    n = '_'.join(n.split('_')[:2])
                    assert getattr(self.rnn[l], n).size() == p.size()
                    getattr(self.rnn[l], n).data = p.data
                    logger.info('Overwrite %s' % n)
            else:
                for l in range(lm_init.n_layers):
                    for n, p in lm_init.rnn[l].named_parameters():
                        assert getattr(self.rnn[l], n).size() == p.size()
                        getattr(self.rnn[l], n).data = p.data
                        logger.info('Overwrite %s' % n)

            # embedding
            assert self.embed.embed.weight.size() == lm_init.embed.embed.weight.size()
            self.embed.embed.weight.data = lm_init.embed.embed.weight.data
            logger.info('Overwrite %s' % 'embed.embed.weight')

    def reset_parameters(self, param_init):
        """Initialize parameters with uniform distribution."""
        logger = logging.getLogger('training')
        logger.info('===== Initialize %s =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if p.dim() == 1:
                if 'linear_lm_gate.fc.bias' in n:
                    # Initialize bias in gating with -1 for cold fusion
                    nn.init.constant_(p, val=-1)  # bias
                    logger.info('Initialize %s with %s / %.3f' % (n, 'constant', -1))
                else:
                    nn.init.constant_(p, val=0)  # bias
                    logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0))
            elif p.dim() in [2, 4]:
                nn.init.uniform_(p, a=-param_init, b=param_init)
                logger.info('Initialize %s with %s / %.3f' % (n, 'uniform', param_init))
            else:
                raise ValueError

    def start_scheduled_sampling(self):
        self._ss_prob = self.ss_prob

    def forward(self, eouts, elens, ys, task='all', ys_hist=[], teacher_probs=None):
        """Forward computation.

        Args:
            eouts (FloatTensor): `[B, T, dec_n_units]`
            elens (IntTensor): `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
            task (str): all or ys or ys_sub*
            ys_hist (list):
            teacher_probs (FloatTensor): `[B, L, vocab]`
        Returns:
            loss (FloatTensor): `[1]`
            observation (dict):

        """
        observation = {'loss': None,
                       'loss_att': None, 'loss_ctc': None,
                       'loss_lmobj': None,
                       'acc_att': None, 'acc_lmobj': None,
                       'ppl_att': None, 'ppl_lmobj': None}
        w = next(self.parameters())
        loss = w.new_zeros((1,))

        # if self.lm is not None:
        #     self.lm.eval()

        # CTC loss
        if self.ctc_weight > 0 and (task == 'all' or 'ctc' in task):
            loss_ctc = self.ctc(eouts, elens, ys)
            observation['loss_ctc'] = loss_ctc.item()
            if self.mtl_per_batch:
                loss += loss_ctc
            else:
                loss += loss_ctc * self.ctc_weight

        # LM objective for the decoder
        if self.lmobj_weight > 0 and (task == 'all' or 'lmobj' in task):
            loss_lmobj, acc_lmobj, ppl_lmobj = self.forward_lmobj(ys)
            observation['loss_lmobj'] = loss_lmobj.item()
            observation['acc_lmobj'] = acc_lmobj
            observation['ppl_lmobj'] = ppl_lmobj
            if self.mtl_per_batch:
                loss += loss_lmobj
            else:
                loss += loss_lmobj * self.lmobj_weight

        # XE loss
        if self.global_weight - self.ctc_weight > 0 and (task == 'all' or ('ctc' not in task and 'lmobj' not in task)):
            loss_att, acc_att, ppl_att = self.forward_att(eouts, elens, ys, ys_hist,
                                                          teacher_probs=teacher_probs)
            observation['loss_att'] = loss_att.item()
            observation['acc_att'] = acc_att
            observation['ppl_att'] = ppl_att
            if self.mtl_per_batch:
                loss += loss_att
            else:
                loss += loss_att * (self.global_weight - self.ctc_weight)

        observation['loss'] = loss.item()
        return loss, observation

    def forward_lmobj(self, ys):
        """Compute XE loss for LM objective.

        Args:
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
        Returns:
            loss (FloatTensor): `[1]`
            acc (float): accuracy
            ppl (float): perplexity

        """
        bs = len(ys)
        w = next(self.parameters())

        # Append <sos> and <eos>
        eos = w.new_zeros(1).fill_(self.eos)
        ys = [np2tensor(np.fromiter(y[::-1] if self.bwd else y, dtype=np.int64),
                        self.device_id) for y in ys]
        ys_in_pad = pad_list([torch.cat([eos, y], dim=0) for y in ys], self.pad)
        ys_out_pad = pad_list([torch.cat([y, eos], dim=0) for y in ys], self.pad)

        # Initialization
        dstates = self.zero_state(bs)
        cv = w.new_zeros((bs, 1, self.enc_n_units))
        attn_v = w.new_zeros((bs, 1, self.dec_n_units))

        # Pre-computation of embedding
        ys_emb = self.embed(ys_in_pad)

        logits = []
        for t in range(ys_in_pad.size(1)):
            # Recurrency -> Generate
            dstates = self.recurrency(ys_emb[:, t:t + 1], cv, dstates['dstate'])
            if self.loop_type == 'lmdecoder':
                logits_t = self.output_lmobj(dstates['dout_lmdec'])
            elif self.loop_type == 'normal':
                attn_v, _ = self.generate(cv, dstates['dout_gen'])
                logits_t = self.output(attn_v)
            logits.append(logits_t)

        # Compute XE loss for LM objective
        logits = torch.cat(logits, dim=1)
        loss = F.cross_entropy(logits.view((-1, logits.size(2))), ys_out_pad.view(-1),
                               ignore_index=self.pad, size_average=False) / bs
        # TODO(hirofumi): adaptive_softmax

        # Compute token-level accuracy in teacher-forcing
        acc = compute_accuracy(logits, ys_out_pad, self.pad)
        ppl = min(np.exp(loss.item()), np.inf)

        return loss, acc, ppl

    def forward_att(self, eouts, elens, ys, ys_hist=[], return_logits=False, teacher_probs=None):
        """Compute XE loss for the attention-based sequence-to-sequence model.

        Args:
            eouts (FloatTensor): `[B, T, dec_n_units]`
            elens (IntTensor): `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
            ys_hist (list):
            return_logits (bool): return logits for knowledge distillation
            teacher_probs (FloatTensor): `[B, L, vocab]`
        Returns:
            loss (FloatTensor): `[1]`
            acc (float):
            ppl (float):

        """
        bs, xmax = eouts.size()[:2]

        # Append <sos> and <eos>
        eos = eouts.new_zeros(1).fill_(self.eos).long()
        ys = [np2tensor(np.fromiter(y[::-1] if self.bwd else y, dtype=np.int64),
                        self.device_id) for y in ys]
        if self.replace_sos:
            ylens = np2tensor(np.fromiter([y[1:].size(0) + 1 for y in ys], dtype=np.int32))  # +1 for <eos>
            ys_in_pad = pad_list([y for y in ys], self.pad)
            ys_out_pad = pad_list([torch.cat([y[1:], eos], dim=0) for y in ys], self.pad)
        else:
            ylens = np2tensor(np.fromiter([y.size(0) + 1 for y in ys], dtype=np.int32))  # +1 for <eos>
            ys_in_pad = pad_list([torch.cat([eos, y], dim=0) for y in ys], self.pad)
            ys_out_pad = pad_list([torch.cat([y, eos], dim=0) for y in ys], self.pad)

        # Initialization
        if self.contextualize:
            raise NotImplementedError
        else:
            dstates = self.zero_state(bs)
        cv = eouts.new_zeros(bs, 1, self.enc_n_units)
        attn_v = eouts.new_zeros(bs, 1, self.dec_n_units)
        self.score.reset()
        aw = None
        lmstate = None
        lmfeat = eouts.new_zeros(bs, 1, self.dec_n_units)

        # Pre-computation of embedding
        ys_emb = self.embed(ys_in_pad)

        # Create the attention mask
        mask = make_pad_mask(elens, self.device_id).expand(bs, xmax)

        logits = []
        for t in range(ys_in_pad.size(1)):
            is_sample = t > 0 and self._ss_prob > 0 and random.random() < self._ss_prob and self.adaptive_softmax is None

            # Update LM states for LM fusion
            lmout = None
            if self.lm is not None:
                y_lm = self.output(logits[-1]).detach().argmax(-1) if is_sample else ys_in_pad[:, t:t + 1]
                lmout, lmstate = self.lm.decode(y_lm, lmstate)

            # Recurrency -> Score -> Generate
            dec_in = attn_v if self.input_feeding else cv
            y_emb = self.embed(self.output(logits[-1]).detach().argmax(-1)) if is_sample else ys_emb[:, t:t + 1]
            dstates = self.recurrency(y_emb, dec_in, dstates['dstate'])
            cv, aw = self.score(eouts, eouts, dstates['dout_score'], mask, aw)
            attn_v, lmfeat = self.generate(cv, dstates['dout_gen'], lmout)
            logits.append(attn_v)

        logits = torch.cat(logits, dim=1)
        if self.adaptive_softmax is None:
            logits = self.output(logits)
        if return_logits:
            return logits

        # Compute XE sequence loss
        if self.adaptive_softmax is None:
            if teacher_probs is not None:
                # Knowledge distillation
                loss = distillation(logits, teacher_probs, ylens,
                                    temperature=1,
                                    size_average=False) / bs
            else:
                if self.lsm_prob > 0:
                    # Label smoothing
                    loss = cross_entropy_lsm(logits, ys_out_pad, ylens,
                                             lsm_prob=self.lsm_prob, size_average=False) / bs
                else:
                    loss = F.cross_entropy(logits.view((-1, logits.size(2))), ys_out_pad.view(-1),
                                           ignore_index=self.pad, size_average=False) / bs

            # Focal loss
            if self.fl_weight > 0:
                fl = focal_loss(logits, ys_out_pad, ylens,
                                alpha=self.fl_weight,
                                gamma=self.fl_gamma,
                                size_average=False) / bs
                loss = loss * (1 - self.fl_weight) + fl * self.fl_weight
        else:
            loss = self.adaptive_softmax(logits.view((-1, logits.size(2))),
                                         ys_out_pad.view(-1)).loss

        # Compute token-level accuracy in teacher-forcing
        if self.adaptive_softmax is None:
            acc = compute_accuracy(logits, ys_out_pad, pad=self.pad)
        else:
            acc = compute_accuracy(self.adaptive_softmax.log_prob(
                logits.view((-1, logits.size(2)))), ys_out_pad, pad=self.pad)
        ppl = np.exp(loss.item())

        return loss, acc, ppl

    def zero_state(self, batch_size):
        """Initialize decoder state.

        Args:
            batch_size (int):
        Returns:
            dstates (dict):
                dout (FloatTensor): `[B, 1, dec_n_units]`
                dstate (tuple): A tuple of (hxs, cxs)
                    hxs (list of FloatTensor):
                    cxs (list of FloatTensor):

        """
        dstates = {'dout_score': None,  # for attention scoring
                   'dout_gen': None,  # for token generation
                   'dstate': None}
        w = next(self.parameters())
        dstates['dout_score'] = w.new_zeros((batch_size, 1, self.dec_n_units))
        dstates['dout_gen'] = w.new_zeros((batch_size, 1, self.dec_n_units))
        zero_state = w.new_zeros((batch_size, self.dec_n_units))
        hxs = [zero_state for _ in range(self.n_layers)]
        cxs = [zero_state for _ in range(self.n_layers)] if self.rnn_type == 'lstm' else []
        dstates['dstate'] = (hxs, cxs)
        return dstates

    def recurrency(self, y_emb, cv, dstate):
        """Recurrency function.

        Args:
            y_emb (FloatTensor): `[B, 1, emb_dim]`
            cv (FloatTensor): `[B, 1, enc_n_units]`
            dstate (tuple): A tuple of (hxs, cxs)
        Returns:
            dstates_new (dict):
                dout_score (FloatTensor): `[B, 1, n_units]`
                dout_gen (FloatTensor): `[B, 1, n_units]`
                dstate (tuple): A tuple of (hxs, cxs)
                    hxs (list of FloatTensor):
                    cxs (list of FloatTensor):

        """
        hxs, cxs = dstate
        y_emb = y_emb.squeeze(1)
        cv = cv.squeeze(1)

        dstates_new = {'dout_score': None,  # for attention scoring
                       'dout_gen': None,  # for token generation
                       'dout_lmdec': None,
                       'dstate': None}

        # 1st layer
        if self.loop_type == 'lmdecoder':
            if self.rnn_type == 'lstm':
                hxs[0], cxs[0] = self.rnn[0](y_emb, (hxs[0], cxs[0]))
            elif self.rnn_type == 'gru':
                hxs[0] = self.rnn[0](y_emb, hxs[0])
        elif self.loop_type == 'normal':
            if self.rnn_type == 'lstm':
                hxs[0], cxs[0] = self.rnn[0](torch.cat([y_emb, cv], dim=-1), (hxs[0], cxs[0]))
            elif self.rnn_type == 'gru':
                hxs[0] = self.rnn[0](torch.cat([y_emb, cv], dim=-1), hxs[0])
        dout = self.dropout[0](hxs[0])
        if self.n_projs > 0:
            dout = torch.tanh(self.proj[0](dout))

        if self.loop_type == 'lmdecoder' and self.lmobj_weight > 0:
            dstates_new['dout_lmdec'] = dout.unsqueeze(1)

        if self.loop_type == 'normal':
            # the bottom layer
            dstates_new['dout_score'] = dout.unsqueeze(1)

        # after 2nd layers
        for l in range(1, self.n_layers):
            if self.loop_type == 'lmdecoder' and l == 1:
                if self.rnn_type == 'lstm':
                    hxs[l], cxs[l] = self.rnn[l](torch.cat([dout, cv], dim=-1), (hxs[l], cxs[l]))
                elif self.rnn_type == 'gru':
                    hxs[l] = self.rnn[l](torch.cat([dout, cv], dim=-1), hxs[l])
            else:
                if self.rnn_type == 'lstm':
                    hxs[l], cxs[l] = self.rnn[l](dout, (hxs[l], cxs[l]))
                elif self.rnn_type == 'gru':
                    hxs[l] = self.rnn[l](dout, hxs[l])
            dout_tmp = self.dropout[l](hxs[l])
            if self.n_projs > 0:
                dout_tmp = torch.tanh(self.proj[l](dout_tmp))

            if self.loop_type == 'lmdecoder' and l == 1:
                # the bottom layer
                dstates_new['dout_score'] = dout_tmp.unsqueeze(1)

            if self.residual:
                dout = dout_tmp + dout
            else:
                dout = dout_tmp

        # the top layer
        dstates_new['dout_gen'] = dout.unsqueeze(1)
        dstates_new['dstate'] = (hxs[:], cxs[:])
        return dstates_new

    def generate(self, cv, dout, lmout):
        """Generate function.

        Args:
            cv (FloatTensor): `[B, 1, enc_n_units]`
            dout (FloatTensor): `[B, 1, dec_n_units]`
            lmout (FloatTensor): `[B, 1, lm_n_units]`
        Returns:
            attn_v (FloatTensor): `[B, 1, vocab]`
            gated_lmfeat (FloatTensor): `[B, 1 , dec_n_units]`

        """
        gated_lmfeat = None
        if self.lm is not None:
            # LM fusion
            dec_feat = self.linear_dec_feat(torch.cat([dout, cv], dim=-1))

            if self.lm_fusion_type in ['cold', 'deep']:
                lmfeat = self.linear_lm_feat(lmout)
                gate = torch.sigmoid(self.linear_lm_gate(torch.cat([dec_feat, lmfeat], dim=-1)))
                gated_lmfeat = gate * lmfeat
            elif self.lm_fusion_type == 'cold_prob':
                lmfeat = self.linear_lm_feat(self.lm.output(lmout))
                gate = torch.sigmoid(self.linear_lm_gate(torch.cat([dec_feat, lmfeat], dim=-1)))
                gated_lmfeat = gate * lmfeat

            out = self.output_bn(torch.cat([dec_feat, gated_lmfeat], dim=-1))
        else:
            out = self.output_bn(torch.cat([dout, cv], dim=-1))
        attn_v = torch.tanh(out)
        return attn_v, gated_lmfeat

    def greedy(self, eouts, elens, max_len_ratio, idx2token,
               exclude_eos=False, refs_id=None,
               speakers=None, oracle=False):
        """Greedy decoding.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            elens (IntTensor): `[B]`
            max_len_ratio (int): maximum sequence length of tokens
            refs_id (list):
            exclude_eos (bool):
            idx2token ():
            speakers (list):
            oracle (bool):
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw (list): A list of length `[B]`, which contains arrays of size `[L, T, n_heads]`

        """
        bs, xmax, _ = eouts.size()

        # Initialization
        dstates = self.zero_state(bs)
        cv = eouts.new_zeros(bs, 1, self.enc_n_units)
        attn_v = eouts.new_zeros(bs, 1, self.dec_n_units)
        self.score.reset()
        aw = None
        lmstate = None
        lmfeat = eouts.new_zeros(bs, 1, self.dec_n_units)
        if self.replace_sos:
            y = eouts.new_zeros(bs, 1).fill_(refs_id[0][0])
        else:
            y = eouts.new_zeros(bs, 1).fill_(self.eos)

        # Create the attention mask
        mask = make_pad_mask(elens, self.device_id).expand(bs, xmax)

        best_hyps_batch, aws_tmp = [], []
        ylens = torch.zeros(bs).int()
        eos_flags = [False] * bs
        if oracle:
            assert refs_id is not None
            ymax = max([len(refs_id[b]) for b in range(bs)]) + 1
        else:
            ymax = int(math.floor(xmax * max_len_ratio)) + 1
        for t in range(ymax):
            if oracle and t > 0:
                y = eouts.new_zeros(bs, 1)
                for b in range(bs):
                    y[b] = refs_id[b, t - 1]

            # Update LM states for LM fusion
            lmout = None
            if self.lm is not None:
                lmout, lmstate = self.lm.decode(self.lm(y), lmstate)

            # Recurrency -> Score -> Generate
            dec_in = attn_v if self.input_feeding else cv
            y_emb = self.embed(y)
            dstates = self.recurrency(y_emb, dec_in, dstates['dstate'])
            cv, aw = self.score(eouts, eouts, dstates['dout_score'], mask, aw)
            attn_v, lmfeat = self.generate(cv, dstates['dout_gen'], lmout)

            # Pick up 1-best
            if self.adaptive_softmax is None:
                y = self.output(attn_v).argmax(-1)
            else:
                y = self.adaptive_softmax.predict(attn_v.view(-1, attn_v.size(2))).unsqueeze(1)
            best_hyps_batch += [y]
            aws_tmp += [aw]

            # Count lengths of hypotheses
            for b in range(bs):
                if not eos_flags[b]:
                    if y[b].item() == self.eos:
                        eos_flags[b] = True
                    ylens[b] += 1
                    # NOTE: include <eos>

            # Break if <eos> is outputed in all mini-bs
            if sum(eos_flags) == bs:
                break

        # LM state carry over
        self.lmstate_final = lmstate

        # Concatenate in L dimension
        best_hyps_batch = tensor2np(torch.cat(best_hyps_batch, dim=1))
        aws_tmp = tensor2np(torch.stack(aws_tmp, dim=1))

        # Truncate by the first <eos> (<sos> in case of the backward decoder)
        if self.bwd:
            # Reverse the order
            best_hyps = [best_hyps_batch[b, :ylens[b]][::-1] for b in range(bs)]
            aws = [aws_tmp[b, :ylens[b]][::-1] for b in range(bs)]
        else:
            best_hyps = [best_hyps_batch[b, :ylens[b]] for b in range(bs)]
            aws = [aws_tmp[b, :ylens[b]] for b in range(bs)]

        # Exclude <eos> (<sos> in case of the backward decoder)
        if exclude_eos:
            if self.bwd:
                best_hyps = [best_hyps[b][1:] if eos_flags[b] else best_hyps[b] for b in range(bs)]
            else:
                best_hyps = [best_hyps[b][:-1] if eos_flags[b] else best_hyps[b] for b in range(bs)]

        return best_hyps, aws

    def beam_search(self, eouts, elens, params, idx2token,
                    lm=None, lm_rev=None, ctc_log_probs=None,
                    nbest=1, exclude_eos=False,
                    refs_id=None, utt_ids=None, speakers=None,
                    ensmbl_eouts=None, ensmbl_elens=None, ensmbl_decs=[]):
        """Beam search decoding.

        Args:
            eouts (FloatTensor): `[B, T, dec_n_units]`
            elens (IntTensor): `[B]`
            params (dict):
                recog_beam_width (int): size of beam
                recog_max_len_ratio (int): maximum sequence length of tokens
                recog_min_len_ratio (float): minimum sequence length of tokens
                recog_length_penalty (float): length penalty
                recog_coverage_penalty (float): coverage penalty
                recog_coverage_threshold (float): threshold for coverage penalty
                recog_lm_weight (float): weight of LM score
                recog_n_caches (int):
            idx2token (): converter from index to token
            lm (RNNLM or GatedConvLM or TransformerLM):
            lm_rev (RNNLM or GatedConvLM or TransformerLM):
            ctc_log_probs (FloatTensor):
            nbest (int):
            exclude_eos (bool):
            refs_id (list):
            utt_ids (list):
            speakers (list):
            ensmbl_eouts (list): list of FloatTensor
            ensmbl_elens (list) list of list
            ensmbl_decs (list): list of torch.nn.Module
        Returns:
            nbest_hyps_idx (list): A list of length `[B]`, which contains list of N hypotheses
            aws (list): A list of length `[B]`, which contains arrays of size `[L, T]`
            scores (list):
            cache_info (tuple):

        """
        logger = logging.getLogger("decoding")

        bs = eouts.size(0)
        n_models = len(ensmbl_decs) + 1

        oracle = params['recog_oracle']
        beam_width = params['recog_beam_width']
        ctc_weight = params['recog_ctc_weight']
        max_len_ratio = params['recog_max_len_ratio']
        min_len_ratio = params['recog_min_len_ratio']
        lp_weight = params['recog_length_penalty']
        cp_weight = params['recog_coverage_penalty']
        cp_threshold = params['recog_coverage_threshold']
        lm_weight = params['recog_lm_weight']
        gnmt_decoding = params['recog_gnmt_decoding']
        eos_threshold = params['recog_eos_threshold']
        asr_state_carry_over = params['recog_asr_state_carry_over']
        lm_state_carry_over = params['recog_lm_state_carry_over']
        n_caches = params['recog_n_caches']
        cache_theta_sp = params['recog_cache_theta_speech']
        cache_lambda_sp = params['recog_cache_lambda_speech']
        cache_theta_lm = params['recog_cache_theta_lm']
        cache_lambda_lm = params['recog_cache_lambda_lm']
        cache_type = params['recog_cache_type']

        if lm is not None:
            lm.eval()
        if lm_rev is not None:
            lm_rev.eval()

        # For joint CTC-Attention decoding
        if ctc_weight > 0 and ctc_log_probs is not None:
            if self.bwd:
                ctc_prefix_score = CTCPrefixScore(tensor2np(ctc_log_probs)[0][::-1], self.blank, self.eos)
            else:
                ctc_prefix_score = CTCPrefixScore(tensor2np(ctc_log_probs)[0], self.blank, self.eos)

        nbest_hyps_idx, aws, scores = [], [], []
        eos_flags = []
        for b in range(bs):
            # Initialization per utterance
            dstates = self.zero_state(1)
            cv = eouts.new_zeros(1, 1, self.dec_n_units if self.input_feeding else self.enc_n_units)
            self.score.reset()
            lmstate = None

            # Ensemble initialization
            ensmbl_dstate, ensmbl_cv = [], []
            if n_models > 1:
                for dec in ensmbl_decs:
                    ensmbl_dstate += [dec.zero_state(1)]
                    ensmbl_cv += [eouts.new_zeros(1, 1, dec.dec_n_units if dec.input_feeding else dec.enc_n_units)]
                    dec.score.reset()

            if speakers is not None and speakers[b] != self.prev_spk:
                self.reset_global_cache()
            else:
                if lm_state_carry_over and isinstance(lm, RNNLM):
                    lmstate = self.lmstate_final
                if asr_state_carry_over:
                    dstates = self.dstates_final
            self.prev_spk = speakers[b]

            end_hyps = []
            hyps = [{'hyp': [self.eos],
                     'score': 0.0,
                     'hist_score': [0.0],
                     'score_attn': 0.0,
                     'score_ctc': 0.0,
                     'score_lm': 0.0,
                     'dstates': dstates,
                     'cv': cv,
                     'aws': [None],
                     'lmstate': lmstate,
                     'ensmbl_dstate': ensmbl_dstate,
                     'ensmbl_cv': ensmbl_cv,
                     'ensmbl_aws':[[None]] * (n_models - 1),
                     'ctc_state': ctc_prefix_score.initial_state() if ctc_weight > 0 and ctc_log_probs is not None else None,
                     'ctc_score': 0.0,
                     'cache_ids': [],
                     'dict_cache_sp': {},
                     'cache_sp_key': [],
                     'cache_lm_key': [],
                     'cache_idx_hist': [],
                     'cache_sp_attn_hist': [],
                     'cache_lm_attn_hist': [],
                     }]
            if oracle:
                assert refs_id is not None
                ymax = len(refs_id[b]) + 1
            else:
                ymax = int(math.floor(elens[b] * max_len_ratio)) + 1
            for t in range(ymax):
                new_hyps = []
                for beam in hyps:
                    if self.replace_sos and t == 0:
                        prev_idx = refs_id[0][0]
                    else:
                        prev_idx = ([self.eos] + refs_id[b])[t] if oracle else beam['hyp'][-1]

                    if self.lm is not None:
                        # Update LM states for LM fusion
                        lmout, lmstate, lm_log_probs = self.lm.predict(
                            eouts.new_zeros(1, 1).fill_(prev_idx), beam['lmstate'])
                    elif lm_weight > 0 and lm is not None:
                        # Update LM states for shallow fusion
                        lmout, lmstate, lm_log_probs = lm.predict(
                            eouts.new_zeros(1, 1).fill_(prev_idx), beam['lmstate'])
                    else:
                        lmout, lmstate, lm_log_probs = None, None, None

                    # Recurrency for the main model
                    dstates = self.recurrency(self.embed(eouts.new_zeros(1, 1).fill_(prev_idx)),
                                              beam['cv'],
                                              beam['dstates']['dstate'])
                    # Recurrency for the ensemble
                    ensmbl_dstate = []
                    if n_models > 1:
                        ensmbl_dstate = [dec.recurrency(dec.embed(eouts.new_zeros(1, 1).fill_(prev_idx)),
                                                        beam['ensmbl_cv'][i_e],
                                                        beam['ensmbl_dstate'][i_e]['dstate'])
                                         for i_e, dec in enumerate(ensmbl_decs)]

                    # Score for the main model
                    cv, aw = self.score(eouts[b:b + 1, :elens[b]],
                                        eouts[b:b + 1, :elens[b]],
                                        dstates['dout_score'],
                                        None,
                                        beam['aws'][-1])
                    # Score for the ensemble
                    ensmbl_cv, ensmbl_aws = [], []
                    if n_models > 1:
                        for i_e, dec in enumerate(ensmbl_decs):
                            cv_e, aw_e = dec.score(ensmbl_eouts[i_e][b:b + 1, :ensmbl_elens[i_e][b]],
                                                   ensmbl_eouts[i_e][b:b + 1, :ensmbl_elens[i_e][b]],
                                                   ensmbl_dstate[i_e]['dout_score'],
                                                   None,
                                                   beam['ensmbl_aws'][i_e][-1])
                            ensmbl_cv += [cv_e]
                            ensmbl_aws += [aw_e.unsqueeze(0)]  # TODO(hirofumi)] why unsqueeze?

                    # Generate for the main model
                    attn_v, lmfeat = self.generate(cv, dstates['dout_gen'], lmout)
                    if self.adaptive_softmax is None:
                        probs = F.softmax(self.output(attn_v).squeeze(1), dim=1)
                    else:
                        probs = self.adaptive_softmax.log_prob(attn_v.view(-1, attn_v.size(2)))

                    # Cache decoding
                    cache_ids = None
                    cache_sp_attn = None
                    cache_lm_attn = None
                    cache_sp_probs = probs.new_zeros(probs.size())
                    cache_lm_probs = probs.new_zeros(probs.size())
                    if n_caches > 0:
                        assert self.adaptive_softmax is None

                        # Compute inner-product over caches
                        if 'speech_fifo' in cache_type:
                            is_cache = True
                            if 'online' in cache_type and len(self.fifo_cache_ids + beam['cache_ids']) > 0:
                                cache_ids = (self.fifo_cache_ids + beam['cache_ids'])[-n_caches:]
                                cache_sp_key = beam['cache_sp_key'][-n_caches:]
                                if len(cache_sp_key) > 0:
                                    cache_sp_key = torch.cat(cache_sp_key, dim=1)
                                    if self.fifo_cache_sp_key is not None:
                                        cache_sp_key = torch.cat([self.fifo_cache_sp_key, cache_sp_key], dim=1)
                                else:
                                    cache_sp_key = self.fifo_cache_sp_key  # for the first token
                                # Truncate
                                cache_sp_key = cache_sp_key[:, -n_caches:]  # `[1, L, enc_n_units]`
                            elif 'online' not in cache_type and len(self.fifo_cache_ids) > 0:
                                cache_ids = self.fifo_cache_ids
                                cache_sp_key = self.fifo_cache_sp_key
                            else:
                                is_cache = False

                            if is_cache:
                                cache_sp_attn = F.softmax(cache_theta_sp * torch.matmul(
                                    cache_sp_key, torch.cat([cv, dstates['dout_gen']], dim=-1).transpose(2, 1)), dim=1)  # `[1, L, 1]`
                                # Sum all probabilities
                                for c in set(beam['cache_ids']):
                                    for offset in [i for i, key in enumerate(cache_ids) if key == c]:
                                        cache_sp_probs[0, c] += cache_sp_attn[0, offset, 0]
                                probs = (1 - cache_lambda_sp) * probs + cache_lambda_sp * cache_sp_probs

                        if 'lm_fifo' in cache_type:
                            is_cache = True
                            if 'online' in cache_type and len(self.fifo_cache_ids + beam['cache_ids']) > 0:
                                assert lm_weight > 0
                                cache_ids = (self.fifo_cache_ids + beam['cache_ids'])[-n_caches:]
                                cache_lm_key = beam['cache_lm_key'][-n_caches:]
                                if len(cache_lm_key) > 0:
                                    cache_lm_key = torch.cat(cache_lm_key, dim=1)
                                    if self.fifo_cache_lm_key is not None:
                                        cache_lm_key = torch.cat([self.fifo_cache_lm_key, cache_lm_key], dim=1)
                                else:
                                    cache_lm_key = self.fifo_cache_lm_key   # for the first token
                                # Truncate
                                cache_lm_key = cache_lm_key[:, -n_caches:]  # `[1, L, lm_n_units]`
                            elif 'online' not in cache_type and len(self.fifo_cache_ids) > 0:
                                cache_ids = self.fifo_cache_ids
                                cache_lm_key = self.fifo_cache_lm_key
                            else:
                                is_cache = False

                            if is_cache:
                                cache_lm_attn = F.softmax(cache_theta_lm * torch.matmul(
                                    cache_lm_key, lmout.transpose(2, 1)), dim=1)  # `[1, L, 1]`
                                # Sum all probabilities
                                for c in set(beam['cache_ids']):
                                    for offset in [i for i, key in enumerate(cache_ids) if key == c]:
                                        cache_lm_probs[0, c] += cache_lm_attn[0, offset, 0]
                                lm_probs = torch.exp(lm_log_probs)[:, -1]
                                lm_probs = (1 - cache_lambda_lm) * lm_probs + cache_lambda_lm * cache_lm_probs
                                lm_log_probs = torch.log(lm_probs).unsqueeze(1)

                        if 'speech_dict' in cache_type and len(self.dict_cache_sp.keys()) > 0:
                            cache_ids = sorted(list(self.dict_cache_sp.keys()))
                            cache_ids = [self.unk if idx < 0 else idx for idx in cache_ids]
                            cache_sp_key = [v['key']for k, v in sorted(self.dict_cache_sp.items(), key=lambda x: x[0])]
                            cache_sp_key = torch.cat(cache_sp_key, dim=1)
                            cache_sp_attn = F.softmax(cache_theta_sp * torch.matmul(
                                cache_sp_key, torch.cat([cv, dstates['dout_gen']], dim=-1).transpose(2, 1)), dim=1)  # `[1, L, 1]`
                            # Sum all probabilities
                            for offset, c in enumerate(cache_ids):
                                cache_sp_probs[0, c] += cache_sp_attn[0, offset, 0]
                            probs = (1 - cache_lambda_sp) * probs + cache_lambda_sp * cache_sp_probs

                        if 'lm_dict' in cache_type and len(self.dict_cache_lm.keys()) > 0:
                            cache_ids = sorted(list(self.dict_cache_lm.keys()))
                            cache_lm_key = [v['key']for k, v in sorted(self.dict_cache_lm.items(), key=lambda x: x[0])]
                            cache_lm_key = torch.cat(cache_lm_key, dim=1)
                            cache_lm_attn = F.softmax(cache_theta_lm * torch.matmul(
                                cache_lm_key, lmout.transpose(2, 1)), dim=1)  # `[1, L, 1]`
                            # Sum all probabilities
                            for offset, c in enumerate(cache_ids):
                                cache_lm_probs[0, c] += cache_lm_attn[0, offset, 0]
                            probs = (1 - cache_lambda_lm) * probs + cache_lambda_lm * cache_lm_probs

                    if self.adaptive_softmax is None:
                        local_scores_attn = torch.log(probs)
                    else:
                        local_scores_attn = probs  # NOTE: already log-scaled
                    # Generate for the ensemble
                    if n_models > 1:
                        for i_e, dec in enumerate(ensmbl_decs):
                            attn_v_e, _ = dec.generate(ensmbl_cv[i_e],
                                                       ensmbl_dstate[i_e]['dout_gen'],
                                                       lmout)
                            if dec.adaptive_softmax is None:
                                local_scores_attn += F.log_softmax(dec.output(attn_v_e).squeeze(1), dim=1)
                            else:
                                local_scores_attn += dec.adaptive_softmax.log_prob(
                                    attn_v_e.view(-1, attn_v_e.size(2))).unsqueeze(1)
                        local_scores_attn /= n_models

                    # Attention scores
                    scores_attn = beam['score_attn'] + local_scores_attn
                    total_scores = scores_attn * (1 - ctc_weight)

                    # Add LM score <after> top-K selection
                    total_scores_topk, topk_ids = torch.topk(
                        total_scores, k=beam_width, dim=1, largest=True, sorted=True)
                    if lm_weight > 0 and lm is not None:
                        total_scores_lm = beam['score_lm'] + lm_log_probs[0, -1, topk_ids[0]]
                        total_scores_topk += total_scores_lm * lm_weight
                    else:
                        total_scores_lm = torch.zeros((beam_width), dtype=torch.float32)

                    # Add length penalty
                    lp = 1.0
                    if lp_weight > 0:
                        if gnmt_decoding:
                            lp = (math.pow(5 + (len(beam['hyp']) - 1 + 1),
                                           lp_weight)) / math.pow(6, lp_weight)
                            total_scores_topk /= lp
                        else:
                            total_scores_topk += (len(beam['hyp']) - 1 + 1) * lp_weight
                            # NOTE: -1 means removing <sos>

                    # Add coverage penalty
                    cp = 0.0
                    aw_mat = None
                    if cp_weight > 0:
                        aw_mat = torch.stack(beam['aws'][1:] + [aw], dim=-1)  # `[B, T, len(hyp), n_heads]`
                        aw_mat = aw_mat[:, :, :, 0]
                        if gnmt_decoding:
                            aw_mat = torch.log(aw_mat.sum(-1))
                            cp = torch.where(aw_mat < 0, aw_mat, aw_mat.new_zeros(aw_mat.size())).sum()
                            # TODO (hirofumi): mask by elens[b]
                            total_scores_topk += cp * cp_weight
                        else:
                            # Recompute converage penalty in each step
                            if cp_threshold == 0:
                                cp = aw_mat.sum() / self.score.n_heads
                            else:
                                cp = torch.where(aw_mat > cp_threshold, aw_mat,
                                                 aw_mat.new_zeros(aw_mat.size())).sum() / self.score.n_heads
                            total_scores_topk += cp * cp_weight

                    # CTC score
                    if ctc_weight > 0 and ctc_log_probs is not None:
                        ctc_scores, ctc_states = ctc_prefix_score(
                            beam['hyp'], tensor2np(topk_ids[0]), beam['ctc_state'])
                        total_scores_ctc = torch.from_numpy(ctc_scores).cuda(self.device_id)
                        total_scores_topk += total_scores_ctc * ctc_weight
                        # Sort again
                        total_scores_topk, joint_ids_topk = torch.topk(
                            total_scores_topk, k=beam_width, dim=1, largest=True, sorted=True)
                        topk_ids = topk_ids[:, joint_ids_topk[0]]
                    else:
                        total_scores_ctc = torch.zeros((beam_width,), dtype=torch.float32)

                    for k in range(beam_width):
                        idx = topk_ids[0, k].item()
                        total_score = total_scores_topk[0, k].item()

                        # Exclude short hypotheses
                        if idx == self.eos:
                            if len(beam['hyp']) - 1 < elens[b] * min_len_ratio:
                                continue
                            # EOS threshold
                            max_score_no_eos = local_scores_attn[0, :idx].max(0)[0].item()
                            max_score_no_eos = max(max_score_no_eos, local_scores_attn[0, idx + 1:].max(0)[0].item())
                            if local_scores_attn[0, idx].item() <= eos_threshold * max_score_no_eos:
                                continue

                        new_hyps.append(
                            {'hyp': beam['hyp'] + [idx],
                             'score': total_score,
                             'hist_score': beam['hist_score'] + [total_score],
                             'score_attn': scores_attn[0, idx].item(),
                             'score_cp': cp,
                             'score_ctc': total_scores_ctc[k].item(),
                             'score_lm': total_scores_lm[k].item(),
                             'dstates': dstates,
                             'cv': attn_v if self.input_feeding else cv,
                             'aws': beam['aws'] + [aw],
                             'lmstate': lmstate,
                             'ensmbl_dstate': ensmbl_dstate,
                             'ensmbl_cv': ensmbl_cv,
                             'ensmbl_aws': ensmbl_aws,
                             'ctc_state': ctc_states[joint_ids_topk[0, k]] if ctc_log_probs is not None else None,
                             'ctc_score': ctc_scores[joint_ids_topk[0, k]] if ctc_log_probs is not None else None,
                             'cache_ids': beam['cache_ids'] + [idx],
                             'cache_sp_key': beam['cache_sp_key'] + [torch.cat([cv, dstates['dout_gen']], dim=-1)],
                             'cache_lm_key': beam['cache_lm_key'] + [lmout],
                             'cache_idx_hist': beam['cache_idx_hist'] + [cache_ids],
                             'cache_sp_attn_hist': beam['cache_sp_attn_hist'] + [cache_sp_attn] if cache_sp_attn is not None else [],
                             'cache_lm_attn_hist': beam['cache_lm_attn_hist'] + [cache_lm_attn] if cache_lm_attn is not None else [],
                             })

                # Local pruning
                new_hyps_tmp = sorted(new_hyps, key=lambda x: x['score'], reverse=True)[:beam_width]

                # Remove complete hypotheses
                new_hyps = []
                for hyp in new_hyps_tmp:
                    if oracle:
                        if t == len(refs_id[b]):
                            end_hyps += [hyp]
                        else:
                            new_hyps += [hyp]
                    else:
                        if hyp['hyp'][-1] == self.eos:
                            end_hyps += [hyp]
                        else:
                            new_hyps += [hyp]
                if len(end_hyps) >= beam_width:
                    end_hyps = end_hyps[:beam_width]
                    break
                hyps = new_hyps[:]

            # Global pruning
            if len(end_hyps) == 0:
                end_hyps = hyps[:]
            elif len(end_hyps) < nbest and nbest > 1:
                end_hyps.extend(hyps[:nbest - len(end_hyps)])

            # backward LM rescoring
            if lm_rev is not None and lm_weight > 0:
                for i in range(len(end_hyps)):
                    # Initialize
                    lmstate_rev = None
                    score_lm_rev = 0.0
                    lp = 1.0

                    # Append <eos>
                    if end_hyps[i]['hyp'][-1] != self.eos:
                        end_hyps[i]['hyp'].append(self.eos)
                        logger.info('Append <eos>.')

                    if lp_weight > 0 and gnmt_decoding:
                        lp = (math.pow(5 + (len(end_hyps[i]['hyp']) - 1 + 1), lp_weight)) / math.pow(6, lp_weight)
                    for t in range(len(end_hyps[i]['hyp'][::-1]) - 1):
                        lmout_rev, lmstate_rev = lm_rev.decode(
                            eouts.new_zeros(1, 1).fill_(end_hyps[i]['hyp'][::-1][t]), lmstate_rev)
                        lm_log_probs = F.log_softmax(lm_rev.generate(lmout_rev).squeeze(1), dim=-1)
                        score_lm_rev += lm_log_probs[0, -1, end_hyps[i]['hyp'][::-1][t + 1]]
                    if gnmt_decoding:
                        score_lm_rev /= lp  # normalize
                    end_hyps[i]['score'] += score_lm_rev * lm_weight
                    end_hyps[i]['score_lm_rev'] = score_lm_rev

            # Sort by score
            end_hyps = sorted(end_hyps, key=lambda x: x['score'], reverse=True)

            # N-best list
            if self.bwd:
                # Reverse the order
                nbest_hyps_idx += [[np.array(end_hyps[n]['hyp'][1:][::-1]) for n in range(nbest)]]
                aws += [[end_hyps[n]['aws'][1:][::-1] for n in range(nbest)]]
                scores += [[end_hyps[n]['hist_score'][1:][::-1] for n in range(nbest)]]
            else:
                nbest_hyps_idx += [[np.array(end_hyps[n]['hyp'][1:]) for n in range(nbest)]]
                aws += [[end_hyps[n]['aws'][1:] for n in range(nbest)]]
                scores += [[end_hyps[n]['hist_score'][1:] for n in range(nbest)]]

            # Check <eos>
            eos_flag = [True if end_hyps[n]['hyp'][-1] == self.eos else False for n in range(nbest)]
            eos_flags.append(eos_flag)

            if utt_ids is not None:
                logger.info('Utt-id: %s' % utt_ids[b])
            if refs_id is not None and self.vocab == idx2token.vocab:
                logger.info('Ref: %s' % idx2token(refs_id[b]))
            for k in range(len(end_hyps)):
                if self.bwd:
                    logger.info('Hyp: %s' % idx2token(end_hyps[k]['hyp'][1:][::-1]))
                else:
                    logger.info('Hyp: %s' % idx2token(end_hyps[k]['hyp'][1:]))
                logger.info('log prob (hyp): %.7f' % end_hyps[k]['score'])
                logger.info('log prob (hyp, att): %.7f' % (end_hyps[k]['score_attn'] * (1 - ctc_weight)))
                logger.info('log prob (hyp, cp): %.7f' % (end_hyps[k]['score_cp'] * cp_weight))
                if ctc_weight > 0 and ctc_log_probs is not None:
                    logger.info('log prob (hyp, ctc): %.7f' % (end_hyps[k]['score_ctc'] * ctc_weight))
                if lm_weight > 0 and lm is not None:
                    logger.info('log prob (hyp, lm): %.7f' % (end_hyps[k]['score_lm'] * lm_weight))
                    if lm_rev is not None:
                        logger.info('log prob (hyp, lm reverse): %.7f' % (end_hyps[k]['score_lm_rev'] * lm_weight))
                if n_caches > 0:
                    logger.info('Cache: %d' % (len(self.fifo_cache_ids) + len(end_hyps[k]['cache_ids'])))

        # Concatenate in L dimension
        for b in range(len(aws)):
            for n in range(nbest):
                aws[b][n] = tensor2np(torch.stack(aws[b][n], dim=1).squeeze(0))

        # Exclude <eos> (<sos> in case of the backward decoder)
        if exclude_eos:
            if self.bwd:
                nbest_hyps_idx = [[nbest_hyps_idx[b][n][1:] if eos_flags[b][n]
                                   else nbest_hyps_idx[b][n] for n in range(nbest)] for b in range(bs)]
            else:
                nbest_hyps_idx = [[nbest_hyps_idx[b][n][:-1] if eos_flags[b][n]
                                   else nbest_hyps_idx[b][n] for n in range(nbest)] for b in range(bs)]

        # Store in cache
        cache_idx_hist = None
        cache_sp_attn_hist = None
        cache_lm_attn_hist = None
        if n_caches > 0:
            hyp_len = len(end_hyps[0]['hyp'][1:])

            if 'speech_fifo' in cache_type and len(end_hyps[0]['cache_ids']) > 0:
                self.fifo_cache_ids = (self.fifo_cache_ids + end_hyps[0]['cache_ids'])[-n_caches:]
                cache_idx_hist = end_hyps[0]['cache_idx_hist']
                cache_sp_key = end_hyps[0]['cache_sp_key'][-n_caches:]
                cache_sp_key = torch.cat(cache_sp_key, dim=1)
                if self.fifo_cache_sp_key is not None:
                    cache_sp_key = torch.cat([self.fifo_cache_sp_key, cache_sp_key], dim=1)
                # Truncate
                self.fifo_cache_sp_key = cache_sp_key[:, -n_caches:]  # `[1, L, enc_n_units]`
                if len(end_hyps[0]['cache_sp_attn_hist']) > 0:
                    cache_sp_attn_hist = torch.zeros(
                        (1, end_hyps[0]['cache_sp_attn_hist'][-1].size(1), hyp_len), dtype=torch.float32)
                    for i, p in enumerate(end_hyps[0]['cache_sp_attn_hist']):
                        if p.size(1) < n_caches:
                            cache_sp_attn_hist[0, : p.size(1), i] = p[0, :, 0].cpu()
                        else:
                            cache_sp_attn_hist[0, : n_caches - (hyp_len - 1 - i), i] = p[0, (hyp_len - 1 - i):, 0].cpu()

            if ('lm_fifo' in cache_type) and len(end_hyps[0]['cache_ids']) > 0:
                assert lm_weight > 0
                self.fifo_cache_ids = (self.fifo_cache_ids + end_hyps[0]['cache_ids'])[-n_caches:]
                cache_idx_hist = end_hyps[0]['cache_idx_hist']
                cache_lm_key = end_hyps[0]['cache_lm_key'][-n_caches:]
                cache_lm_key = torch.cat(cache_lm_key, dim=1)
                if self.fifo_cache_lm_key is not None:
                    cache_lm_key = torch.cat([self.fifo_cache_lm_key, cache_lm_key], dim=1)
                # Truncate
                self.fifo_cache_lm_key = cache_lm_key[:, -n_caches:]  # `[1, L, lm_n_units]`
                if len(end_hyps[0]['cache_lm_attn_hist']) > 0:
                    cache_lm_attn_hist = torch.zeros((1, end_hyps[0]['cache_lm_attn_hist'][-1].size(1), hyp_len),
                                                     dtype=torch.float32)  # `[B, n_keys, n_values]`
                    for i, p in enumerate(end_hyps[0]['cache_lm_attn_hist']):
                        if p.size(1) < n_caches:
                            cache_lm_attn_hist[0, : p.size(1), i] = p[0, :, 0].cpu()
                        else:
                            cache_lm_attn_hist[0, : n_caches - (hyp_len - 1 - i), i] = p[0, (hyp_len - 1 - i):, 0].cpu()

            if 'speech_dict' in cache_type:
                if len(self.dict_cache_sp.keys()) > 0:
                    cache_idx_hist = sorted(list(self.dict_cache_sp.keys()))
                    cache_idx_hist = [self.unk if i < 0 else i for i in cache_idx_hist]
                    cache_sp_key = [v['key']for k, v in sorted(self.dict_cache_sp.items(), key=lambda x: x[0])]
                    cache_sp_key = torch.cat(cache_sp_key, dim=1)
                    cache_sp_attn_hist = torch.cat(end_hyps[0]['cache_sp_attn_hist'], dim=-1).cpu().numpy()

                for t, idx in enumerate(end_hyps[0]['hyp'][1:]):
                    if idx == self.eos:
                        continue
                    if idx != self.unk and idx in self.dict_cache_sp.keys():
                        if cache_type in ['speech_dict_overwrite', 'joint_dict_overwrite']:
                            self.dict_cache_sp[idx]['value'] = (
                                self.dict_cache_sp[idx]['value'] + end_hyps[0]['cache_sp_key'][t]) / 2
                        else:
                            self.dict_cache_sp[idx]['value'] = end_hyps[0]['cache_sp_key'][t]
                        self.dict_cache_sp[idx]['time'] = self.total_step + t + 1
                        self.dict_cache_sp[idx]['count'] += 1
                    else:
                        if idx == self.unk:
                            self.dict_cache_sp[idx - (self.total_step + t + 1)] = {
                                'key': end_hyps[0]['cache_sp_key'][t],
                                'value': end_hyps[0]['cache_sp_key'][t],
                                'count': 1,
                                'time': self.total_step + t + 1}
                        else:
                            self.dict_cache_sp[idx] = {
                                'key': end_hyps[0]['cache_sp_key'][t],
                                'value': end_hyps[0]['cache_sp_key'][t],
                                'count': 1,
                                'time': self.total_step + t + 1}
                        if len(self.dict_cache_sp.keys()) > n_caches:
                            oldest_id = sorted(self.dict_cache_sp.items(), key=lambda x: x[1]['time'])[0][0]
                            self.dict_cache_sp.pop(oldest_id)
                self.total_step += len(end_hyps[0]['hyp'][1:])

            if 'lm_dict' in cache_type:
                if len(self.dict_cache_lm.keys()) > 0:
                    cache_idx_hist = sorted(list(self.dict_cache_lm.keys()))
                    cache_lm_key = [v['key']for k, v in sorted(self.dict_cache_lm.items(), key=lambda x: x[0])]
                    cache_lm_key = torch.cat(cache_lm_key, dim=1)
                    cache_lm_attn_hist = torch.cat(end_hyps[0]['cache_lm_attn_hist'], dim=-1).cpu().numpy()

                for t, idx in enumerate(end_hyps[0]['hyp'][1:]):
                    if idx == self.eos:
                        continue
                    if idx in self.dict_cache_lm.keys():
                        if cache_type in ['lm_dict_overwrite', 'joint_dict_overwrite']:
                            self.dict_cache_lm[idx]['value'] = (
                                self.dict_cache_lm[idx]['value'] + end_hyps[0]['cache_lm_key'][t]) / 2
                        else:
                            self.dict_cache_lm[idx]['value'] = end_hyps[0]['cache_lm_key'][t]
                        self.dict_cache_lm[idx]['time'] = self.total_step + t + 1
                        self.dict_cache_lm[idx]['count'] += 1
                    else:
                        self.dict_cache_lm[idx] = {
                            'key': end_hyps[0]['cache_lm_key'][t],
                            'value': end_hyps[0]['cache_lm_key'][t],
                            'count': 1,
                            'time': self.total_step + t + 1}
                        if len(self.dict_cache_lm.keys()) > n_caches:
                            oldest_id = sorted(self.dict_cache_lm.items(), key=lambda x: x[1]['time'])[0][0]
                            self.dict_cache_lm.pop(oldest_id)
                self.total_step += len(end_hyps[0]['hyp'][1:])

        # Store ASR/LM state
        self.dstates_final = end_hyps[0]['dstates']
        self.lmstate_final = end_hyps[0]['lmstate']

        if 'speech' in cache_type:
            return nbest_hyps_idx, aws, scores, (cache_sp_attn_hist, cache_idx_hist)
        else:
            return nbest_hyps_idx, aws, scores, (cache_lm_attn_hist, cache_idx_hist)

    def reset_global_cache(self):
        """Reset global cache when the speaker/session is changed."""
        self.fifo_cache_ids = []
        self.fifo_cache_sp_key = None
        self.fifo_cache_lm_key = None
        self.dict_cache_sp = {}
        self.dict_cache_lm = {}
        self.total_step = 0
