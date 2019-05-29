#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""RNN Transducer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import logging
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.criterion import kldiv_lsm_ctc
from neural_sp.models.modules.embedding import Embedding
from neural_sp.models.modules.linear import LinearND
from neural_sp.models.seq2seq.decoders.ctc_beam_search import BeamSearchDecoder
from neural_sp.models.seq2seq.decoders.ctc_beam_search import CTCPrefixScore
from neural_sp.models.seq2seq.decoders.ctc_greedy import GreedyDecoder
from neural_sp.models.seq2seq.decoders.decoder_base import DecoderBase
from neural_sp.models.torch_utils import compute_accuracy
from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list
from neural_sp.models.torch_utils import tensor2np

random.seed(1)


class RNNTransducer(DecoderBase):
    """RNN decoder.

    Args:
        eos (int): index for <eos> (shared with <sos>)
        unk (int): index for <unk>
        pad (int): index for <pad>
        blank (int): index for <blank>
        enc_n_units (int):
        rnn_type (str): lstm_transducer or gru_transducer
        n_units (int): number of units in each RNN layer
        n_projs (int): number of units in each projection layer
        n_layers (int): number of RNN layers
        residual (bool):
        bottleneck_dim (int): dimension of the bottleneck layer before the softmax layer for label generation
        emb_dim (int): dimension of the embedding in target spaces.
        vocab (int): number of nodes in softmax layer
        tie_embedding (bool):
        dropout (float): probability to drop nodes in the RNN layer
        dropout_emb (float): probability to drop nodes of the embedding layer
        lsm_prob (float): label smoothing probability
        ctc_weight (float):
        ctc_fc_list (list):
        lm_init (RNNLM):
        lmobj_weight (float):
        share_lm_softmax (bool):
        global_weight (float):
        mtl_per_batch (bool):
        param_init (float):
        end_pointing (bool):

    """

    def __init__(self,
                 eos,
                 unk,
                 pad,
                 blank,
                 enc_n_units,
                 rnn_type,
                 n_units,
                 n_projs,
                 n_layers,
                 residual,
                 bottleneck_dim,
                 emb_dim,
                 vocab,
                 tie_embedding=False,
                 dropout=0.0,
                 dropout_emb=0.0,
                 lsm_prob=0.0,
                 ctc_weight=0.0,
                 ctc_fc_list=[],
                 lm_init=None,
                 lmobj_weight=0.0,
                 share_lm_softmax=False,
                 global_weight=1.0,
                 mtl_per_batch=False,
                 param_init=0.1,
                 end_pointing=False):

        super(RNNTransducer, self).__init__()
        logger = logging.getLogger('training')

        self.eos = eos
        self.unk = unk
        self.pad = pad
        self.blank = blank
        self.vocab = vocab
        self.rnn_type = rnn_type
        assert rnn_type in ['lstm_transducer', 'gru_transducer']
        self.enc_n_units = enc_n_units
        self.dec_n_units = n_units
        self.n_projs = n_projs
        self.n_layers = n_layers
        self.residual = residual
        self.lsm_prob = lsm_prob
        self.ctc_weight = ctc_weight
        self.ctc_fc_list = ctc_fc_list
        self.lmobj_weight = lmobj_weight
        self.share_lm_softmax = share_lm_softmax
        self.global_weight = global_weight
        self.mtl_per_batch = mtl_per_batch
        self.end_pointing = end_pointing

        if ctc_weight > 0:
            # Fully-connected layers for CTC
            if len(ctc_fc_list) > 0:
                fc_layers = OrderedDict()
                for i in range(len(ctc_fc_list)):
                    input_dim = enc_n_units if i == 0 else ctc_fc_list[i - 1]
                    fc_layers['fc' + str(i)] = LinearND(input_dim, ctc_fc_list[i], dropout=dropout)
                fc_layers['fc' + str(len(ctc_fc_list))] = LinearND(ctc_fc_list[-1], vocab, dropout=0)
                self.output_ctc = nn.Sequential(fc_layers)
            else:
                self.output_ctc = LinearND(enc_n_units, vocab)
            self.decode_ctc_greedy = GreedyDecoder(blank=blank)
            self.decode_ctc_beam = BeamSearchDecoder(blank=blank)
            import warpctc_pytorch
            self.warpctc_loss = warpctc_pytorch.CTCLoss(size_average=True)

        if ctc_weight < global_weight:
            import warprnnt_pytorch
            self.warprnnt_loss = warprnnt_pytorch.RNNTLoss()

            # for MTL with LM objective
            if lmobj_weight > 0:
                if share_lm_softmax:
                    self.output_lmobj = self.output  # share paramters
                else:
                    self.output_lmobj = LinearND(n_units, vocab)

            # Prediction network
            self.fast_impl = False
            rnn = nn.LSTM if rnn_type == 'lstm_transducer' else nn.GRU
            if n_projs == 0 and not residual:
                self.fast_impl = True
                self.rnn = rnn(emb_dim, n_units, n_layers,
                               bias=True,
                               batch_first=True,
                               dropout=dropout,
                               bidirectional=False)
                # NOTE: pytorch introduces a dropout layer on the outputs of each layer EXCEPT the last layer
                dec_idim = n_units
                self.dropout_top = nn.Dropout(p=dropout)
            else:
                self.rnn = torch.nn.ModuleList()
                self.dropout = torch.nn.ModuleList()
                if n_projs > 0:
                    self.proj = torch.nn.ModuleList()
                dec_idim = emb_dim
                for l in range(n_layers):
                    self.rnn += [rnn(dec_idim, n_units, 1,
                                     bias=True,
                                     batch_first=True,
                                     dropout=0,
                                     bidirectional=False)]
                    self.dropout += [nn.Dropout(p=dropout)]
                    dec_idim = n_units

                    if l != self.n_layers - 1 and n_projs > 0:
                        self.proj += [LinearND(dec_idim, n_projs)]
                        dec_idim = n_projs

            self.embed = Embedding(vocab, emb_dim,
                                   dropout=dropout_emb,
                                   ignore_index=pad)

            self.output_bn = LinearND(n_units * 2, bottleneck_dim, bias=True)
            self.output = LinearND(bottleneck_dim, vocab)

        # Initialize parameters
        self.reset_parameters(param_init)

        # prediction network initialization with pre-trained LM
        if lm_init is not None:
            assert lm_init.vocab == vocab
            assert lm_init.n_units == n_units
            assert lm_init.n_projs == n_projs
            assert lm_init.n_layers == n_layers
            assert lm_init.residual == residual

            param_dict = dict(lm_init.named_parameters())
            for n, p in self.named_parameters():
                if n in param_dict.keys() and p.size() == param_dict[n].size():
                    if 'output' in n:
                        continue
                    p.data = param_dict[n].data
                    logger.info('Overwrite %s' % n)

    def reset_parameters(self, param_init):
        """Initialize parameters with uniform distribution."""
        logger = logging.getLogger('training')
        logger.info('===== Initialize %s =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.constant_(p, val=0)  # bias
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0))
            elif p.dim() in [2, 4]:
                nn.init.uniform_(p, a=-param_init, b=param_init)
                logger.info('Initialize %s with %s / %.3f' % (n, 'uniform', param_init))
            else:
                raise ValueError

    def start_scheduled_sampling(self):
        self._ss_prob = self.ss_prob

    def forward(self, eouts, elens, ys, task='all', ys_hist=[], teacher_dist=None):
        """Forward computation.

        Args:
            eouts (FloatTensor): `[B, T, dec_n_units]`
            elens (list): A list of length `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
            task (str): all or ys or ys_sub*
            ys_hist (list):
            teacher_dist (FloatTensor): `[B, L, vocab]`
        Returns:
            loss (FloatTensor): `[1]`
            observation (dict):

        """
        observation = {'loss': None,
                       'loss_transducer': None, 'loss_ctc': None,
                       'loss_lmobj': None,
                       'acc_lmobj': None,
                       'ppl_lmobj': None}
        w = next(self.parameters())
        loss = w.new_zeros((1,))

        # CTC loss
        if self.ctc_weight > 0 and (task == 'all' or 'ctc' in task):
            loss_ctc = self.forward_ctc(eouts, elens, ys)
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
            loss_transducer = self.forward_rnnt(eouts, elens, ys)
            observation['loss_transducer'] = loss_transducer.item()
            if self.mtl_per_batch:
                loss += loss_transducer
            else:
                loss += loss_transducer * (self.global_weight - self.ctc_weight)

        observation['loss'] = loss.item()
        return loss, observation

    def forward_ctc(self, eouts, elens, ys):
        """Compute CTC loss.

        Args:
            eouts (FloatTensor): `[B, T, dec_n_units]`
            elens (list): A list of length `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
        Returns:
            loss (FloatTensor): `[B, L, vocab]`

        """
        # Concatenate all elements in ys for warpctc_pytorch
        elens = np2tensor(np.fromiter(elens, dtype=np.int64)).int()
        ylens = np2tensor(np.fromiter([len(y) for y in ys], dtype=np.int64)).int()
        ys_ctc = torch.cat([np2tensor(np.fromiter(y, dtype=np.int64)).long() for y in ys], dim=0).int()
        # NOTE: do not copy to GPUs here

        # Compute CTC loss
        logits = self.output_ctc(eouts)
        loss = self.warpctc_loss(logits.transpose(1, 0).cpu(),  # time-major
                                 ys_ctc, elens, ylens)
        # NOTE: ctc loss has already been normalized by bs
        # NOTE: index 0 is reserved for blank in warpctc_pytorch
        if self.device_id >= 0:
            loss = loss.cuda(self.device_id)

        # Label smoothing for CTC
        if self.lsm_prob > 0:
            loss = loss * (1 - self.lsm_prob) + kldiv_lsm_ctc(logits,
                                                              ylens=elens,
                                                              size_average=True) * self.lsm_prob

        return loss

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
        eos = w.new_zeros(1).fill_(self.eos).long()
        ys = [np2tensor(np.fromiter(y, dtype=np.int64), self.device_id).long() for y in ys]
        ys_in_pad = pad_list([torch.cat([eos, y], dim=0) for y in ys], self.pad)
        ys_out_pad = pad_list([torch.cat([y, eos], dim=0) for y in ys], self.pad)

        # Update prediction network
        out_pred, _ = self.recurrency(self.embed(ys_in_pad), None)
        logits = self.output_lmobj(out_pred)

        # Compute XE loss for LM objective
        loss = F.cross_entropy(logits.view((-1, logits.size(2))), ys_out_pad.view(-1),
                               ignore_index=self.pad, size_average=False) / bs

        # Compute token-level accuracy in teacher-forcing
        acc = compute_accuracy(logits, ys_out_pad, self.pad)
        ppl = min(np.exp(loss.item()), np.inf)

        return loss, acc, ppl

    def forward_rnnt(self, eouts, elens, ys, ys_hist=[], return_logits=False, teacher_dist=None):
        """Compute XE loss for the attention-based sequence-to-sequence model.

        Args:
            eouts (FloatTensor): `[B, T, dec_n_units]`
            elens (list): A list of length `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
            ys_hist (list):
            return_logits (bool): return logits for knowledge distillation
            teacher_dist (FloatTensor): `[B, L, vocab]`
        Returns:
            loss (FloatTensor): `[1]`
            acc (float):
            ppl (float):

        """
        # Append <null> and <eos>
        null = eouts.new_zeros(1).long()
        if self.end_pointing:
            _ys = [np2tensor(np.fromiter([y] + [self.eos], dtype=np.int64), self.device_id).long() for y in ys]
        else:
            _ys = [np2tensor(np.fromiter(y, dtype=np.int64), self.device_id).long() for y in ys]
        ys_in_pad = pad_list([torch.cat([null, y], dim=0) for y in _ys], self.pad)
        ylens = np2tensor(np.fromiter([y.size(0) for y in _ys], dtype=np.int64), self.device_id).int()
        ys_out_pad = pad_list(_ys, 0).int()

        # Update prediction network
        out_pred, _ = self.recurrency(self.embed(ys_in_pad), None)
        out_pred = out_pred.unsqueeze(1)  # `[B, 1, L, dec_n_units]`
        eouts = eouts.unsqueeze(2)  # `[B, 1, L, dec_n_units]`

        # Compute output distribution
        size = [max(i, j) for i, j in zip(eouts.size()[:-1], out_pred.size()[:-1])]
        eouts = eouts.expand(torch.Size(size + [eouts.shape[-1]]))
        out_pred = out_pred.expand(torch.Size(size + [out_pred.shape[-1]]))
        out = self.joint_dist(eouts, out_pred)

        # Compute Transducer loss
        elens = np2tensor(np.fromiter(elens, dtype=np.int64), self.device_id).int()
        log_probs = F.log_softmax(out, dim=-1)
        loss = self.warprnnt_loss(log_probs, ys_out_pad, elens, ylens)
        # NOTE: Transducer loss has already been normalized by bs
        # NOTE: index 0 is reserved for blank in warprnnt_pytorch

        # Label smoothing for Transducer
        # if self.lsm_prob > 0:
        #     loss = loss * (1 - self.lsm_prob) + kldiv_lsm_ctc(logits,
        #                                                       ylens=elens,
        #                                                       size_average=True) * self.lsm_prob
        # TODO(hirofumi): this leads to out of memory

        return loss

    def joint_dist(self, out_trans, out_pred):
        """
        Args:
            out_trans (FloatTensor): `[B, T, L, dec_n_units]`
            out_pred (FloatTensor): `[B, T, L, dec_n_units]`
        Returns:
            out (FloatTensor): `[B, T, L, vocab]`
        """
        dim = len(out_trans.shape) - 1
        out = torch.tanh(self.output_bn(torch.cat((out_trans, out_pred), dim=dim)))
        return self.output(out)

    def recurrency(self, ys_emb, dstates):
        """Update prediction network.
        Args:
            ys_emb (FloatTensor): `[B, L, emb_dim]`
            dstates ():
        Returns:
            out_pred (FloatTensor):
            dstates ():

        """
        if dstates is None or dstates[0] is None:
            dstates = self.initialize_hidden(ys_emb.size(0))

        residual = None
        if self.fast_impl:
            # Path through RNN
            ys_emb, dstates = self.rnn(ys_emb, hx=dstates)
            ys_emb = self.dropout_top(ys_emb)
        else:
            new_hxs, new_cxs = [], []
            for l in range(self.n_layers):
                # Path through RNN
                if self.rnn_type == 'lstm_transducer':
                    ys_emb, (h_l, c_l) = self.rnn[l](ys_emb, hx=(dstates[0][l:l + 1], dstates[1][l:l + 1]))
                    new_cxs.append(c_l)
                elif self.rnn_type == 'gru_transducer':
                    ys_emb, h_l = self.rnn[l](ys_emb, hx=dstates[0][l:l + 1])
                new_hxs.append(h_l)
                ys_emb = self.dropout[l](ys_emb)

                # Projection layer
                if l < self.n_layers - 1 and self.n_projs > 0:
                    ys_emb = torch.tanh(self.proj[l](ys_emb))

                # Residual connection
                if self.residual and l > 0:
                    ys_emb = ys_emb + residual
                residual = ys_emb
                # NOTE: Exclude residual connection from the raw inputs

        return ys_emb, dstates

    def initialize_hidden(self, batch_size):
        """Initialize hidden states.

        Args:
            batch_size (int):
        Returns:
            hidden (tuple):
                hxs (FloatTensor): `[n_layers, B, dec_n_units]`
                cxs (FloatTensor): `[n_layers, B, dec_n_units]`

        """
        w = next(self.parameters())
        if self.fast_impl:
            h_n = w.new_zeros(self.n_layers, batch_size, self.dec_n_units)
            c_n = None
            if self.rnn_type == 'lstm_transducer':
                c_n = w.new_zeros(self.n_layers, batch_size, self.dec_n_units)
            return (h_n, c_n)
        else:
            hxs, cxs = [], []
            for l in range(self.n_layers):
                if self.rnn_type == 'lstm_transducer':
                    cxs.append(w.new_zeros(1, batch_size, self.dec_n_units))
                hxs.append(w.new_zeros(1, batch_size, self.dec_n_units))
            hxs = torch.cat(hxs, dim=0)
            if self.rnn_type == 'lstm_transducer':
                cxs = torch.cat(cxs, dim=0)
            return (hxs, cxs)

    def greedy(self, eouts, elens, max_len_ratio,
               exclude_eos=False, idx2token=None, refs_id=None,
               speakers=None, oracle=False):
        """Greedy decoding in the inference stage.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            elens (list): A list of length `[B]`
            max_len_ratio (int): maximum sequence length of tokens
            exclude_eos (bool):
            idx2token ():
            refs_id (list):
            speakers (list):
            oracle (bool):
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw (list): A list of length `[B]`, which contains arrays of size `[L, T, n_heads]`

        """
        bs, xlen_max, _ = eouts.size()

        best_hyps = []
        for b in range(bs):
            best_hyp_utt = []
            # Initialization
            y = eouts.new_zeros(1, 1).long()
            out_pred, dstates = self.recurrency(self.embed(y), None)
            for t in range(xlen_max):
                # Pick up 1-best per frame
                out = self.joint_dist(eouts[b:b + 1, t:t + 1], out_pred)
                y = F.log_softmax(out, dim=-1).detach().argmax(-1)

                # Update prediction network only when predicting blank labels
                if y[0].item() != self.blank:
                    if not (self.end_pointing and y[0].item() == self.eos and exclude_eos):
                        best_hyp_utt += [y[0]]

                    if oracle:
                        y = eouts.new_zeros(1, 1).fill_(refs_id[b, len(best_hyp_utt) - 1]).long()
                    else:
                        out_pred, dstates = self.recurrency(self.embed(y), dstates)

                    # early stop
                    if self.end_pointing and y[0].item() == self.eos:
                        break

            best_hyps += [tensor2np(torch.cat(best_hyp_utt, dim=0))]

        return best_hyps, None

    def beam_search(self):
        raise NotImplementedError
