#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""RNN decoder (including CTC loss calculation)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import numpy as np
import os
import random
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.criterion import cross_entropy_lsm
from neural_sp.models.lm.rnnlm import RNNLM
from neural_sp.models.modules.cif import CIF
from neural_sp.models.modules.embedding import Embedding
from neural_sp.models.modules.linear import Linear
from neural_sp.models.seq2seq.decoders.ctc import CTC
from neural_sp.models.seq2seq.decoders.ctc import CTCPrefixScore
from neural_sp.models.seq2seq.decoders.decoder_base import DecoderBase
from neural_sp.models.torch_utils import compute_accuracy
from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list
from neural_sp.models.torch_utils import tensor2np
from neural_sp.utils import mkdir_join

import matplotlib
matplotlib.use('Agg')

random.seed(1)


class CIFRNNDecoder(DecoderBase):
    """RNN decoder.

    Args:
        special_symbols (dict):
            eos (int): index for <eos> (shared with <sos>)
            unk (int): index for <unk>
            pad (int): index for <pad>
            blank (int): index for <blank>
        enc_n_units (int): number of units of the encoder outputs
        rnn_type (str): lstm/gru
        n_units (int): number of units in each RNN layer
        n_projs (int): number of units in each projection layer
        n_layers (int): number of RNN layers
        bottleneck_dim (int): dimension of the bottleneck layer before the softmax layer for label generation
        emb_dim (int): dimension of the embedding in target spaces.
        vocab (int): number of nodes in softmax layer
        tie_embedding (bool): tie parameters of the embedding and output layers
        attn_conv_kernel_size (int):
        dropout (float): dropout probability for the RNN layer
        dropout_emb (float): dropout probability for the embedding layer
        lsm_prob (float): label smoothing probability
        ctc_weight (float):
        ctc_lsm_prob (float): label smoothing probability for CTC
        ctc_fc_list (list):
        backward (bool): decode in the backward order
        lm_fusion (RNNLM):
        lm_fusion_type (str): type of LM fusion
        discourse_aware (str): state_carry_over/hierarchical
        lm_init (RNNLM):
        global_weight (float):
        mtl_per_batch (bool):
        param_init (float):
        replace_sos (bool):

    """

    def __init__(self,
                 special_symbols,
                 enc_n_units,
                 rnn_type,
                 n_units,
                 n_projs,
                 n_layers,
                 bottleneck_dim,
                 emb_dim,
                 vocab,
                 tie_embedding=False,
                 attn_conv_kernel_size=0,
                 dropout=0.0,
                 dropout_emb=0.0,
                 lsm_prob=0.0,
                 ctc_weight=0.0,
                 ctc_lsm_prob=0.0,
                 ctc_fc_list=[],
                 backward=False,
                 lm_fusion=None,
                 lm_fusion_type='cold',
                 discourse_aware='',
                 lm_init=None,
                 global_weight=1.0,
                 mtl_per_batch=False,
                 param_init=0.1,
                 replace_sos=False,
                 soft_label_weight=0.0):

        super(CIFRNNDecoder, self).__init__()
        logger = logging.getLogger('training')

        self.eos = special_symbols['eos']
        self.unk = special_symbols['unk']
        self.pad = special_symbols['pad']
        self.blank = special_symbols['blank']
        self.vocab = vocab
        self.rnn_type = rnn_type
        assert rnn_type in ['lstm', 'gru']
        self.enc_n_units = enc_n_units
        self.dec_n_units = n_units
        self.n_projs = n_projs
        self.n_layers = n_layers
        self.lsm_prob = lsm_prob
        self.ctc_weight = ctc_weight
        self.bwd = backward
        self.lm_fusion_type = lm_fusion_type
        self.global_weight = global_weight
        self.mtl_per_batch = mtl_per_batch
        self.replace_sos = replace_sos
        self.soft_label_weight = soft_label_weight

        self.quantity_loss_weight = 1.0

        # for contextualization
        self.discourse_aware = discourse_aware
        self.dstate_prev = None

        # for cache
        self.prev_spk = ''
        self.total_step = 0
        self.dstates_final = None
        self.lmstate_final = None

        if ctc_weight > 0:
            self.ctc = CTC(eos=self.eos,
                           blank=self.blank,
                           enc_n_units=enc_n_units,
                           vocab=vocab,
                           dropout=dropout,
                           lsm_prob=ctc_lsm_prob,
                           fc_list=ctc_fc_list,
                           param_init=param_init)

        if ctc_weight < global_weight:
            # Attention layer
            self.score = CIF(enc_dim=self.enc_n_units,
                             conv_kernel_size=attn_conv_kernel_size,
                             conv_out_channels=self.enc_n_units)

            # Decoder
            self.rnn = nn.ModuleList()
            if self.n_projs > 0:
                self.proj = nn.ModuleList([Linear(n_units, n_projs) for _ in range(n_layers)])
            self.dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(n_layers)])
            rnn = nn.LSTM if rnn_type == 'lstm' else nn.GRU
            dec_odim = enc_n_units + emb_dim
            for l in range(n_layers):
                self.rnn += [rnn(dec_odim, n_units, 1)]
                dec_odim = n_units
                if self.n_projs > 0:
                    dec_odim = n_projs

            # LM fusion
            if lm_fusion is not None:
                self.linear_dec_feat = Linear(dec_odim + enc_n_units, n_units)
                if lm_fusion_type in ['cold', 'deep']:
                    self.linear_lm_feat = Linear(lm_fusion.n_units, n_units)
                    self.linear_lm_gate = Linear(n_units * 2, n_units)
                elif lm_fusion_type == 'cold_prob':
                    self.linear_lm_feat = Linear(lm_fusion.vocab, n_units)
                    self.linear_lm_gate = Linear(n_units * 2, n_units)
                else:
                    raise ValueError(lm_fusion_type)
                self.output_bn = Linear(n_units * 2, bottleneck_dim)

                # fix LM parameters
                for p in lm_fusion.parameters():
                    p.requires_grad = False
            elif discourse_aware == 'hierarchical':
                raise NotImplementedError
            else:
                self.output_bn = Linear(dec_odim + enc_n_units, bottleneck_dim)

            self.embed = Embedding(vocab, emb_dim,
                                   dropout=dropout_emb,
                                   ignore_index=pad)

            self.output = Linear(bottleneck_dim, vocab)
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
            assert lm_init.n_units_null_context == enc_n_units

            # RNN
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
            elif p.dim() in [2, 3, 4]:
                nn.init.uniform_(p, a=-param_init, b=param_init)
                logger.info('Initialize %s with %s / %.3f' % (n, 'uniform', param_init))
            else:
                raise ValueError

    def start_scheduled_sampling(self):
        self._ss_prob = 0

    def forward(self, eouts, elens, ys, task='all', ys_hist=[], teacher_logits=None):
        """Forward computation.

        Args:
            eouts (FloatTensor): `[B, T, dec_n_units]`
            elens (IntTensor): `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
            task (str): all/ys/ys_sub*
            ys_hist (list):
            teacher_logits (FloatTensor): `[B, L, vocab]`
        Returns:
            loss (FloatTensor): `[1]`
            observation (dict):

        """
        observation = {'loss': None, 'loss_att': None, 'loss_ctc': None,
                       'acc_att': None, 'ppl_att': None}
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

        # XE loss
        if self.global_weight - self.ctc_weight > 0 and (task == 'all' or ('ctc' not in task and 'lmobj' not in task)):
            loss_att, acc_att, ppl_att = self.forward_att(eouts, elens, ys)
            observation['loss_att'] = loss_att.item()
            observation['acc_att'] = acc_att
            observation['ppl_att'] = ppl_att
            if self.mtl_per_batch:
                loss += loss_att
            else:
                loss += loss_att * (self.global_weight - self.ctc_weight)

        observation['loss'] = loss.item()
        return loss, observation

    def append_sos_eos(self, ys, bwd=False, replace_sos=False):
        """Append <sos> and <eos> and return padded sequences.

        Args:
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
        Returns:
            ys_in_pad (LongTensor): `[B, L]`
            ys_out_pad (LongTensor): `[B, L]`
            ylens (IntTensor): `[B]`

        """
        w = next(self.parameters())
        eos = w.new_zeros(1).fill_(self.eos).long()
        ys = [np2tensor(np.fromiter(y[::-1] if bwd else y, dtype=np.int64),
                        self.device_id) for y in ys]
        if replace_sos:
            ylens = np2tensor(np.fromiter([y[1:].size(0) + 1 for y in ys], dtype=np.int32))  # +1 for <eos>
            ys_in_pad = pad_list([y for y in ys], self.pad)
            ys_out_pad = pad_list([torch.cat([y[1:], eos], dim=0) for y in ys], self.pad)
        else:
            ylens = np2tensor(np.fromiter([y.size(0) + 1 for y in ys], dtype=np.int32))  # +1 for <eos>
            ys_in_pad = pad_list([torch.cat([eos, y], dim=0) for y in ys], self.pad)
            ys_out_pad = pad_list([torch.cat([y, eos], dim=0) for y in ys], self.pad)
        return ys_in_pad, ys_out_pad, ylens

    def forward_att(self, eouts, elens, ys):
        """Compute XE loss for the CIF model.

        Args:
            eouts (FloatTensor): `[B, T, dec_n_units]`
            elens (IntTensor): `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
        Returns:
            loss (FloatTensor): `[1]`
            acc (float): accuracy for token prediction
            ppl (float): perplexity

        """
        bs, xtime, enc_dim = eouts.size()

        # Append <sos> and <eos>
        ys_in_pad, ys_out_pad, ylens = self.append_sos_eos(ys, self.bwd)

        # Initialization
        dstate = self.zero_state(bs)
        lmouts, lmstate = None, None

        # CIF
        cvs, alpha, aws = self.score(eouts, elens, ylens.cuda(self.device_id).float())
        cvs = torch.cat([eouts.new_zeros(bs, 1, enc_dim), cvs], dim=1)

        # Update LM states for LM fusion
        if self.lm is not None:
            lmouts, _ = self.lm.decode(ys_in_pad, lmstate)

        # Recurrency -> Score -> Generate
        ys_emb = self.embed(ys_in_pad)
        dstate, logits = self.decode_step(cvs[:, :-1], cvs[:, 1:], dstate, ys_emb, lmouts)
        logits = self.output(logits)

        # for attention plot
        if not self.training:
            self.aws = tensor2np(aws)  # `[B, n_heads, L, T]`

        # Compute XE sequence loss
        if self.lsm_prob > 0 and self.training:
            # Label smoothing
            loss = cross_entropy_lsm(logits.view((-1, logits.size(2))), ys_out_pad.view(-1),
                                     self.lsm_prob, self.pad)
        else:
            loss = F.cross_entropy(logits.view((-1, logits.size(2))), ys_out_pad.view(-1),
                                   ignore_index=self.pad, size_average=True)

        # Quantity loss for CIF
        quantity_loss = torch.mean(torch.abs(alpha.sum(1) - ylens.float().cuda(self.device_id)))
        loss += quantity_loss * self.quantity_loss_weight

        # Compute token-level accuracy in teacher-forcing
        acc = compute_accuracy(logits, ys_out_pad, self.pad)
        ppl = np.exp(loss.item())

        # scale loss for CTC
        loss *= ylens.float().mean()

        return loss, acc, ppl

    def decode_step(self, cv_prev, cv, dstate, y_emb, lmout):
        dout, dstate = self.recurrency(torch.cat([y_emb, cv_prev], dim=-1), dstate)
        attn_v = self.generate(cv, dout, lmout)
        return dstate, attn_v

    def zero_state(self, batch_size):
        """Initialize decoder state.

        Args:
            batch_size (int): batch size
        Returns:
            zero_state (dict):
                hxs (FloatTensor): `[n_layers, B, dec_n_units]`
                cxs (FloatTensor): `[n_layers, B, dec_n_units]`

        """
        zero_state = {'hxs': None, 'cxs': None}
        w = next(self.parameters())
        zero_state['hxs'] = w.new_zeros(self.n_layers, batch_size, self.dec_n_units)
        if self.rnn_type == 'lstm':
            zero_state['cxs'] = w.new_zeros(self.n_layers, batch_size, self.dec_n_units)
        return zero_state

    def recurrency(self, inputs, dstate):
        """Recurrency function.

        Args:
            inputs (FloatTensor): `[B, L, emb_dim + enc_n_units]`
            dstate (dict):
                hxs (FloatTensor): `[n_layers, B, dec_n_units]`
                cxs (FloatTensor): `[n_layers, B, dec_n_units]`
        Returns:
            new_dstate (dict):
                dout (FloatTensor): `[B, L, dec_n_units]`
                dstate (dict):
                    hxs (FloatTensor): `[n_layers, B, dec_n_units]`
                    cxs (FloatTensor): `[n_layers, B, dec_n_units]`

        """
        new_dstate = {'hxs': None, 'cxs': None}
        new_hxs, new_cxs = [], []

        dout = inputs.transpose(1, 0)

        for l in range(self.n_layers):
            if self.rnn_type == 'lstm':
                dout, (h_l, c_l) = self.rnn[l](dout, hx=(dstate['hxs'][l:l + 1],
                                                         dstate['cxs'][l:l + 1]))
                new_cxs.append(c_l)
            elif self.rnn_type == 'gru':
                dout, h_l = self.rnn[l](dout, hx=dstate['hxs'][l:l + 1])
            new_hxs.append(h_l)
            dout = self.dropout[l](dout)
            if self.n_projs > 0:
                dout = torch.tanh(self.proj[l](dout))

        # use oupput in the the last layer for label generation
        dout = dout.transpose(1, 0)
        new_dstate['hxs'] = torch.cat(new_hxs, dim=0)
        if self.rnn_type == 'lstm':
            new_dstate['cxs'] = torch.cat(new_cxs, dim=0)
        return dout, new_dstate

    def generate(self, cv, dout, lmout):
        """Generate function.

        Args:
            cv (FloatTensor): `[B, L, enc_n_units]`
            dout (FloatTensor): `[B, L, dec_n_units]`
            lmout (FloatTensor): `[B, L, lm_n_units]`
        Returns:
            attn_v (FloatTensor): `[B, L, vocab]`

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
        return attn_v

    def _plot_attention(self, save_path, n_cols=1):
        """Plot attention."""
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator

        _save_path = mkdir_join(save_path, 'dec_att_weights')

        # Clean directory
        if _save_path is not None and os.path.isdir(_save_path):
            shutil.rmtree(_save_path)
            os.mkdir(_save_path)

        if hasattr(self, 'aws'):
            plt.clf()
            fig, axes = plt.subplots(max(1, self.score.n_heads // n_cols), n_cols,
                                     figsize=(20, 8), squeeze=False)
            for h in range(self.score.n_heads):
                ax = axes[h // n_cols, h % n_cols]
                ax.imshow(self.aws[-1,  h, :, :], aspect="auto")
                ax.grid(False)
                ax.set_xlabel("Input (head%d)" % h)
                ax.set_ylabel("Output (head%d)" % h)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            fig.tight_layout()
            fig.savefig(os.path.join(_save_path, 'attention.png'), dvi=500)
            plt.close()

    def greedy(self, eouts, elens, max_len_ratio, idx2token,
               exclude_eos=False, oracle=False,
               refs_id=None, utt_ids=None, speakers=None):
        """Greedy decoding.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            elens (IntTensor): `[B]`
            max_len_ratio (int): maximum sequence length of tokens
            idx2token (): converter from index to token
            exclude_eos (bool): exclude <eos> from hypothesis
            oracle (bool): teacher-forcing mode
            refs_id (list): reference list
            utt_ids (list): utterance id list
            speakers (list): speaker list
        Returns:
            hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aws (list): A list of length `[B]`, which contains arrays of size `[n_heads, L, T]`

        """
        logger = logging.getLogger("decoding")
        bs, xtime, enc_dim = eouts.size()

        # Initialization
        dstate = self.zero_state(bs)
        lmout, lmstate = None, None
        y = eouts.new_zeros(bs, 1).fill_(refs_id[0][0] if self.replace_sos else self.eos)

        # CIF
        cvs, alpha, aws = self.score(eouts, elens)
        cvs = torch.cat([eouts.new_zeros(bs, 1, enc_dim), cvs], dim=1)

        hyps_batch = []
        ylens = torch.zeros(bs).int()
        eos_flags = [False] * bs
        if oracle:
            assert refs_id is not None
            ytime = max([len(refs_id[b]) for b in range(bs)]) + 1
        else:
            ytime = int(math.floor(xtime * max_len_ratio)) + 1
        for t in range(ytime):
            # Update LM states for LM fusion
            if self.lm is not None:
                lmout, lmstate = self.lm.decode(self.lm(y), lmstate)

            # Recurrency -> Score -> Generate
            y_emb = self.embed(y)
            dstate, attn_v = self.decode_step(cvs[:, t:t + 1], cvs[:, t + 1:t + 2],
                                              dstate, y_emb, lmout)

            # Pick up 1-best
            y = self.output(attn_v).argmax(-1)
            hyps_batch += [y]

            # Count lengths of hypotheses
            for b in range(bs):
                if not eos_flags[b]:
                    if y[b].item() == self.eos:
                        eos_flags[b] = True
                    ylens[b] += 1  # include <eos>

            # Break if <eos> is outputed in all mini-batch
            if sum(eos_flags) == bs:
                break
            if t == ytime - 1:
                break

            if oracle:
                y = eouts.new_zeros(bs, 1)
                for b in range(bs):
                    y[b] = refs_id[b][t]

        # LM state carry over
        self.lmstate_final = lmstate

        # Concatenate in L dimension
        hyps_batch = tensor2np(torch.cat(hyps_batch, dim=1))

        # Truncate by the first <eos> (<sos> in case of the backward decoder)
        if self.bwd:
            # Reverse the order
            hyps = [hyps_batch[b, :ylens[b]][::-1] for b in range(bs)]
        else:
            hyps = [hyps_batch[b, :ylens[b]] for b in range(bs)]

        # Exclude <eos> (<sos> in case of the backward decoder)
        if exclude_eos:
            if self.bwd:
                hyps = [hyps[b][1:] if eos_flags[b] else hyps[b] for b in range(bs)]
            else:
                hyps = [hyps[b][:-1] if eos_flags[b] else hyps[b] for b in range(bs)]

        for b in range(bs):
            if utt_ids is not None:
                logger.info('Utt-id: %s' % utt_ids[b])
            if refs_id is not None and self.vocab == idx2token.vocab:
                logger.info('Ref: %s' % idx2token(refs_id[b]))
            if self.bwd:
                logger.info('Hyp: %s' % idx2token(hyps[b][::-1]))
            else:
                logger.info('Hyp: %s' % idx2token(hyps[b]))

        return hyps, None

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
            idx2token (): converter from index to token
            lm (RNNLM/GatedConvLM/TransformerLM):
            lm_rev (RNNLM/GatedConvLM/TransformerLM):
            ctc_log_probs (FloatTensor):
            nbest (int):
            exclude_eos (bool): exclude <eos> from hypothesis
            refs_id (list): reference list
            utt_ids (list): utterance id list
            speakers (list): speaker list
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

        bs, _, enc_dim = eouts.size()
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

        # CIF
        cvs, alpha, aws = self.score(eouts, elens, max_len=200)
        cvs = torch.cat([eouts.new_zeros(bs, 1, enc_dim), cvs], dim=1)
        # TODO: fix later

        nbest_hyps_idx, aws, scores = [], [], []
        eos_flags = []
        for b in range(bs):
            # Initialization per utterance
            dstate = self.zero_state(1)
            lmstate = None

            # Ensemble initialization
            ensmbl_dstate, ensmbl_cv = [], []
            if n_models > 1:
                for dec in ensmbl_decs:
                    ensmbl_dstate += [dec.zero_state(1)]
                    dec.score.reset()

            # TODO
            if speakers is not None and speakers[b] == self.prev_spk:
                if lm_state_carry_over and isinstance(lm, RNNLM):
                    lmstate = self.lmstate_final
                if asr_state_carry_over:
                    dstate = self.dstates_final
            self.prev_spk = speakers[b]

            end_hyps = []
            hyps = [{'hyp': [self.eos],
                     'score': 0.0,
                     'hist_score': [0.0],
                     'score_attn': 0.0,
                     'score_ctc': 0.0,
                     'score_lm': 0.0,
                     'dstate': dstate,
                     'aws': [None],
                     'lmstate': lmstate,
                     'ensmbl_dstate': ensmbl_dstate,
                     'ensmbl_cv': ensmbl_cv,
                     'ensmbl_aws':[[None]] * (n_models - 1),
                     'ctc_state': ctc_prefix_score.initial_state() if ctc_weight > 0 and ctc_log_probs is not None else None,
                     'ctc_score': 0.0,
                     }]
            if oracle:
                assert refs_id is not None
                ytime = len(refs_id[b]) + 1
            else:
                ytime = int(math.floor(elens[b] * max_len_ratio)) + 1
            for t in range(ytime):
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

                    # for the main model
                    y_emb = self.embed(eouts.new_zeros(1, 1).fill_(prev_idx))
                    dstate, attn_v = self.decode_step(
                        cvs[b:b + 1, t:t + 1],
                        cvs[b:b + 1, t + 1:t + 2],
                        beam['dstate'], y_emb, lmout)
                    probs = F.softmax(self.output(attn_v).squeeze(1), dim=1)

                    # for the ensemble
                    ensmbl_dstate, ensmbl_cv, ensmbl_aws = [], [], []
                    ensmbl_log_probs = 0.
                    if n_models > 1:
                        for i_e, dec in enumerate(ensmbl_decs):
                            y_emb = dec.embed(eouts.new_zeros(1, 1).fill_(prev_idx))
                            ensmbl_dstate, cv_e, aw_e, attn_v_e = dec.decode_step(
                                ensmbl_eouts[i_e][b:b + 1, t:t + 1],
                                ensmbl_eouts[i_e][b:b + 1, t + 1:t + 2],
                                beam['ensmbl_dstate'][i_e], beam['ensmbl_cv'], y_emb,
                                lmout)

                            ensmbl_dstate += [ensmbl_dstate]
                            ensmbl_cv += [cv_e]
                            ensmbl_aws += [aw_e.unsqueeze(0)]  # TODO(hirofumi): unsqueeze?
                            ensmbl_log_probs += F.log_softmax(dec.output(attn_v_e).squeeze(1), dim=1)

                    local_scores_attn = torch.log(probs)

                    # Ensemble in log-scale
                    if n_models > 1:
                        local_scores_attn += ensmbl_log_probs
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
                            lp = math.pow(6 + len(beam['hyp'][1:]), lp_weight) / math.pow(6, lp_weight)
                            total_scores_topk /= lp
                        else:
                            total_scores_topk += (len(beam['hyp'][1:]) + 1) * lp_weight

                    # Add coverage penalty
                    cp = 0.0
                    aw_mat = None
                    if cp_weight > 0:
                        aw_mat = torch.stack(beam['aws'][1:] + [aw], dim=-1)  # `[B, T, L, n_heads]`
                        aw_mat = aw_mat[:, :, :, 0]
                        if gnmt_decoding:
                            aw_mat = torch.log(aw_mat.sum(-1))
                            cp = torch.where(aw_mat < 0, aw_mat, aw_mat.new_zeros(aw_mat.size())).sum()
                            # TODO(hirofumi): mask by elens[b]
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
                             'dstate': dstate,
                             # 'aws': beam['aws'] + [aw],
                             'lmstate': lmstate,
                             'ensmbl_dstate': ensmbl_dstate,
                             'ensmbl_cv': ensmbl_cv,
                             'ensmbl_aws': ensmbl_aws,
                             'ctc_state': ctc_states[joint_ids_topk[0, k]] if ctc_log_probs is not None else None,
                             'ctc_score': ctc_scores[joint_ids_topk[0, k]] if ctc_log_probs is not None else None,
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
                        lp = math.pow(6 + (len(end_hyps[i]['hyp'][1:])), lp_weight) / math.pow(6, lp_weight)
                    for t in range(len(end_hyps[i]['hyp'][1:])):
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

        # Store ASR/LM state
        self.dstates_final = end_hyps[0]['dstate']
        self.lmstate_final = end_hyps[0]['lmstate']

        return nbest_hyps_idx, aws, scores, None
