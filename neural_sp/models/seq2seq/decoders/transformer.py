#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer decoder (including CTC loss calculation)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import os
import random
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.criterion import cross_entropy_lsm
from neural_sp.models.criterion import focal_loss
from neural_sp.models.modules.embedding import Embedding
from neural_sp.models.modules.linear import LinearND
from neural_sp.models.modules.transformer import PositionalEncoding
from neural_sp.models.modules.transformer import TransformerDecoderBlock
from neural_sp.models.seq2seq.decoders.ctc import CTC
# from neural_sp.models.seq2seq.decoders.ctc import CTCPrefixScore
from neural_sp.models.seq2seq.decoders.decoder_base import DecoderBase
from neural_sp.models.torch_utils import compute_accuracy
from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list
from neural_sp.models.torch_utils import tensor2np
from neural_sp.utils import mkdir_join

import matplotlib
matplotlib.use('Agg')

random.seed(1)


class TransformerDecoder(DecoderBase):
    """Transformer decoder.

    Args:
        eos (int): index for <eos> (shared with <sos>)
        unk (int): index for <unk>
        pad (int): index for <pad>
        blank (int): index for <blank>
        enc_n_units (int):
        attn_type (str):
        attn_n_heads (int): number of attention heads
        n_layers (int): number of decoder layers.
        d_model (int): size of the model
        d_ff (int): size of the inner FF layer
        vocab (int): number of nodes in softmax layer
        tie_embedding (bool):
        pe_type (str): concat or add or learn or False
        layer_norm_eps (float):
        dropout (float): dropout probability for linear layers
        dropout_emb (float): dropout probability for the embedding layer
        dropout_att (float): dropout probability for attention distributions
        lsm_prob (float): label smoothing probability
        fl_weight (float):
        fl_gamma (float):
        ctc_weight (float):
        ctc_lsm_prob (float): label smoothing probability for CTC
        ctc_fc_list (list):
        backward (bool): decode in the backward order
        global_weight (float):
        mtl_per_batch (bool):
        adaptive_softmax (bool):

    """

    def __init__(self,
                 eos,
                 unk,
                 pad,
                 blank,
                 enc_n_units,
                 attn_type,
                 attn_n_heads,
                 n_layers,
                 d_model,
                 d_ff,
                 vocab,
                 tie_embedding=False,
                 pe_type='add',
                 layer_norm_eps=1e-6,
                 dropout=0.0,
                 dropout_emb=0.0,
                 dropout_att=0.0,
                 lsm_prob=0.0,
                 fl_weight=0.0,
                 fl_gamma=2.0,
                 ctc_weight=0.0,
                 ctc_lsm_prob=0.0,
                 ctc_fc_list=[],
                 backward=False,
                 global_weight=1.0,
                 mtl_per_batch=False,
                 adaptive_softmax=False):

        super(TransformerDecoder, self).__init__()
        logger = logging.getLogger('training')

        self.eos = eos
        self.unk = unk
        self.pad = pad
        self.blank = blank
        self.enc_n_units = enc_n_units
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = attn_n_heads
        self.pe_type = pe_type
        self.lsm_prob = lsm_prob
        self.fl_weight = fl_weight
        self.fl_gamma = fl_gamma
        self.ctc_weight = ctc_weight
        self.bwd = backward
        self.global_weight = global_weight
        self.mtl_per_batch = mtl_per_batch

        if ctc_weight > 0:
            self.ctc = CTC(eos=eos,
                           blank=blank,
                           enc_n_units=enc_n_units,
                           vocab=vocab,
                           dropout=dropout,
                           lsm_prob=ctc_lsm_prob,
                           fc_list=ctc_fc_list,
                           param_init=0.1)

        if ctc_weight < global_weight:
            self.embed = Embedding(vocab, d_model,
                                   dropout=0,  # NOTE: do not apply dropout here
                                   ignore_index=pad)
            self.pos_enc = PositionalEncoding(d_model, dropout_emb, pe_type)
            self.layers = nn.ModuleList(
                [TransformerDecoderBlock(d_model, d_ff, attn_type, attn_n_heads,
                                         dropout, dropout_att, layer_norm_eps)
                 for _ in range(n_layers)])
            self.norm_top = nn.LayerNorm(d_model, eps=layer_norm_eps)

            if adaptive_softmax:
                self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
                    d_model, vocab,
                    cutoffs=[round(self.vocab / 15), 3 * round(self.vocab / 15)],
                    # cutoffs=[self.vocab // 25, 3 * self.vocab // 5],
                    div_value=4.0)
                self.output = None
            else:
                self.adaptive_softmax = None
                self.output = LinearND(d_model, vocab)

                # Optionally tie weights as in:
                # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
                # https://arxiv.org/abs/1608.05859
                # and
                # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
                # https://arxiv.org/abs/1611.01462
                if tie_embedding:
                    self.output.fc.weight = self.embed.embed.weight

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with xavier_uniform style."""
        logger = logging.getLogger('training')
        logger.info('===== Initialize %s =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.constant_(p, val=0)  # bias
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0))
            elif p.dim() == 2:
                if 'embed' in n:
                    nn.init.normal_(p, mean=0, std=self.d_model**-0.5)
                    logger.info('Initialize %s with %s / %.3f' % (n, 'normal', self.d_model**-0.5))
                else:
                    nn.init.xavier_uniform_(p, gain=1.0)
                    logger.info('Initialize %s with %s' % (n, 'xavier_uniform'))
            else:
                raise ValueError

    def forward(self, eouts, elens, ys, task='all', ys_hist=[], teacher_dist=None):
        """Forward computation.

        Args:
            eouts (FloatTensor): `[B, T, d_model]`
            elens (IntTensor): `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
            task (str): all or ys or ys_sub*
            ys_hist (list): dummy (not used)
        Returns:
            loss (FloatTensor): `[1]`
            observation (dict):

        """
        observation = {'loss': None,
                       'loss_att': None, 'loss_ctc': None, 'loss_lmobj': None,
                       'acc_att': None, 'acc_lmobj': None,
                       'ppl_att': None, 'ppl_lmobj': None}
        # NOTE: lmobj is not supported now
        loss = eouts.new_zeros((1,))

        # CTC loss
        if self.ctc_weight > 0 and (not self.mtl_per_batch or (self.mtl_per_batch and 'ctc' in task)):
            loss_ctc = self.ctc(eouts, elens, ys)
            observation['loss_ctc'] = loss_ctc.item()
            if self.mtl_per_batch:
                loss += loss_ctc
            else:
                loss += loss_ctc * self.ctc_weight

        # XE loss
        if self.global_weight - self.ctc_weight > 0 and 'ctc' not in task and 'lmobj' not in task:
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

    def forward_att(self, eouts, elens, ys, return_logits=False):
        """Compute XE loss for the sequence-to-sequence model.

        Args:
            eouts (FloatTensor): `[B, T, d_model]`
            elens (IntTensor): `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
            return_logits (bool): return logits for knowledge distillation
        Returns:
            loss (FloatTensor): `[1]`
            acc (float):
            ppl (float):

        """
        bs = eouts.size(0)

        # Append <sos> and <eos>
        eos = eouts.new_zeros(1).fill_(self.eos).long()
        ys = [np2tensor(np.fromiter(y[::-1] if self.bwd else y, dtype=np.int64), self.device_id) for y in ys]
        ylens = np2tensor(np.fromiter([y.size(0) + 1 for y in ys], dtype=np.int32))  # +1 for <eos>
        ys_in_pad = pad_list([torch.cat([eos, y], dim=0) for y in ys], self.pad)
        ys_out_pad = pad_list([torch.cat([y, eos], dim=0) for y in ys], self.pad)

        ys_emb = self.embed(ys_in_pad) * (self.d_model ** 0.5)
        ys_emb = self.pos_enc(ys_emb)
        for l in range(self.n_layers):
            ys_emb, yy_aws, xy_aws = self.layers[l](ys_emb, ylens, eouts, elens)
            if not self.training:
                setattr(self, 'yy_aws_layer%d' % l, tensor2np(yy_aws))
                setattr(self, 'xy_aws_layer%d' % l, tensor2np(xy_aws))
        logits = self.norm_top(ys_emb)
        if self.adaptive_softmax is None:
            logits = self.output(logits)
        if return_logits:
            return logits

        # Compute XE sequence loss
        if self.adaptive_softmax is None:
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
        ppl = min(np.exp(loss.item()), np.inf)

        return loss, acc, ppl

    def greedy(self, eouts, elens, max_len_ratio,
               exclude_eos=False, idx2token=None, refs_id=None,
               speakers=None, oracle=False):
        """Greedy decoding in the inference stage (used only for evaluation during training).

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            elens (IntTensor): `[B]`
            max_len_ratio (int): maximum sequence length of tokens
            exclude_eos (bool):
            idx2token ():
            refs_id (list):
            speakers (list):
            oracle (bool):
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw (list): A list of length `[B]`, which contains arrays of size `[L, T]`

        """
        bs, max_xlen, d_model = eouts.size()

        # Start from <sos> (<eos> in case of the backward decoder)
        ys = eouts.new_zeros(bs, 1).fill_(self.eos).long()

        best_hyps_tmp = []
        ylens = torch.zeros(bs).int()
        yy_aws_tmp = [None] * bs
        xy_aws_tmp = [None] * bs
        eos_flags = [False] * bs
        for t in range(int(np.floor(max_xlen * max_len_ratio)) + 1):
            out = self.embed(ys) * (self.d_model ** 0.5)
            out = self.pos_enc(out)
            for l in range(self.n_layers):
                out, yy_aws, xy_aws = self.layers[l](out, ylens + 1, eouts, elens)
            out = self.norm_top(out)
            logits_t = self.output(out)

            # Pick up 1-best
            y = logits_t.detach().argmax(-1)[:, -1:]
            best_hyps_tmp += [y]

            # Count lengths of hypotheses
            for b in range(bs):
                if not eos_flags[b]:
                    if y[b].item() == self.eos:
                        eos_flags[b] = True
                        yy_aws_tmp[b] = yy_aws[b:b + 1]  # TODO: fix this
                        xy_aws_tmp[b] = xy_aws[b:b + 1]
                    ylens[b] += 1
                    # NOTE: include <eos>

            # Break if <eos> is outputed in all mini-bs
            if sum(eos_flags) == bs:
                break

            ys = torch.cat([ys, y], dim=-1)

        # Concatenate in L dimension
        best_hyps_tmp = torch.cat(best_hyps_tmp, dim=1)
        # xy_aws_tmp = torch.stack(xy_aws_tmp, dim=0)

        # Convert to numpy
        best_hyps_tmp = tensor2np(best_hyps_tmp)
        # xy_aws_tmp = tensor2np(xy_aws_tmp)

        # if self.score.attn_n_heads > 1:
        #     xy_aws_tmp = xy_aws_tmp[:, :, :, 0]
        #     # TODO(hirofumi): fix for MHA

        # Truncate by the first <eos> (<sos> in case of the backward decoder)
        if self.bwd:
            # Reverse the order
            best_hyps = [best_hyps_tmp[b, :ylens[b]][::-1] for b in range(bs)]
            # aws = [xy_aws_tmp[b, :ylens[b]][::-1] for b in range(bs)]
        else:
            best_hyps = [best_hyps_tmp[b, :ylens[b]] for b in range(bs)]
            # aws = [xy_aws_tmp[b, :ylens[b]] for b in range(bs)]

        # Exclude <eos> (<sos> in case of the backward decoder)
        if exclude_eos:
            if self.bwd:
                best_hyps = [best_hyps[b][1:] if eos_flags[b]
                             else best_hyps[b] for b in range(bs)]
            else:
                best_hyps = [best_hyps[b][:-1] if eos_flags[b]
                             else best_hyps[b] for b in range(bs)]

        # return best_hyps, aws
        return best_hyps, None

    def _plot_attention(self, save_path, n_cols=2):
        """Plot attention for each head in all layers."""
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator

        for attn in ['yy', 'xy']:
            _save_path = mkdir_join(save_path, 'dec_%s_att_weights' % attn)

            # Clean directory
            if _save_path is not None and os.path.isdir(_save_path):
                shutil.rmtree(_save_path)
                os.mkdir(_save_path)

            for l in range(self.n_layers):
                if not hasattr(self, '%s_aws_layer%d' % (attn, l)):
                    continue

                aws = getattr(self, '%s_aws_layer%d' % (attn, l))

                plt.clf()
                fig, axes = plt.subplots(self.n_heads // n_cols, n_cols, figsize=(20, 8))
                for h in range(self.n_heads):
                    if self.n_heads > n_cols:
                        ax = axes[h // n_cols, h % n_cols]
                    else:
                        ax = axes[h]
                    ax.imshow(aws[-1, h, :, :], aspect="auto")
                    ax.grid(False)
                    ax.set_xlabel("Input (head%d)" % h)
                    ax.set_ylabel("Output (head%d)" % h)
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

                fig.tight_layout()
                fig.savefig(os.path.join(_save_path, 'layer%d.png' % (l)), dvi=500)
                plt.close()
