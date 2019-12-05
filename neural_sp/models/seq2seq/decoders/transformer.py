#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer decoder (including CTC loss calculation)."""

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
from neural_sp.models.modules.transformer import PositionalEncoding
from neural_sp.models.modules.transformer import TransformerDecoderBlock
from neural_sp.models.seq2seq.decoders.ctc import CTC
# from neural_sp.models.seq2seq.decoders.ctc import CTCPrefixScore
from neural_sp.models.seq2seq.decoders.decoder_base import DecoderBase
from neural_sp.models.torch_utils import compute_accuracy
from neural_sp.models.torch_utils import make_pad_mask
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
        special_symbols (dict):
            eos (int): index for <eos> (shared with <sos>)
            unk (int): index for <unk>
            pad (int): index for <pad>
            blank (int): index for <blank>
        blank (int): index for <blank>
        enc_n_units (int): number of units of the encoder outputs
        attn_type (str): type of attention mechanism
        attn_n_heads (int): number of attention heads
        d_model (int): size of the model
        d_ff (int): size of the inner FF layer
        n_layers (int): number of self-attention layers
        vocab (int): number of nodes in softmax layer
        tie_embedding (bool): tie parameters of the embedding and output layers
        pe_type (str): type of positional encoding
        layer_norm_eps (float): epsilon value for layer normalization
        dropout (float): dropout probability for linear layers
        dropout_emb (float): dropout probability for the embedding layer
        dropout_att (float): dropout probability for attention distributions
        lsm_prob (float): label smoothing probability
        ctc_weight (float):
        ctc_lsm_prob (float): label smoothing probability for CTC
        ctc_fc_list (list):
        backward (bool): decode in the backward order
        global_weight (float):
        mtl_per_batch (bool):

    """

    def __init__(self,
                 special_symbols,
                 enc_n_units,
                 attn_type,
                 attn_n_heads,
                 d_model,
                 d_ff,
                 n_layers,
                 vocab,
                 tie_embedding=False,
                 pe_type='add',
                 layer_norm_eps=1e-12,
                 dropout=0.,
                 dropout_emb=0.,
                 dropout_att=0.,
                 lsm_prob=0.,
                 ctc_weight=0.,
                 ctc_lsm_prob=0.,
                 ctc_fc_list=[],
                 backward=False,
                 global_weight=1.0,
                 mtl_per_batch=False):

        super(TransformerDecoder, self).__init__()
        logger = logging.getLogger('training')

        self.eos = special_symbols['eos']
        self.unk = special_symbols['unk']
        self.pad = special_symbols['pad']
        self.blank = special_symbols['blank']
        self.vocab = vocab
        self.enc_n_units = enc_n_units
        self.d_model = d_model
        self.n_layers = n_layers
        self.attn_n_heads = attn_n_heads
        self.pe_type = pe_type
        self.lsm_prob = lsm_prob
        self.ctc_weight = ctc_weight
        self.bwd = backward
        self.global_weight = global_weight
        self.mtl_per_batch = mtl_per_batch

        if ctc_weight > 0:
            self.ctc = CTC(eos=self.eos,
                           blank=self.blank,
                           enc_n_units=enc_n_units,
                           vocab=vocab,
                           dropout=dropout,
                           lsm_prob=ctc_lsm_prob,
                           fc_list=ctc_fc_list,
                           param_init=0.1)

        if ctc_weight < global_weight:
            self.embed = nn.Embedding(vocab, d_model, padding_idx=self.pad)
            self.pos_enc = PositionalEncoding(d_model, dropout_emb, pe_type)
            self.layers = nn.ModuleList(
                [TransformerDecoderBlock(d_model, d_ff, attn_type, attn_n_heads,
                                         dropout, dropout_att, layer_norm_eps)
                 for _ in range(n_layers)])
            self.norm_out = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.output = nn.Linear(d_model, vocab)
            if tie_embedding:
                self.output.fc.weight = self.embed.weight

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with xavier_uniform style."""
        logger = logging.getLogger('training')
        logger.info('===== Initialize %s =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.constant_(p, val=0.)  # bias
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.))
            elif p.dim() == 2:
                if 'embed' in n:
                    nn.init.normal_(p, mean=0., std=self.d_model**-0.5)
                    logger.info('Initialize %s with %s / %.3f' % (n, 'normal', self.d_model**-0.5))
                else:
                    nn.init.xavier_uniform_(p, gain=1.0)
                    logger.info('Initialize %s with %s' % (n, 'xavier_uniform'))
            else:
                raise ValueError

    def forward(self, eouts, elens, ys, task='all', ys_hist=[], teacher_logits=None):
        """Forward computation.

        Args:
            eouts (FloatTensor): `[B, T, d_model]`
            elens (IntTensor): `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
            task (str): all/ys/ys_sub*
            ys_hist (list): dummy (not used)
            teacher_logits (FloatTensor): `[B, L, vocab]`
        Returns:
            loss (FloatTensor): `[1]`
            observation (dict):

        """
        observation = {'loss': None, 'loss_att': None, 'loss_ctc': None,
                       'acc_att': None, 'ppl_att': None}
        loss = eouts.new_zeros((1,))

        # CTC loss
        if self.ctc_weight > 0 and (task == 'all' or 'ctc' in task):
            loss_ctc = self.ctc(eouts, elens, ys)
            observation['loss_ctc'] = loss_ctc.item()
            if self.mtl_per_batch:
                loss += loss_ctc
            else:
                loss += loss_ctc * self.ctc_weight

        # XE loss
        if self.global_weight - self.ctc_weight > 0 and (task == 'all' or ('ctc' not in task)):
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

    def forward_att(self, eouts, elens, ys, ys_hist=[],
                    return_logits=False, teacher_logits=None):
        """Compute XE loss for the Transformer model.

        Args:
            eouts (FloatTensor): `[B, T, d_model]`
            elens (IntTensor): `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
            ys_hist (list):
            return_logits (bool): return logits for knowledge distillation
            teacher_logits (FloatTensor): `[B, L, vocab]`
        Returns:
            loss (FloatTensor): `[1]`
            acc (float): accuracy for token prediction
            ppl (float): perplexity

        """
        bs = eouts.size(0)

        # Append <sos> and <eos>
        ys_in_pad, ys_out_pad, ylens = self.append_sos_eos(ys, self.bwd)

        # Create the self-attention mask
        bs, ytime = ys_in_pad.size()[:2]
        tgt_mask = make_pad_mask(ylens, self.device_id).unsqueeze(1).repeat([1, ytime, 1])
        subsequent_mask = tgt_mask.new_ones(ytime, ytime).byte()
        subsequent_mask = torch.tril(subsequent_mask, out=subsequent_mask).unsqueeze(0)
        tgt_mask = tgt_mask & subsequent_mask

        # Create the source-target mask
        src_mask = make_pad_mask(elens, self.device_id).unsqueeze(1).repeat([1, ytime, 1])

        ys_emb = self.pos_enc(self.embed(ys_in_pad))
        for l in range(self.n_layers):
            ys_emb, yy_aws, xy_aws = self.layers[l](ys_emb, tgt_mask, eouts, src_mask)
            if not self.training:
                setattr(self, 'yy_aws_layer%d' % l, tensor2np(yy_aws))
                setattr(self, 'xy_aws_layer%d' % l, tensor2np(xy_aws))
        ys_emb = self.norm_out(ys_emb)
        logits = self.output(ys_emb)

        # for knowledge distillation
        if return_logits:
            return logits

        # Compute XE sequence loss
        if self.lsm_prob > 0 and self.training:
            # Label smoothing
            loss = cross_entropy_lsm(logits.view((-1, logits.size(2))), ys_out_pad.view(-1),
                                     self.lsm_prob, self.pad)
        else:
            loss = F.cross_entropy(logits.view((-1, logits.size(2))), ys_out_pad.view(-1),
                                   ignore_index=self.pad, size_average=True)

        # Compute token-level accuracy in teacher-forcing
        acc = compute_accuracy(logits, ys_out_pad, self.pad)
        # ppl = min(np.exp(loss.item()), np.inf)
        ppl = np.exp(loss.item())

        # scale loss for CTC
        loss *= ylens.float().mean()

        return loss, acc, ppl

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
                if hasattr(self, '%s_aws_layer%d' % (attn, l)):
                    aws = getattr(self, '%s_aws_layer%d' % (attn, l))

                    plt.clf()
                    fig, axes = plt.subplots(max(1, self.attn_n_heads // n_cols), n_cols,
                                             figsize=(20, 8), squeeze=False)
                    for h in range(self.attn_n_heads):
                        ax = axes[h // n_cols, h % n_cols]
                        ax.imshow(aws[-1, h, :, :], aspect="auto")
                        ax.grid(False)
                        ax.set_xlabel("Input (head%d)" % h)
                        ax.set_ylabel("Output (head%d)" % h)
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

                    fig.tight_layout()
                    fig.savefig(os.path.join(_save_path, 'layer%d.png' % (l)), dvi=500)
                    plt.close()

    def greedy(self, eouts, elens, max_len_ratio, idx2token,
               exclude_eos=False, oracle=False,
               refs_id=None, utt_ids=None, speakers=None):
        """Greedy decoding in the inference stage (used only for evaluation during training).

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
            aw (list): A list of length `[B]`, which contains arrays of size `[L, T]`

        """
        logger = logging.getLogger("decoding")
        bs, xtime = eouts.size()[:2]

        # Start from <sos> (<eos> in case of the backward decoder)
        y_seq = eouts.new_zeros(bs, 1).fill_(self.eos).long()

        # Append <sos> and <eos>
        ys_in_pad, ys_out_pad, ylens = self.append_sos_eos(refs_id, self.bwd)

        hyps_batch = []
        ylens = torch.zeros(bs).int()
        eos_flags = [False] * bs
        if oracle:
            assert refs_id is not None
            ytime = max([len(refs_id[b]) for b in range(bs)]) + 1
        else:
            ytime = int(math.floor(xtime * max_len_ratio)) + 1
        for t in range(ytime):
            subsequent_mask = eouts.new_ones(t + 1, t + 1).byte()
            subsequent_mask = torch.tril(subsequent_mask, out=subsequent_mask).unsqueeze(0)

            dout = self.pos_enc(self.embed(y_seq))
            for l in range(self.n_layers):
                dout, _, xy_aws = self.layers[l](dout, subsequent_mask, eouts, None)
            dout = self.norm_out(dout)

            # Pick up 1-best
            y = self.output(dout).argmax(-1)[:, -1:]
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
                y = eouts.new_zeros(bs, 1).long()
                for b in range(bs):
                    y[b, 0] = refs_id[b][t]
            y_seq = torch.cat([y_seq, y], dim=-1)

        # Concatenate in L dimension
        hyps_batch = tensor2np(torch.cat(hyps_batch, dim=1))
        xy_aws = tensor2np(xy_aws.transpose(1, 2).transpose(2, 3))

        # Truncate by the first <eos> (<sos> in case of the backward decoder)
        if self.bwd:
            # Reverse the order
            hyps = [hyps_batch[b, :ylens[b]][::-1] for b in range(bs)]
            aws = [xy_aws[b, :, :ylens[b]][::-1] for b in range(bs)]
        else:
            hyps = [hyps_batch[b, :ylens[b]] for b in range(bs)]
            aws = [xy_aws[b, :, :ylens[b]] for b in range(bs)]

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

        return hyps, aws
