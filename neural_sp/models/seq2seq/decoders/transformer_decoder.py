#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer decoder (including CTC loss calculation)."""

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
try:
    import warpctc_pytorch
except:
    raise ImportError('Install warpctc_pytorch.')

from neural_sp.models.criterion import cross_entropy_lsm
from neural_sp.models.criterion import focal_loss
from neural_sp.models.criterion import kldiv_lsm_ctc
from neural_sp.models.model_utils import Embedding
from neural_sp.models.model_utils import LinearND
from neural_sp.models.model_utils import SublayerConnection
from neural_sp.models.model_utils import PositionwiseFeedForward
from neural_sp.models.model_utils import PositionalEncoding
from neural_sp.models.seq2seq.decoders.multihead_attention import TransformerMultiheadAttentionMechanism
from neural_sp.models.seq2seq.decoders.ctc_beam_search_decoder import BeamSearchDecoder
from neural_sp.models.seq2seq.decoders.ctc_beam_search_decoder import CTCPrefixScore
from neural_sp.models.seq2seq.decoders.ctc_greedy_decoder import GreedyDecoder
from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list
from neural_sp.models.torch_utils import tensor2np

random.seed(1)

logger = logging.getLogger("decoding")


class TransformerDecoder(nn.Module):
    """Transformer decoder.

    Args:
        sos (int): index for <sos>
        eos (int): index for <eos>
        pad (int): index for <pad>
        blank (int): index for <blank>
        enc_nunits (int):
        attn_type (str):
        attn_nheads (int): number of attention heads
        nlayers (int): number of encoder layers.
        d_model (int): size of the model
        d_ff (int): size of the inner FF layer
        tie_embedding (bool):
        vocab (int): number of nodes in softmax layer
        dropout (float): dropout probabilities for linear layers
        dropout_emb (float): probability to drop nodes of the embedding layer
        dropout_att (float): dropout probabilities for attention distributions
        lsm_prob (float): label smoothing probability
        ctc_weight (float):
        ctc_fc_list (list):
        backward (bool): decode in the backward order
        global_weight (float):
        mtl_per_batch (bool):

    """

    def __init__(self,
                 sos,
                 eos,
                 pad,
                 blank,
                 enc_nunits,
                 attn_type,
                 attn_nheads,
                 nlayers,
                 d_model,
                 d_ff,
                 tie_embedding,
                 vocab,
                 dropout,
                 dropout_emb,
                 dropout_att,
                 lsm_prob,
                 ctc_weight,
                 ctc_fc_list,
                 backward,
                 global_weight,
                 mtl_per_batch):

        super(TransformerDecoder, self).__init__()

        self.sos = sos
        self.eos = eos
        self.pad = pad
        self.blank = blank
        self.enc_nunits = enc_nunits
        self.nlayers = nlayers
        self.lsm_prob = lsm_prob
        self.ctc_weight = ctc_weight
        self.ctc_fc_list = ctc_fc_list
        self.backward = backward
        self.global_weight = global_weight
        self.mtl_per_batch = mtl_per_batch

        if ctc_weight > 0:
            # Fully-connected layers for CTC
            if len(ctc_fc_list) > 0:
                fc_layers = OrderedDict()
                for i in range(len(ctc_fc_list)):
                    input_dim = enc_nunits if i == 0 else ctc_fc_list[i - 1]
                    fc_layers['fc' + str(i)] = LinearND(input_dim, ctc_fc_list[i], dropout=dropout)
                fc_layers['fc' + str(len(ctc_fc_list))] = LinearND(ctc_fc_list[-1], vocab, dropout=0)
                self.output_ctc = nn.Sequential(fc_layers)
            else:
                self.output_ctc = LinearND(enc_nunits, vocab)
            self.decode_ctc_greedy = GreedyDecoder(blank=blank)
            self.decode_ctc_beam = BeamSearchDecoder(blank=blank)
            self.warpctc_loss = warpctc_pytorch.CTCLoss(size_average=True)

        assert global_weight > ctc_weight

        self.layers = nn.ModuleList(
            [TransformerDecoderBlock(d_model, d_ff,
                                     attn_type, attn_nheads,
                                     dropout, dropout_att)
             for _ in range(nlayers)])

        self.embed = Embedding(vocab, d_model,
                               dropout=dropout_emb,
                               ignore_index=pad, scale=True)
        self.output = LinearND(d_model, vocab)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_embedding:
            self.output.fc.weight.data = self.embed.embed.weight.data

    @property
    def device_id(self):
        return torch.cuda.device_of(next(self.parameters()).data).idx

    def forward(self, eouts, elens, ys, task='all'):
        """Forward computation.

        Args:
            eouts (FloatTensor): `[B, T, d_model]`
            elens (list): A list of length `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
            task (str): all or ys or ys_sub*
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
            loss_ctc = self.forward_ctc(eouts, elens, ys)
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

    def forward_ctc(self, eouts, elens, ys):
        """Compute CTC loss.

        Args:
            eouts (FloatTensor): `[B, T, d_model]`
            elens (list): A list of length `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
        Returns:
            loss (FloatTensor): `[1]`

        """
        logits = self.output_ctc(eouts)

        # Compute the auxiliary CTC loss
        elens_ctc = np2tensor(np.fromiter(elens, dtype=np.int32), -1).int()
        ys_ctc = [np2tensor(np.fromiter(y, dtype=np.int64)).long() for y in ys]  # always fwd
        ylens = np2tensor(np.fromiter([y.size(0) for y in ys_ctc], dtype=np.int32), -1).int()
        ys_ctc = torch.cat(ys_ctc, dim=0).int()
        # NOTE: Concatenate all elements in ys for warpctc_pytorch
        # NOTE: do not copy to GPUs here

        # Compute CTC loss
        loss = self.warpctc_loss(logits.transpose(0, 1).cpu(),  # time-major
                                 ys_ctc, elens_ctc, ylens)
        # NOTE: ctc loss has already been normalized by bs
        # NOTE: index 0 is reserved for blank in warpctc_pytorch

        if self.device_id >= 0:
            loss = loss.cuda(self.device_id)

        # Label smoothing for CTC
        if self.lsm_prob > 0 and self.ctc_weight == 1:
            loss = loss * (1 - self.lsm_prob) + kldiv_lsm_ctc(
                logits, ylens=elens,
                lsm_prob=self.lsm_prob, size_average=True) * self.lsm_prob

        return loss

    def forward_att(self, eouts, elens, ys):
        """Compute XE loss for the sequence-to-sequence model.

        Args:
            eouts (FloatTensor): `[B, T, d_model]`
            elens (list): A list of length `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
        Returns:
            loss (FloatTensor): `[1]`
            acc (float):
            ppl (float):

        """
        # Append <sos> and <eos>
        sos = eouts.new_zeros((1,)).fill_(self.sos).long()
        eos = eouts.new_zeros((1,)).fill_(self.eos).long()
        if self.backward:
            ys = [np2tensor(np.fromiter(y[::-1], dtype=np.int64), self.device_id).long() for y in ys]
            ys_in = [torch.cat([eos, y], dim=0) for y in ys]
            ys_out = [torch.cat([y, sos], dim=0) for y in ys]
        else:
            ys = [np2tensor(np.fromiter(y, dtype=np.int64), self.device_id).long() for y in ys]
            ys_in = [torch.cat([sos, y], dim=0) for y in ys]
            ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        ys_in_pad = pad_list(ys_in, self.pad)
        ys_out_pad = pad_list(ys_out, -1)

        # Pre-computation of embedding
        ys_emb = self.embed(ys_in_pad)

        # Make source-target attention mask: `[B, L(query), T(key)]`
        bs, max_xlen = eouts.size()[:2]
        y_len_max = ys_in_pad.size(1)
        yx_mask = (ys_in_pad != self.pad).unsqueeze(-1).expand(bs, y_len_max, max_xlen)
        for b in range(bs):
            if elens[b] < max_xlen:
                yx_mask[b, :, elens[b]:] = 0

        # Make target-side self-attention mask (hide future tokens): `[B, L(query), L(key)]`
        yy_mask = (ys_in_pad != self.pad).unsqueeze(-2).expand(bs, y_len_max, y_len_max)
        history_mask = torch.triu(torch.ones((y_len_max, y_len_max),
                                             device=self.device_id, dtype=torch.uint8), diagonal=1)
        history_mask = history_mask.unsqueeze(0).expand(bs, -1, -1) == 0
        yy_mask = yy_mask & history_mask

        for l in range(self.nlayers):
            ys_emb, yy_aw, xy_aw = self.layers[l](eouts, ys_emb, yx_mask, yy_mask)

        # Compute XE sequence loss
        logits = self.output(ys_emb)
        if self.lsm_prob > 0:
            # Label smoothing
            ylens = [y.size(0) for y in ys_out]
            loss = cross_entropy_lsm(logits, ys=ys_out_pad, ylens=ylens,
                                     lsm_prob=self.lsm_prob, size_average=True)
        else:
            loss = F.cross_entropy(input=logits.view((-1, logits.size(2))),
                                   target=ys_out_pad.view(-1),  # long
                                   ignore_index=-1, size_average=False) / bs
        # ppl = math.exp(loss.item())
        ppl = np.exp(loss.item())

        # Compute token-level accuracy in teacher-forcing
        pad_pred = logits.view(ys_out_pad.size(0), ys_out_pad.size(1), logits.size(-1)).argmax(2)
        mask = ys_out_pad != -1
        numerator = torch.sum(pad_pred.masked_select(mask) == ys_out_pad.masked_select(mask))
        denominator = torch.sum(mask)
        acc = float(numerator) * 100 / float(denominator)

        return loss, acc, ppl

    def greedy(self, eouts, elens, max_len_ratio, exclude_eos=False):
        """Greedy decoding in the inference stage.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            elens (list): A list of length `[B]`
            max_len_ratio (int): maximum sequence length of tokens
            exclude_eos (bool):
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw (list): A list of length `[B]`, which contains arrays of size `[L, T]`

        """
        bs, max_xlen, d_model = eouts.size()

        if self.backward:
            sos, eos = self.eos, self.sos
        else:
            sos, eos = self.sos, self.eos

        # Start from <sos> (<eos> in case of the backward decoder)
        ys = eouts.new_zeros(bs, 1).fill_(sos).long()

        yy_mask = None

        best_hyps_tmp = []
        ylens = np.zeros((bs,), dtype=np.int32)
        yy_aws_tmp = [None] * bs
        xy_aws_tmp = [None] * bs
        eos_flags = [False] * bs
        for t in range(int(math.floor(max_xlen * max_len_ratio)) + 1):
            # Make source-target attention mask
            yx_mask = eouts.new_ones(bs, t + 1, max_xlen)
            for b in range(bs):
                if elens[b] < max_xlen:
                    yx_mask[b, :, elens[b]:] = 0

            out = self.embed(ys)
            for l in range(self.nlayers):
                out, yy_aw, xy_aw = self.layers[l](eouts, out, yx_mask, yy_mask)
                # xy_aw: `[B, head, T, L]`
            logits_t = self.output(out)

            # Pick up 1-best
            y = np.argmax(logits_t.detach(), axis=2).cuda(self.device_id)[:, -1:]
            best_hyps_tmp += [y]

            # Count lengths of hypotheses
            for b in range(bs):
                if not eos_flags[b]:
                    if y[b].item() == eos:
                        eos_flags[b] = True
                        yy_aws_tmp[b] = yy_aw[b:b + 1]  # TODO: fix this
                        xy_aws_tmp[b] = xy_aw[b:b + 1]
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

        # if self.score.attn_nheads > 1:
        #     xy_aws_tmp = xy_aws_tmp[:, :, :, 0]
        #     # TODO(hirofumi): fix for MHA

        # Truncate by the first <eos> (<sos> in case of the backward decoder)
        if self.backward:
            # Reverse the order
            best_hyps = [best_hyps_tmp[b, :ylens[b]][::-1] for b in range(bs)]
            # aws = [xy_aws_tmp[b, :ylens[b]][::-1] for b in range(bs)]
        else:
            best_hyps = [best_hyps_tmp[b, :ylens[b]] for b in range(bs)]
            # aws = [xy_aws_tmp[b, :ylens[b]] for b in range(bs)]

        # Exclude <eos> (<sos> in case of the backward decoder)
        if exclude_eos:
            if self.backward:
                best_hyps = [best_hyps[b][1:] if eos_flags[b]
                             else best_hyps[b] for b in range(bs)]
            else:
                best_hyps = [best_hyps[b][:-1] if eos_flags[b]
                             else best_hyps[b] for b in range(bs)]

        # return best_hyps, aws
        return best_hyps, None


class TransformerDecoderBlock(nn.Module):
    """A single layer of the transformer decoder.

        Args:
            d_model (int): dimension of keys/values/queries in
                           TransformerMultiheadAttentionMechanism, also the input size of
                           the first-layer of the PositionwiseFeedForward
            d_ff (int): second-layer of the PositionwiseFeedForward
            attn_type (str):
            attn_nheads (int): number of heads for multi-head attention
            dropout (float): dropout probabilities for linear layers
            dropout_att (float): dropout probabilities for attention probabilities
            attn_type (string): type of self-attention, scaled_dot_product or average

    """

    def __init__(self, d_model, d_ff, attn_type, attn_nheads,
                 dropout, dropout_att):
        super(TransformerDecoderBlock, self).__init__()

        self.attn_type = attn_type

        # self-attention
        if attn_type == "scaled_dot_product":
            self.self_attn = TransformerMultiheadAttentionMechanism(attn_nheads, d_model, dropout_att)
        elif attn_type == "average":
            raise NotImplementedError()
            # self.self_attn = AverageAttention(d_model, dropout, layer_norm=True)
        else:
            raise NotImplementedError(attn_type)
        self.add_norm1 = SublayerConnection(d_model, dropout, layer_norm=True)

        # attention for encoder stacks
        self.enc_attn = TransformerMultiheadAttentionMechanism(attn_nheads, d_model, dropout_att)
        self.add_norm2 = SublayerConnection(d_model, dropout, layer_norm=True)

        # feed-forward
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.add_norm3 = SublayerConnection(d_model, dropout, layer_norm=True)

    def forward(self, x, y, yx_mask, yy_mask):
        """Transformer decoder layer definition.

        Args:
            x (FloatTensor): encoder outputs. `[B, T, d_model]`
            y (FloatTensor): `[B, L, d_model]`
            yx_mask (LongTensor): mask for source-target connection. `[B, L(query), T(key)]`
            yy_mask (LongTensor): mask for target-target connection. `[B, L(query), L(key)]`
                0: place to pad with -1024
                1: otherwise
        Returns:
            y (FloatTensor): `[B, L, d_model]`
            yy_aw (FloatTensor)`[B, L, L]`
            xy_aw (FloatTensor): `[B, L, T]`

        """
        # self-attention
        if self.attn_type == "scaled_dot_product":
            y, yy_aw = self.add_norm1(y, lambda y: self.self_attn(y, y, y, yy_mask))  # key/value/query
        elif self.attn_type == "average":
            raise NotImplementedError()

        # attention for encoder stacks
        y, xy_aw = self.add_norm2(y, lambda y: self.enc_attn(x, x, y, yx_mask))  # key/value/query

        # position-wise feed-forward
        y = self.add_norm3(y, lambda y: self.feed_forward(y))  # key/value/query

        return y, yy_aw, xy_aw
