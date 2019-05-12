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
from neural_sp.models.modules.embedding import Embedding
from neural_sp.models.modules.linear import LinearND
from neural_sp.models.modules.transformer import SublayerConnection
from neural_sp.models.modules.transformer import PositionwiseFeedForward
from neural_sp.models.modules.transformer import PositionalEncoding
from neural_sp.models.seq2seq.decoders.multihead_attention import MultiheadAttentionMechanism
from neural_sp.models.seq2seq.decoders.ctc_beam_search import BeamSearchDecoder
from neural_sp.models.seq2seq.decoders.ctc_beam_search import CTCPrefixScore
from neural_sp.models.seq2seq.decoders.ctc_greedy import GreedyDecoder
from neural_sp.models.torch_utils import compute_accuracy
from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list
from neural_sp.models.torch_utils import tensor2np

random.seed(1)

logger = logging.getLogger("decoding")


class TransformerDecoder(nn.Module):
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
        pe_type (str): concat or add or learn or False
        tie_embedding (bool):
        vocab (int): number of nodes in softmax layer
        dropout (float): dropout probabilities for linear layers
        dropout_emb (float): probability to drop nodes of the embedding layer
        dropout_att (float): dropout probabilities for attention distributions
        lsm_prob (float): label smoothing probability
        layer_norm_eps (float):
        ctc_weight (float):
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
                 pe_type,
                 tie_embedding,
                 vocab,
                 dropout=0.0,
                 dropout_emb=0.0,
                 dropout_att=0.0,
                 lsm_prob=0.0,
                 layer_norm_eps=1e-6,
                 ctc_weight=0.0,
                 ctc_fc_list=[],
                 backward=False,
                 global_weight=1.0,
                 mtl_per_batch=False,
                 adaptive_softmax=False):

        super(TransformerDecoder, self).__init__()

        self.eos = eos
        self.unk = unk
        self.pad = pad
        self.blank = blank
        self.enc_n_units = enc_n_units
        self.d_model = d_model
        self.n_layers = n_layers
        self.pe_type = pe_type
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
                    input_dim = d_model if i == 0 else ctc_fc_list[i - 1]
                    fc_layers['fc' + str(i)] = LinearND(input_dim, ctc_fc_list[i], dropout=dropout)
                fc_layers['fc' + str(len(ctc_fc_list))] = LinearND(ctc_fc_list[-1], vocab, dropout=0)
                self.output_ctc = nn.Sequential(fc_layers)
            else:
                self.output_ctc = LinearND(d_model, vocab)
            self.decode_ctc_greedy = GreedyDecoder(blank=blank)
            self.decode_ctc_beam = BeamSearchDecoder(blank=blank)
            self.warpctc_loss = warpctc_pytorch.CTCLoss(size_average=True)

        if ctc_weight < global_weight:
            self.layers = nn.ModuleList(
                [TransformerDecoderBlock(d_model, d_ff, attn_type, attn_n_heads,
                                         dropout, dropout_att, layer_norm_eps)
                 for _ in range(n_layers)])

            self.embed = Embedding(vocab, d_model,
                                   dropout=0,  # NOTE: do not apply dropout here
                                   ignore_index=pad)
            if pe_type:
                self.pos_emb_out = PositionalEncoding(d_model, dropout_emb, pe_type)

            if adaptive_softmax:
                self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
                    d_model, vocab,
                    cutoffs=[round(self.vocab / 15), 3 * round(self.vocab / 15)],
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

            self.layer_norm_top = nn.LayerNorm(d_model, eps=layer_norm_eps)

    @property
    def device_id(self):
        return torch.cuda.device_of(next(self.parameters()).data).idx

    def forward(self, eouts, elens, ys, task='all', ys_hist=[]):
        """Forward computation.

        Args:
            eouts (FloatTensor): `[B, T, d_model]`
            elens (list): A list of length `[B]`
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
        loss = self.warpctc_loss(logits.transpose(1, 0).cpu(),  # time-major
                                 ys_ctc, elens_ctc, ylens)
        # NOTE: ctc loss has already been normalized by bs
        # NOTE: index 0 is reserved for blank in warpctc_pytorch

        if self.device_id >= 0:
            loss = loss.cuda(self.device_id)

        # Label smoothing for CTC
        if self.lsm_prob > 0 and self.ctc_weight == 1:
            loss = loss * (1 - self.lsm_prob) + kldiv_lsm_ctc(logits,
                                                              ylens=elens,
                                                              size_average=True) * self.lsm_prob

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
        bs = eouts.size(0)

        # Append <sos> and <eos>
        eos = eouts.new_zeros((1,)).fill_(self.eos).long()
        ylens = [len(y) for y in ys]
        ys = [np2tensor(np.fromiter(y[::-1] if self.backward else y, dtype=np.int64), self.device_id).long()
              for y in ys]
        ys_in = [torch.cat([eos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        ys_in_pad = pad_list(ys_in, self.pad)
        ys_out_pad = pad_list(ys_out, self.pad)

        # Add positional embedding
        ys_emb = self.embed(ys_in_pad) * (self.d_model ** 0.5)
        if self.pe_type:
            ys_emb = self.pos_emb_out(ys_emb)

        for l in range(self.n_layers):
            ys_emb, yy_aw, xy_aw = self.layers[l](eouts, elens, ys_emb, ylens)

        logits = self.layer_norm_top(ys_emb)
        if self.adaptive_softmax is None:
            logits = self.output(logits)

        # Compute XE sequence loss
        if self.adaptive_softmax is None:
            if self.lsm_prob > 0:
                # Label smoothing
                loss = cross_entropy_lsm(logits, ys_out_pad,
                                         ylens=[y.size(0) for y in ys_out],
                                         lsm_prob=self.lsm_prob, size_average=False) / bs
            else:
                loss = F.cross_entropy(logits.view((-1, logits.size(2))), ys_out_pad.view(-1),
                                       ignore_index=self.pad, size_average=False) / bs
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
            elens (list): A list of length `[B]`
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
        ylens = np.zeros((bs,), dtype=np.int32)
        yy_aws_tmp = [None] * bs
        xy_aws_tmp = [None] * bs
        eos_flags = [False] * bs
        for t in range(int(np.floor(max_xlen * max_len_ratio)) + 1):
            # Add positional embedding
            out = self.embed(ys) * (self.d_model ** 0.5)
            if self.pe_type:
                out = self.pos_emb_out(out)

            for l in range(self.n_layers):
                out, yy_aw, xy_aw = self.layers[l](eouts, elens, out, ylens + 1)
                # xy_aw: `[B, head, T, L]`
            out = self.layer_norm_top(out)
            logits_t = self.output(out)

            # Pick up 1-best
            y = logits_t.detach().argmax(-1)[:, -1:]
            best_hyps_tmp += [y]

            # Count lengths of hypotheses
            for b in range(bs):
                if not eos_flags[b]:
                    if y[b].item() == self.eos:
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

        # if self.score.attn_n_heads > 1:
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

    def decode_ctc(self, eouts, xlens, beam_width=1, lm=None, lm_weight=0.0):
        """Decoding by the CTC layer in the inference stage.

            This is only used for Joint CTC-Attention model.
        Args:
            eouts (FloatTensor): `[B, T, d_model]`
            beam_width (int): size of beam
            lm ():
            lm_weight (float):
        Returns:
            best_hyps (list): A list of length `[B]`, which contains arrays of size `[L]`

        """
        log_probs = F.log_softmax(self.output_ctc(eouts), dim=-1)
        if beam_width == 1:
            best_hyps = self.decode_ctc_greedy(log_probs, xlens)
        else:
            best_hyps = self.decode_ctc_beam(log_probs, xlens, beam_width, lm, lm_weight)
            # TODO(hirofumi): add decoding paramters
        return best_hyps


class TransformerDecoderBlock(nn.Module):
    """A single layer of the transformer decoder.

        Args:
            d_model (int): dimension of keys/values/queries in
                           MultiheadAttentionMechanism, also the input size of
                           the first-layer of the PositionwiseFeedForward
            d_ff (int): second-layer of the PositionwiseFeedForward
            attn_type (str):
            attn_n_heads (int): number of heads for multi-head attention
            dropout (float): dropout probabilities for linear layers
            dropout_att (float): dropout probabilities for attention probabilities
            attn_type (str): type of self-attention, scaled_dot_product or average
            layer_norm_eps (float):

    """

    def __init__(self,
                 d_model,
                 d_ff,
                 attn_type,
                 attn_n_heads,
                 dropout,
                 dropout_att,
                 layer_norm_eps):
        super(TransformerDecoderBlock, self).__init__()

        self.attn_type = attn_type

        # self-attention
        if attn_type == "scaled_dot_product":
            self.self_attn = MultiheadAttentionMechanism(key_dim=d_model,
                                                         query_dim=d_model,
                                                         attn_dim=d_model,
                                                         n_heads=attn_n_heads,
                                                         dropout=dropout_att)
        elif attn_type == "average":
            raise NotImplementedError
            # self.self_attn = AverageAttention(d_model, dropout, layer_norm=True)
        else:
            raise NotImplementedError(attn_type)
        self.add_norm_self_attn = SublayerConnection(d_model, dropout, layer_norm_eps)

        # attention for encoder stacks
        self.src_attn = MultiheadAttentionMechanism(key_dim=d_model,
                                                    query_dim=d_model,
                                                    attn_dim=d_model,
                                                    n_heads=attn_n_heads,
                                                    dropout=dropout_att)
        self.add_norm_src_attn = SublayerConnection(d_model, dropout, layer_norm_eps)

        # feed-forward
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.add_norm_ff = SublayerConnection(d_model, dropout, layer_norm_eps)

    def forward(self, x, xlens, y, ylens):
        """Transformer decoder layer definition.

        Args:
            x (FloatTensor): encoder outputs. `[B, T, d_model]`
            xlens (list): `[B]`
            y (FloatTensor): `[B, L, d_model]`
            ylens (list): `[B]`
        Returns:
            y (FloatTensor): `[B, L, d_model]`
            yy_aw (FloatTensor)`[B, L, L]`
            xy_aw (FloatTensor): `[B, L, T]`

        """
        # self-attention
        if self.attn_type == "scaled_dot_product":
            y, yy_aw = self.add_norm_self_attn(y, lambda y: self.self_attn(
                key=y, key_lens=ylens, value=y, query=y, diagonal=True))
        elif self.attn_type == "average":
            raise NotImplementedError
        self.self_attn.reset()

        # attention for encoder stacks
        y, xy_aw = self.add_norm_src_attn(y, lambda y: self.src_attn(
            key=x, key_lens=xlens, value=x, query=y))
        self.src_attn.reset()

        # position-wise feed-forward
        y = self.add_norm_ff(y, lambda y: self.ff(y))

        return y, yy_aw, xy_aw
