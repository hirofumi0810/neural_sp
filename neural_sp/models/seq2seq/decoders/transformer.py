#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer decoder (including CTC loss calculation)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import numpy as np
import os
import random
import shutil
import torch
import torch.nn as nn

from neural_sp.models.criterion import cross_entropy_lsm
from neural_sp.models.lm.rnnlm import RNNLM
from neural_sp.models.modules.transformer import PositionalEncoding
from neural_sp.models.modules.transformer import TransformerDecoderBlock
from neural_sp.models.modules.transformer import SyncBidirTransformerDecoderBlock
from neural_sp.models.seq2seq.decoders.beam_search import BeamSearch
from neural_sp.models.seq2seq.decoders.ctc import CTC
from neural_sp.models.seq2seq.decoders.ctc import CTCPrefixScore
from neural_sp.models.seq2seq.decoders.decoder_base import DecoderBase
from neural_sp.models.torch_utils import append_sos_eos
from neural_sp.models.torch_utils import compute_accuracy
from neural_sp.models.torch_utils import make_pad_mask
from neural_sp.models.torch_utils import repeat
from neural_sp.models.torch_utils import tensor2np
from neural_sp.utils import mkdir_join

import matplotlib
matplotlib.use('Agg')

random.seed(1)

logger = logging.getLogger(__name__)


class TransformerDecoder(DecoderBase):
    """Transformer decoder.

    Args:
        special_symbols (dict):
            eos (int): index for <eos> (shared with <sos>)
            unk (int): index for <unk>
            pad (int): index for <pad>
            blank (int): index for <blank>
        enc_n_units (int): number of units of the encoder outputs
        attn_type (str): type of attention mechanism
        n_heads (int): number of attention heads
        n_layers (int): number of self-attention layers
        d_model (int): dimension of MultiheadAttentionMechanism
        d_ff (int): dimension of PositionwiseFeedForward
        pe_type (str): type of positional encoding
        layer_norm_eps (float): epsilon value for layer normalization
        ffn_activation (str): nonolinear function for PositionwiseFeedForward
        vocab (int): number of nodes in softmax layer
        tie_embedding (bool): tie parameters of the embedding and output layers
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
        param_init (str):
        mocha_n_heads_mono (int):
        mocha_tie_mono_attn (bool):
        mocha_n_heads_chunk (int):
        mocha_chunk_size (int):
        mocha_quantity_loss_weight (float):
        mocha_head_divergence_loss_weight (float):
        mocha_ctc_sync (str):
        mocha_minlt_loss_weight (float):

    """

    def __init__(self, special_symbols,
                 enc_n_units, attn_type, n_heads, n_layers, d_model, d_ff,
                 pe_type, layer_norm_eps, ffn_activation,
                 vocab, tie_embedding,
                 dropout, dropout_emb, dropout_att, lsm_prob,
                 ctc_weight, ctc_lsm_prob, ctc_fc_list,
                 backward, global_weight, mtl_per_batch, param_init,
                 mocha_n_heads_mono, mocha_tie_mono_attn,
                 mocha_n_heads_chunk, mocha_chunk_size,
                 mocha_quantity_loss_weight, mocha_head_divergence_loss_weight,
                 mocha_ctc_sync, mocha_minlt_loss_weight):

        super(TransformerDecoder, self).__init__()

        self.eos = special_symbols['eos']
        self.unk = special_symbols['unk']
        self.pad = special_symbols['pad']
        self.blank = special_symbols['blank']
        self.vocab = vocab
        self.enc_n_units = enc_n_units
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pe_type = pe_type
        self.lsm_prob = lsm_prob
        self.ctc_weight = ctc_weight
        self.bwd = backward
        self.global_weight = global_weight
        self.mtl_per_batch = mtl_per_batch

        self.prev_spk = ''
        self.lmstate_final = None

        # for attention plot
        self.aws_dict = {}
        self.data_dict = {}

        # for synchronous bidirectional decoding
        self.sync_bidir = ('sync_bidir' in attn_type)
        self.sync_bidir_half = (attn_type == 'sync_bidir_half')
        if self.sync_bidir:
            # expand vocabulary
            if self.sync_bidir_half:
                self.null = self.vocab
                self.l2r = self.vocab + 1
                self.r2l = self.vocab + 2
                self.vocab += 1
            else:
                self.l2r = self.vocab
                self.r2l = self.vocab + 1

        # for mocha
        self.attn_type = attn_type
        self.quantity_loss_weight = mocha_quantity_loss_weight
        self.headdiv_loss_weight = mocha_head_divergence_loss_weight
        self.ctc_sync = mocha_ctc_sync
        self.minlt_loss_weight = mocha_minlt_loss_weight

        if ctc_weight > 0:
            self.ctc = CTC(eos=self.eos,
                           blank=self.blank,
                           enc_n_units=enc_n_units,
                           vocab=self.vocab,
                           dropout=dropout,
                           lsm_prob=ctc_lsm_prob,
                           fc_list=ctc_fc_list,
                           param_init=0.1)

        if ctc_weight < global_weight:
            if self.sync_bidir:
                self.embed = nn.Embedding(self.vocab + 2, d_model, padding_idx=self.pad)
            else:
                self.embed = nn.Embedding(self.vocab, d_model, padding_idx=self.pad)
            self.pos_enc = PositionalEncoding(d_model, dropout_emb, pe_type)
            if self.sync_bidir:
                self.layers = repeat(SyncBidirTransformerDecoderBlock(
                    d_model, d_ff, n_heads, dropout, dropout_att,
                    layer_norm_eps, ffn_activation, param_init), n_layers)
            else:
                self.layers = repeat(TransformerDecoderBlock(
                    d_model, d_ff, attn_type, n_heads, dropout, dropout_att,
                    layer_norm_eps, ffn_activation, param_init,
                    mocha_chunk_size=mocha_chunk_size,
                    mocha_n_heads_mono=mocha_n_heads_mono,
                    mocha_n_heads_chunk=mocha_n_heads_chunk), n_layers)
            self.norm_out = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.output = nn.Linear(d_model, self.vocab)
            if tie_embedding:
                assert not self.sync_bidir
                self.output.weight = self.embed.weight

            if mocha_tie_mono_attn and attn_type == 'mocha':
                for l in range(1, self.n_layers):
                    self.layers[l].src_attn.monotonic_energy.w_key.weight = self.layers[0].src_attn.monotonic_energy.w_key.weight
                    self.layers[l].src_attn.monotonic_energy.w_key.bias = self.layers[0].src_attn.monotonic_energy.w_key.bias
                    self.layers[l].src_attn.monotonic_energy.w_query.weight = self.layers[0].src_attn.monotonic_energy.w_query.weight
                    self.layers[l].src_attn.monotonic_energy.w_query.bias = self.layers[0].src_attn.monotonic_energy.w_query.bias
                    self.layers[l].src_attn.monotonic_energy.r = self.layers[0].src_attn.monotonic_energy.r

            if param_init == 'xavier_uniform':
                self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
        # see https://github.com/pytorch/fairseq/blob/master/fairseq/models/transformer.py
        # embedding
        nn.init.normal_(self.embed.weight, mean=0., std=self.d_model**-0.5)
        nn.init.constant_(self.embed.weight[self.pad], 0.)
        # output layer
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.constant_(self.output.bias, 0.)

    def forward(self, eouts, elens, ys, task='all', ys_hist=[],
                teacher_logits=None, recog_params={}):
        """Forward computation.

        Args:
            eouts (FloatTensor): `[B, T, d_model]`
            elens (IntTensor): `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
            task (str): all/ys*/ys_sub*
            ys_hist (list): dummy (not used)
            teacher_logits (FloatTensor): `[B, L, vocab]`
            recog_params (dict): parameters for MBR training
        Returns:
            loss (FloatTensor): `[1]`
            observation (dict):

        """
        observation = {'loss': None, 'loss_att': None, 'loss_ctc': None, 'loss_mbr': None,
                       'acc_att': None, 'ppl_att': None}
        loss = eouts.new_zeros((1,))

        # CTC loss
        if self.ctc_weight > 0 and (task == 'all' or 'ctc' in task):
            loss_ctc, trigger_points = self.ctc(eouts, elens, ys,
                                                forced_align=self.ctc_sync and self.training)
            observation['loss_ctc'] = loss_ctc.item()
            if self.mtl_per_batch:
                loss += loss_ctc
            else:
                loss += loss_ctc * self.ctc_weight
        else:
            trigger_points = None

        # XE loss
        if self.global_weight - self.ctc_weight > 0 and (task == 'all' or 'ctc' not in task):
            if self.sync_bidir:
                loss_att, acc_att, ppl_att = self.forward_sync_bidir_att(eouts, elens, ys)
            else:
                loss_att, acc_att, ppl_att, loss_quantity, loss_headdiv, loss_latency = self.forward_att(
                    eouts, elens, ys, trigger_points=trigger_points)
            observation['loss_att'] = loss_att.item()
            observation['acc_att'] = acc_att
            observation['ppl_att'] = ppl_att
            if self.quantity_loss_weight > 0:
                loss_att += loss_quantity * self.quantity_loss_weight
                observation['loss_quantity'] = loss_quantity.item()
            if self.headdiv_loss_weight > 0:
                loss_att += loss_headdiv * self.headdiv_loss_weight
                observation['loss_headdiv'] = loss_headdiv.item()
            if self.ctc_sync in ['decot', 'minlt']:
                observation['loss_latency'] = loss_latency.item() if self.training else 0
                if self.ctc_sync == 'minlt' and self.minlt_loss_weight > 0:
                    loss_att += loss_latency * self.minlt_loss_weight
            if self.mtl_per_batch:
                loss += loss_att
            else:
                loss += loss_att * (self.global_weight - self.ctc_weight)

        observation['loss'] = loss.item()
        return loss, observation

    def forward_att(self, eouts, elens, ys,
                    return_logits=False, teacher_logits=None, trigger_points=None):
        """Compute XE loss for the Transformer decoder.

        Args:
            eouts (FloatTensor): `[B, T, d_model]`
            elens (IntTensor): `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
            return_logits (bool): return logits for knowledge distillation
            teacher_logits (FloatTensor): `[B, L, vocab]`
            trigger_points (IntTensor): `[B, T]`
        Returns:
            loss (FloatTensor): `[1]`
            acc (float): accuracy for token prediction
            ppl (float): perplexity
            loss_quantity (FloatTensor): `[1]`
            loss_headdiv (FloatTensor): `[1]`
            loss_latency (FloatTensor): `[1]`

        """
        # Append <sos> and <eos>
        ys_in, ys_out, ylens = append_sos_eos(eouts, ys, self.eos, self.eos, self.pad, self.bwd)
        if not self.training:
            self.data_dict['elens'] = tensor2np(elens)
            self.data_dict['ylens'] = tensor2np(ylens)
            self.data_dict['ys'] = tensor2np(ys_out)

        # Create target self-attention mask
        xtime = eouts.size(1)
        bs, ytime = ys_in.size()[:2]
        tgt_mask = (ys_out != self.pad).unsqueeze(1).repeat([1, ytime, 1])
        causal_mask = tgt_mask.new_ones(ytime, ytime).byte()
        causal_mask = torch.tril(causal_mask, out=causal_mask).unsqueeze(0)
        tgt_mask = tgt_mask & causal_mask  # `[B, L, L]`

        # Create source-target mask
        src_mask = make_pad_mask(elens, self.device_id).unsqueeze(1).repeat([1, ytime, 1])  # `[B, L, T]`

        out = self.pos_enc(self.embed(ys_in))
        xy_aws_layers = []
        for l in range(self.n_layers):
            out, yy_aws, xy_aws, xy_aws_beta = self.layers[l](
                out, tgt_mask, eouts, src_mask, mode='parallel')
            xy_aws_layers.append(xy_aws.clone())
            if not self.training:
                self.aws_dict['yy_aws_layer%d' % l] = tensor2np(yy_aws)
                self.aws_dict['xy_aws_layer%d' % l] = tensor2np(xy_aws)
                if xy_aws_beta is not None:
                    self.aws_dict['xy_aws_beta_layer%d' % l] = tensor2np(xy_aws_beta)
        logits = self.output(self.norm_out(out))

        # for knowledge distillation
        if return_logits:
            return logits

        # Compute XE sequence loss (+ label smoothing)
        loss, ppl = cross_entropy_lsm(logits, ys_out, self.lsm_prob, self.pad, self.training)

        # Attention padding
        if self.quantity_loss_weight > 0 or self.headdiv_loss_weight > 0 or self.minlt_loss_weight > 0:
            for l in range(self.n_layers):
                n_heads = xy_aws_layers[l].size(1)
                xy_aws_layers[l] = xy_aws_layers[l].masked_fill_(
                    src_mask.unsqueeze(1).repeat([1, n_heads, 1, 1]) == 0, 0)
                xy_aws_layers[l] = xy_aws_layers[l].masked_fill_(
                    tgt_mask[:, :, -1:].unsqueeze(1).repeat([1, n_heads, 1, xtime]) == 0, 0)
                # NOTE: attention padding is quite effective for quantity loss
        n_heads = xy_aws_layers[0].size(1)  # mono
        # NOTE: debug for multihead mono + multihead chunk

        # Quantity loss
        loss_quantity = 0.
        if self.quantity_loss_weight > 0:
            # Average over all heads across all layers
            n_tokens_ref = tgt_mask[:, -1, :].sum(1).float()  # `[B]`
            # NOTE: count <eos> tokens
            loss_quantity = sum([torch.mean(torch.abs(aws.sum(3).sum(2).sum(1) / n_heads - n_tokens_ref))
                                 for aws in xy_aws_layers])  # `[B]`
            loss_quantity = loss_quantity.sum() / self.n_layers

        # Head divergence loss
        loss_headdiv = 0.
        if self.headdiv_loss_weight > 0.:
            # Calculate variance over all heads across all layers
            js = torch.arange(xtime).float().cuda(self.device_id)
            js = js.repeat([bs, n_heads, ytime, 1])
            avg_head_pos = sum([(js * aws).sum(3).sum(1)
                                for aws in xy_aws_layers]) / (n_heads * self.n_layers)  # `[B, L]`
            loss_headdiv = sum([((js * aws).sum(3).sum(1) - avg_head_pos) ** 2
                                for aws in xy_aws_layers]) / (n_heads * self.n_layers)  # `[B, L]`
            loss_headdiv = loss_headdiv.sum() / ylens.sum()

        # Latency loss
        loss_latency = 0.
        if trigger_points is not None and self.ctc_sync == 'minlt':
            assert self.minlt_loss_weight > 0
            # Calculate weight average latency
            js = torch.arange(xtime).float().cuda(self.device_id)
            js = js.repeat([bs, n_heads, ytime, 1])
            weighted_avg_head_pos = torch.cat(
                [(js * aws).sum(3) for aws in xy_aws_layers], dim=1)  # `[B, H_mono * n_layers, L]`
            weighted_avg_head_pos *= torch.softmax(weighted_avg_head_pos.clone(), dim=1)
            trigger_points = trigger_points.float().cuda(self.device_id)  # `[B, L]`
            trigger_points = trigger_points.unsqueeze(1)
            loss_latency = torch.abs(weighted_avg_head_pos - trigger_points)  # `[B, H_mono * n_layers, L]`
            # NOTE: trigger_points are padded with 0
            loss_latency = loss_latency.sum() / ylens.sum()

        # Compute token-level accuracy in teacher-forcing
        acc = compute_accuracy(logits, ys_out, self.pad)

        return loss, acc, ppl, loss_quantity, loss_headdiv, loss_latency

    def forward_sync_bidir_att(self, eouts, elens, ys, return_logits=False, teacher_logits=None):
        """Compute XE loss for the synchronous bidirectional Transformer decoder.

        Args:
            eouts (FloatTensor): `[B, T, d_model]`
            elens (IntTensor): `[B]`
            ys (list): A list of length `[B]`, which contains a list of size `[L]`
            return_logits (bool): return logits for knowledge distillation
            teacher_logits (FloatTensor): `[B, L, vocab]`
        Returns:
            loss (FloatTensor): `[1]`
            acc_fwd (float): accuracy for token prediction
            ppl_fwd (float): perplexity

        """
        # Append <sos> and <eos>
        if self.sync_bidir_half:
            # Add randomness
            if random.random() < 0.5:
                null_first = True
                ys_first = [y[:math.ceil(len(y) / 2)] for y in ys]
                ys_second = [y[len(y) // 2:] for y in ys]
            else:
                null_first = False
                ys_first = [y[:len(y) // 2] for y in ys]
                ys_second = [y[math.ceil(len(y) / 2):] for y in ys]
            ys_in, ys_out, ylens_fwd = append_sos_eos(
                eouts, ys_first, self.l2r, self.eos, self.pad)
            ys_bwd_in, ys_bwd_out, ylens_bwd = append_sos_eos(
                eouts, ys_second, self.r2l, self.eos, self.pad, bwd=True)
            for b in range(eouts.size(0)):
                if len(ys[b]) % 2 != 0:
                    if null_first:
                        ylens = ylens_fwd
                        ys_bwd_in[b, ylens[b] - 1] = self.null
                        ys_bwd_out[b, ylens[b] - 2] = self.null
                        ys_bwd_out[b, ylens[b] - 1] = self.eos
                    else:
                        ylens = ylens_bwd
                        ys_in[b, ylens[b] - 1] = self.null
                        ys_out[b, ylens[b] - 2] = self.null
                        ys_out[b, ylens[b] - 1] = self.eos
                    # NOTE: ylens counts <eos>
        else:
            ys_in, ys_out, ylens = append_sos_eos(
                eouts, ys, self.l2r, self.eos, self.pad)
            ys_bwd_in, ys_bwd_out, _ = append_sos_eos(
                eouts, ys, self.r2l, self.eos, self.pad, bwd=True)
        if not self.training:
            self.data_dict['elens'] = tensor2np(elens)
            self.data_dict['ylens'] = tensor2np(ylens)
            self.data_dict['ys'] = tensor2np(ys_out)
            self.data_dict['ys_bwd'] = tensor2np(ys_bwd_out)

        # Create target self-attention mask
        bs, ytime = ys_in.size()[:2]
        ylens_mask = (ys_out != self.pad).unsqueeze(1).repeat([1, ytime, 1])
        causal_mask = ylens_mask.new_ones(ytime, ytime).byte()
        causal_mask = torch.tril(causal_mask, out=causal_mask).unsqueeze(0)
        tgt_mask = ylens_mask & causal_mask

        # Create source-target mask
        src_mask = make_pad_mask(elens, self.device_id).unsqueeze(1).repeat([1, ytime, 1])

        # Create idendity token mask for synchronous bidirectional attention
        identity_mask = tgt_mask.clone()
        # if not self.sync_bidir_half:
        #     for b in range(bs):
        #         for i in range(ylens[b] - 1):
        #             identity_mask[b, i, (ylens[b] - 1) - i] = 0
        #             # NOTE: ylens counts <eos>

        out = self.pos_enc(self.embed(ys_in))
        out_bwd = self.pos_enc(self.embed(ys_bwd_in))
        for l in range(self.n_layers):
            out, out_bwd, yy_aws_h, yy_aws_f, yy_aws_bwd_h, yy_aws_bwd_f, xy_aws, xy_aws_bwd = self.layers[l](
                out, out_bwd, tgt_mask, identity_mask, eouts, src_mask)
            if not self.training:
                self.aws_dict['yy_aws_fwd_history_layer%d' % l] = tensor2np(yy_aws_h)
                self.aws_dict['yy_aws_fwd_future_layer%d' % l] = tensor2np(yy_aws_f)
                self.aws_dict['yy_aws_bwd_history_layer%d' % l] = tensor2np(yy_aws_bwd_h)
                self.aws_dict['yy_aws_bwd_future_layer%d' % l] = tensor2np(yy_aws_bwd_f)
                self.aws_dict['xy_aws_fwd_layer%d' % l] = tensor2np(xy_aws)
                self.aws_dict['xy_aws_bwd_layer%d' % l] = tensor2np(xy_aws_bwd)
        logits = self.output(self.norm_out(out))
        logits_bwd = self.output(self.norm_out(out_bwd))

        # for knowledge distillation
        if return_logits:
            return logits

        # Compute XE sequence loss (+ label smoothing)
        loss_fwd, ppl_fwd = cross_entropy_lsm(logits, ys_out, self.lsm_prob, self.pad, self.training)
        loss_bwd, ppl_bwd = cross_entropy_lsm(logits_bwd, ys_bwd_out, self.lsm_prob, self.pad, self.training)
        loss = loss_fwd * 0.5 + loss_bwd * 0.5

        # Compute token-level accuracy in teacher-forcing
        acc_fwd = compute_accuracy(logits, ys_out, self.pad)
        # acc_bwd = compute_accuracy(logits_bwd, ys_bwd_out, self.pad)

        return loss, acc_fwd, ppl_fwd

    def _plot_attention(self, save_path, n_cols=2):
        """Plot attention for each head in all layers."""
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator

        _save_path = mkdir_join(save_path, 'dec_att_weights')

        # Clean directory
        if _save_path is not None and os.path.isdir(_save_path):
            shutil.rmtree(_save_path)
            os.mkdir(_save_path)

        for k, aw in self.aws_dict.items():
            elens = self.data_dict['elens']
            ylens = self.data_dict['ylens']
            # ys = self.data_dict['ys']

            plt.clf()
            n_heads = aw.shape[1]
            n_cols_tmp = 1 if n_heads == 1 else n_cols
            fig, axes = plt.subplots(max(1, n_heads // n_cols_tmp), n_cols_tmp,
                                     figsize=(20, 8), squeeze=False)
            for h in range(n_heads):
                ax = axes[h // n_cols_tmp, h % n_cols_tmp]
                if 'xy' in k:
                    ax.imshow(aw[-1, h, :ylens[-1], :elens[-1]], aspect="auto")
                else:
                    ax.imshow(aw[-1, h, :ylens[-1], :ylens[-1]], aspect="auto")
                ax.grid(False)
                ax.set_xlabel("Input (head%d)" % h)
                ax.set_ylabel("Output (head%d)" % h)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                # ax.set_yticks(np.linspace(0, ylens[-1] - 1, ylens[-1]))
                # ax.set_yticks(np.linspace(0, ylens[-1] - 1, 1), minor=True)
                # ax.set_yticklabels(ys + [''])

            fig.tight_layout()
            fig.savefig(os.path.join(_save_path, '%s.png' % k), dvi=500)
            plt.close()

    def greedy(self, eouts, elens, max_len_ratio, idx2token,
               exclude_eos=False, refs_id=None, utt_ids=None, speakers=None):
        """Greedy decoding.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            elens (IntTensor): `[B]`
            max_len_ratio (int): maximum sequence length of tokens
            idx2token (): converter from index to token
            exclude_eos (bool): exclude <eos> from hypothesis
            refs_id (list): reference list
            utt_ids (list): utterance id list
            speakers (list): speaker list
        Returns:
            hyps (list): A list of length `[B]`, which contains arrays of size `[L]`
            aw (list): A list of length `[B]`, which contains arrays of size `[L, T]`

        """
        bs, xtime = eouts.size()[:2]

        if self.sync_bidir:
            ys = eouts.new_zeros(bs, 1).fill_(self.l2r).long()
            ys_bwd = eouts.new_zeros(bs, 1).fill_(self.r2l).long()
        else:
            ys = eouts.new_zeros(bs, 1).fill_(self.eos).long()

        hyps_batch = []
        ylens = torch.zeros(bs).int()
        eos_flags = [False] * bs
        ytime = int(math.floor(xtime * max_len_ratio)) + 1
        for t in range(ytime):
            causal_mask = eouts.new_ones(t + 1, t + 1).byte()
            causal_mask = torch.tril(causal_mask, out=causal_mask).unsqueeze(0)

            dout = self.pos_enc(self.embed(ys))
            if self.sync_bidir:
                dout_bwd = self.pos_enc(self.embed(ys_bwd))
                for l in range(self.n_layers):
                    dout, dout_bwd, _, _, _, _, xy_aws, xy_aws_bwd = self.layers[l](
                        dout, dout_bwd, causal_mask, None, eouts, None)

                # Pick up 1-best
                y_bwd = self.output(self.norm_out(dout_bwd))[:, -1:].argmax(-1)
            else:
                for l in range(self.n_layers):
                    dout, _, xy_aws, _ = self.layers[l](dout, causal_mask, eouts, None)

            # Pick up 1-best
            y = self.output(self.norm_out(dout))[:, -1:].argmax(-1)
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

            ys = torch.cat([ys, y], dim=-1)
            if self.sync_bidir:
                ys_bwd = torch.cat([ys_bwd, y_bwd], dim=-1)

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
                logger.debug('Utt-id: %s' % utt_ids[b])
            if refs_id is not None and self.vocab == idx2token.vocab:
                logger.debug('Ref: %s' % idx2token(refs_id[b]))
            if self.bwd:
                logger.debug('Hyp: %s' % idx2token(hyps[b][::-1]))
            else:
                logger.debug('Hyp: %s' % idx2token(hyps[b]))

        return hyps, aws

    def beam_search(self, eouts, elens, params, idx2token=None,
                    lm=None, lm_second=None, lm_bwd=None, ctc_log_probs=None,
                    nbest=1, exclude_eos=False,
                    refs_id=None, utt_ids=None, speakers=None,
                    ensmbl_eouts=None, ensmbl_elens=None, ensmbl_decs=[],
                    cache_states=False):
        """Beam search decoding.

        Args:
            eouts (FloatTensor): `[B, T, d_model]`
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
            lm: firsh path LM
            lm_second: second path LM
            lm_bwd: first/secoding path backward LM
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
            aws (list): A list of length `[B]`, which contains arrays of size `[H, L, T]`
            scores (list):

        """
        bs, xmax, _ = eouts.size()
        n_models = len(ensmbl_decs) + 1

        beam_width = params['recog_beam_width']
        if self.sync_bidir:
            beam_width //= 2
        beam_width_bwd = params['recog_beam_width'] // 2 if self.sync_bidir else 0
        assert 1 <= nbest <= beam_width
        ctc_weight = params['recog_ctc_weight']
        max_len_ratio = params['recog_max_len_ratio']
        min_len_ratio = params['recog_min_len_ratio']
        lp_weight = params['recog_length_penalty']
        length_norm = params['recog_length_norm']
        lm_weight = params['recog_lm_weight']
        lm_weight_bwd = 0.1
        lm_weight_second = params['recog_lm_second_weight']
        lm_weight_bwd = params['recog_lm_bwd_weight']
        eos_threshold = params['recog_eos_threshold']
        lm_state_carry_over = params['recog_lm_state_carry_over']
        softmax_smoothing = params['recog_softmax_smoothing']

        # TODO:
        # - aws
        # - visualization
        # - cache
        # shallow fusion for bwd

        if lm is not None:
            assert lm_weight > 0
            lm.eval()
        if lm_second is not None:
            assert lm_weight_second > 0
            lm_second.eval()
        if lm_bwd is not None:
            assert lm_weight_bwd > 0
            lm_bwd.eval()

        if ctc_log_probs is not None:
            assert ctc_weight > 0
            ctc_log_probs = tensor2np(ctc_log_probs)

        nbest_hyps_idx, aws, scores = [], [], []
        eos_flags = []
        for b in range(bs):
            # Initialization per utterance
            lmstate = None
            if self.sync_bidir:
                ys = eouts.new_zeros(1, 1).fill_(self.l2r).long()
                ys_bwd = eouts.new_zeros(1, 1).fill_(self.r2l).long()
            else:
                ys = eouts.new_zeros(1, 1).fill_(self.eos).long()
                ys_bwd = None

            # For joint CTC-Attention decoding
            ctc_prefix_scorer = None
            ctc_prefix_scorer_bwd = None
            if ctc_log_probs is not None:
                if self.bwd:
                    ctc_prefix_scorer = CTCPrefixScore(ctc_log_probs[b][::-1], self.blank, self.eos)
                else:
                    ctc_prefix_scorer = CTCPrefixScore(ctc_log_probs[b], self.blank, self.eos)
                    if self.sync_bidir:
                        ctc_prefix_scorer_bwd = CTCPrefixScore(ctc_log_probs[b][::-1], self.blank, self.eos)

            if speakers is not None:
                if speakers[b] == self.prev_spk:
                    if lm_state_carry_over and isinstance(lm, RNNLM):
                        lmstate = self.lmstate_final
                self.prev_spk = speakers[b]

            helper = BeamSearch(beam_width, self.eos, ctc_weight, self.device_id,
                                beam_width_bwd=beam_width_bwd)

            end_hyps = []
            hyps = [{'hyp': [self.eos],
                     'hyp_bwd': [self.eos],
                     'ys': ys,
                     'ys_bwd': ys_bwd,
                     'cache': None,
                     'cache_bwd': None,
                     'score': 0.,
                     'score_attn': 0.,
                     'score_attn_bwd': 0.,
                     'score_ctc': 0.,
                     'score_lm': 0.,
                     'score_lm_bwd': 0.,
                     'aws': [None],
                     'lmstate': lmstate,
                     'lmstate_bwd': None,
                     'ensmbl_aws':[[None]] * (n_models - 1),
                     'ctc_state': ctc_prefix_scorer.initial_state() if ctc_prefix_scorer is not None else None,
                     'ctc_state_bwd': ctc_prefix_scorer_bwd.initial_state() if ctc_prefix_scorer is not None and self.sync_bidir else None,
                     'bwd': False}]
            if self.sync_bidir:
                hyps_bwd = copy.deepcopy(hyps)
                hyps_bwd[0]['bwd'] = True
            ytime = int(math.floor(elens[b] * max_len_ratio)) + 1
            for t in range(ytime):
                # merge fwd and bwd hypotheses for batch decoding
                hyps_merge = hyps + hyps_bwd if self.sync_bidir else hyps[:]
                if len(hyps_merge) == 0:
                    break

                # preprocess for batch decoding
                cache = [None] * self.n_layers
                if cache_states and t > 0:
                    for l in range(self.n_layers):
                        cache[l] = torch.cat([beam['cache'][l] for beam in hyps_merge], dim=0)
                ys = eouts.new_zeros(len(hyps_merge), t + 1).long()
                for j, beam in enumerate(hyps_merge):
                    ys[j, :] = beam['ys']
                if t > 0:
                    xy_aws_prev = torch.cat([beam['aws'][-1] for beam in hyps_merge], dim=0)
                    xy_aws_prev = xy_aws_prev.view(len(hyps_merge), self.n_layers, -1, 1, xmax)
                    # `[B, n_layers * H_mono, qlen, klen]`
                else:
                    xy_aws_prev = None

                cache_bwd = [None] * self.n_layers
                if self.sync_bidir:
                    if cache_states and t > 0:
                        for l in range(self.n_layers):
                            cache_bwd[l] = torch.cat([beam['cache_bwd'][l] for beam in hyps_merge], dim=0)
                    ys_bwd = eouts.new_zeros(len(hyps_merge), t + 1).long()
                    for j, beam in enumerate(hyps_merge):
                        ys_bwd[j, :] = beam['ys_bwd']

                # Update LM states for shallow fusion
                lmstate, scores_lm = None, None
                lmstate_bwd, scores_lm_bwd = None, None
                if lm is not None:
                    if beam['lmstate'] is not None:
                        lm_hxs = torch.cat([beam['lmstate']['hxs'] for beam in hyps_merge], dim=1)
                        lm_cxs = torch.cat([beam['lmstate']['cxs'] for beam in hyps_merge], dim=1)
                        lmstate = {'hxs': lm_hxs, 'cxs': lm_cxs}
                    y = ys[:, -1:].clone()  # NOTE: this is important
                    if self.sync_bidir and t == 0:
                        y[:, 0] = self.eos
                    _, lmstate, scores_lm = lm.predict(y, lmstate)
                if self.sync_bidir and lm_bwd is not None:
                    if beam['lmstate_bwd'] is not None:
                        lm_hxs_bwd = torch.cat([beam['lmstate_bwd']['hxs'] for beam in hyps_merge], dim=1)
                        lm_cxs_bwd = torch.cat([beam['lmstate_bwd']['cxs'] for beam in hyps_merge], dim=1)
                        lmstate_bwd = {'hxs': lm_hxs_bwd, 'cxs': lm_cxs_bwd}
                    y_bwd = ys_bwd[:, -1:].clone()  # NOTE: this is important
                    if t == 0:
                        y_bwd[:, 0] = self.eos
                    _, lmstate_bwd, scores_lm_bwd = lm_bwd.predict(y_bwd, lmstate_bwd)

                # for the main model
                causal_mask = eouts.new_ones(t + 1, t + 1).byte()
                causal_mask = torch.tril(causal_mask, out=causal_mask).unsqueeze(
                    0).repeat([ys.size(0), 1, 1])

                eouts_b = eouts[b:b + 1, :elens[b]].repeat([ys.size(0), 1, 1])
                new_cache = [None] * self.n_layers
                new_cache_bwd = [None] * self.n_layers
                dout = self.pos_enc(self.embed(ys))
                xy_aws_all_layers = []
                if self.sync_bidir:
                    dout_bwd = self.pos_enc(self.embed(ys_bwd))
                    for l in range(self.n_layers):
                        dout, dout_bwd, _, _, _, _, xy_aws, xy_aws_bwd = self.layers[l](
                            dout, dout_bwd, causal_mask, None, eouts_b, None,
                            cache=cache[l], cache_bwd=cache_bwd[l])
                        new_cache[l] = dout
                        new_cache_bwd[l] = dout_bwd
                        xy_aws_all_layers.append(xy_aws)
                    logits_bwd = self.output(self.norm_out(dout_bwd))
                    probs_bwd = torch.softmax(logits_bwd[:, -1] * softmax_smoothing, dim=1)
                else:
                    for l in range(self.n_layers):
                        dout, _, xy_aws, xy_aws_beta = self.layers[l](dout, causal_mask, eouts_b, None,
                                                                      cache=cache[l],
                                                                      xy_aws_prev=xy_aws_prev[:, l] if t > 0 else None)
                        new_cache[l] = dout
                        xy_aws_all_layers.append(xy_aws)
                logits = self.output(self.norm_out(dout))
                probs = torch.softmax(logits[:, -1] * softmax_smoothing, dim=1)
                xy_aws_all_layers = torch.stack(xy_aws_all_layers, dim=2)  # `[B, H, n_layers, L, T]`
                xy_aws_all_layers = xy_aws_all_layers.view(
                    xy_aws_all_layers.size(0), -1, xy_aws_all_layers.size(3), xmax)

                # for the ensemble
                ensmbl_new_cache = []
                if n_models > 1:
                    # Ensemble initialization
                    ensmbl_cache = []
                    # cache_e = [None] * self.n_layers
                    # if cache_states and t > 0:
                    #     for l in range(self.n_layers):
                    #         cache_e[l] = torch.cat([beam['ensmbl_cache'][l] for beam in hyps_merge], dim=0)
                    for i_e, dec in enumerate(ensmbl_decs):
                        dout_e = dec.pos_enc(dec.embed(ys))
                        eouts_e = ensmbl_eouts[i_e][b:b + 1, :elens[b]].repeat([ys.size(0), 1, 1])
                        new_cache_e = [None] * dec.n_layers
                        for l in range(dec.n_layers):
                            dout_e, _, xy_aws_e, _ = dec.layers[l](dout_e, causal_mask, eouts_e, None,
                                                                   cache=cache[l])
                            new_cache_e[l] = dout_e
                        ensmbl_new_cache.append(new_cache_e)
                        logits_e = dec.output(dec.norm_out(dout_e))
                        probs += torch.softmax(logits_e[:, -1] * softmax_smoothing, dim=1)
                        # NOTE: sum in the probability scale (not log-scale)

                # Ensemble in log-scale
                scores_attn = torch.log(probs) / n_models
                if self.sync_bidir:
                    scores_attn_bwd = torch.log(probs_bwd)

                new_hyps, new_hyps_bwd = [], []
                for j, beam in enumerate(hyps_merge):
                    # Attention scores
                    total_scores_attn = beam['score_attn'] + scores_attn[j:j + 1]
                    total_scores = total_scores_attn * (1 - ctc_weight)
                    total_scores_topk, topk_ids = torch.topk(
                        total_scores, k=beam_width, dim=1, largest=True, sorted=True)
                    if self.sync_bidir:
                        total_scores_attn_bwd = beam['score_attn_bwd'] + scores_attn_bwd[j:j + 1]
                        total_scores_bwd = total_scores_attn_bwd * (1 - ctc_weight)
                        total_scores_topk_bwd, topk_ids_bwd = torch.topk(
                            total_scores_bwd, k=beam_width, dim=1, largest=True, sorted=True)

                    # Add LM score <after> top-K selection
                    if lm is not None:
                        total_scores_lm = beam['score_lm'] + scores_lm[j, -1, topk_ids[0]]
                        total_scores_topk += total_scores_lm * lm_weight
                    else:
                        total_scores_lm = eouts.new_zeros(beam_width)
                    if self.sync_bidir and lm_bwd is not None:
                        total_scores_lm_bwd = beam['score_lm_bwd'] + scores_lm_bwd[j, -1, topk_ids_bwd[0]]
                        total_scores_topk_bwd += total_scores_lm_bwd * lm_weight_bwd
                    else:
                        total_scores_lm_bwd = eouts.new_zeros(beam_width)

                    # Add length penalty
                    if lp_weight > 0:
                        total_scores_topk += (len(beam['hyp'][1:]) + 1) * lp_weight
                        if self.sync_bidir:
                            total_scores_topk_bwd += (len(beam['hyp_bwd'][1:]) + 1) * lp_weight

                    # Add CTC score
                    new_ctc_states, total_scores_ctc, total_scores_topk = helper.add_ctc_score(
                        beam['hyp'], topk_ids, beam['ctc_state'],
                        total_scores_topk, ctc_prefix_scorer)
                    if self.sync_bidir:
                        new_ctc_states_bwd, total_scores_ctc_bwd, total_scores_topk_bwd = helper.add_ctc_score(
                            beam['hyp_bwd'], topk_ids_bwd, beam['ctc_state_bwd'],
                            total_scores_topk_bwd, ctc_prefix_scorer_bwd)
                    else:
                        total_scores_ctc_bwd = eouts.new_zeros(beam_width)

                    # forward direction
                    k_bwd = 0  # choose the best hyp from backward direction
                    for k in range(beam_width):
                        if beam['bwd']:
                            continue

                        idx = topk_ids[0, k].item()
                        length_norm_factor = len(beam['hyp'][1:]) + 1 if length_norm else 1
                        total_scores_topk /= length_norm_factor
                        if self.sync_bidir:
                            total_scores_topk_bwd /= length_norm_factor

                        if idx == self.eos:
                            # Exclude short hypotheses
                            if len(beam['hyp']) - 1 < elens[b] * min_len_ratio:
                                continue
                            # EOS threshold
                            max_score_no_eos = scores_attn[j, :idx].max(0)[0].item()
                            max_score_no_eos = max(max_score_no_eos, scores_attn[j, idx + 1:].max(0)[0].item())
                            if scores_attn[j, idx].item() <= eos_threshold * max_score_no_eos:
                                continue

                        ys = torch.cat([beam['ys'], eouts.new_zeros(1, 1).fill_(idx).long()], dim=-1)
                        if self.sync_bidir:
                            if t > 0 and beam['hyp_bwd'][-1] == self.eos:
                                idx_bwd = self.eos  # pad with <eos>
                            else:
                                idx_bwd = topk_ids_bwd[0, k_bwd].item()
                            ys_bwd = torch.cat(
                                [beam['ys_bwd'], eouts.new_zeros(1, 1).fill_(idx_bwd).long()], dim=-1)

                        new_hyps.append(
                            {'hyp': beam['hyp'] + [idx],
                             'hyp_bwd': beam['hyp_bwd'] + [idx_bwd] if self.sync_bidir else None,
                             'ys': ys,
                             'ys_bwd': ys_bwd if self.sync_bidir else None,
                             'cache': [new_cache_l[j:j + 1] for new_cache_l in new_cache] if cache_states else cache,
                             'cache_bwd': [new_cache_l[j:j + 1] for new_cache_l in new_cache_bwd] if self.sync_bidir and cache_states else cache_bwd,
                             'score': total_scores_topk[0, k].item(),
                             'score_attn': total_scores_attn[0, idx].item(),
                             'score_attn_bwd': total_scores_attn_bwd[0, idx_bwd].item() if self.sync_bidir else None,
                             'score_ctc': total_scores_ctc[k].item(),
                             'score_ctc_bwd': total_scores_ctc_bwd[k_bwd].item(),
                             'score_lm': total_scores_lm[k].item(),
                             'score_lm_bwd': total_scores_lm_bwd[k_bwd].item(),
                             'aws': beam['aws'] + [xy_aws_all_layers[j:j + 1, :, -1:]],
                             'lmstate': {'hxs': lmstate['hxs'][:, j:j + 1], 'cxs': lmstate['cxs'][:, j:j + 1]} if lmstate is not None else None,
                             'lmstate_bwd': {'hxs': lmstate_bwd['hxs'][:, j:j + 1], 'cxs': lmstate_bwd['cxs'][:, j:j + 1]} if lmstate_bwd is not None else None,
                             'ctc_state': new_ctc_states[k] if ctc_prefix_scorer is not None else None,
                             'ctc_state_bwd': new_ctc_states_bwd[k_bwd] if ctc_prefix_scorer is not None and self.sync_bidir else None,
                             'ensmbl_cache': ensmbl_new_cache,
                             'bwd': False})

                    # backward direction
                    if self.sync_bidir:
                        k = 0  # choose the best hyp from forward direction
                        for k_bwd in range(beam_width_bwd):
                            if not beam['bwd']:
                                continue

                            idx_bwd = topk_ids_bwd[0, k_bwd].item()
                            length_norm_factor = len(beam['hyp_bwd'][1:]) + 1 if length_norm else 1
                            total_scores_topk /= length_norm_factor
                            total_scores_topk_bwd /= length_norm_factor

                            if idx_bwd == self.eos:
                                # Exclude short hypotheses
                                if len(beam['hyp_bwd']) - 1 < elens[b] * min_len_ratio:
                                    continue
                                # EOS threshold
                                max_score_no_eos = scores_attn_bwd[j, :idx_bwd].max(0)[0].item()
                                max_score_no_eos = max(
                                    max_score_no_eos, scores_attn_bwd[j, idx_bwd + 1:].max(0)[0].item())
                                if scores_attn_bwd[j, idx_bwd].item() <= eos_threshold * max_score_no_eos:
                                    continue

                            if t > 0 and beam['hyp'][-1] == self.eos:
                                idx = self.eos
                            else:
                                idx = topk_ids[0, k].item()
                            ys = torch.cat([beam['ys'], eouts.new_zeros(1, 1).fill_(idx).long()], dim=-1)
                            ys_bwd = torch.cat(
                                [beam['ys_bwd'], eouts.new_zeros(1, 1).fill_(idx_bwd).long()], dim=-1)

                            new_hyps_bwd.append(
                                {'hyp': beam['hyp'] + [idx],
                                 'hyp_bwd': beam['hyp_bwd'] + [idx_bwd],
                                 'ys': ys,
                                 'ys_bwd': ys_bwd,
                                 'cache': [new_cache_l[j:j + 1] for new_cache_l in new_cache] if cache_states else cache,
                                 'cache_bwd': [new_cache_l[j:j + 1] for new_cache_l in new_cache_bwd] if cache_states else cache_bwd,
                                 'score': total_scores_topk_bwd[0, k_bwd].item(),
                                 'score_attn': total_scores_attn[0, idx].item(),
                                 'score_attn_bwd': total_scores_attn_bwd[0, idx_bwd].item(),
                                 'score_ctc': total_scores_ctc[k].item(),
                                 'score_ctc_bwd': total_scores_ctc_bwd[k_bwd].item(),
                                 'score_lm': total_scores_lm[k].item(),
                                 'score_lm_bwd': total_scores_lm_bwd[k_bwd].item(),
                                 'aws': beam['aws'] + [xy_aws_all_layers[j:j + 1, :, -1:]],
                                 'lmstate': {'hxs': lmstate['hxs'][:, j:j + 1], 'cxs': lmstate['cxs'][:, j:j + 1]} if lmstate is not None else None,
                                 'lmstate_bwd': {'hxs': lmstate_bwd['hxs'][:, j:j + 1], 'cxs': lmstate_bwd['cxs'][:, j:j + 1]} if lmstate_bwd is not None else None,
                                 'ctc_state': new_ctc_states[k] if ctc_prefix_scorer is not None else None,
                                 'ctc_state_bwd': new_ctc_states_bwd[k_bwd] if ctc_prefix_scorer is not None else None,
                                 'ensmbl_cache': ensmbl_new_cache,
                                 'bwd': True})

                # Local pruning
                new_hyps_sorted = sorted(new_hyps, key=lambda x: x['score'], reverse=True)[:beam_width]
                if self.sync_bidir:
                    new_hyps_bwd_sorted = sorted(new_hyps_bwd, key=lambda x: x['score'], reverse=True)[:beam_width_bwd]

                # Remove complete hypotheses
                # new_hyps, end_hyps, is_finish = helper.remove_complete_hyp(
                #     new_hyps_sorted, end_hyps, prune=not self.sync_bidir)
                new_hyps, end_hyps, is_finish = helper.remove_complete_hyp(
                    new_hyps_sorted, end_hyps, prune=True)
                hyps = new_hyps[:]
                if self.sync_bidir:
                    # new_hyps_bwd, end_hyps, is_finish = helper.remove_complete_hyp(
                    #     new_hyps_bwd_sorted, end_hyps, backward=True)
                    # hyps_bwd = new_hyps_bwd[:]
                    hyps_bwd = new_hyps_bwd_sorted[:]
                if is_finish:
                    break

            # Global pruning
            if len(end_hyps) == 0:
                end_hyps = hyps_merge[:]
            elif len(end_hyps) < nbest and nbest > 1:
                end_hyps.extend(hyps_merge[:nbest - len(end_hyps)])

            if self.sync_bidir_half:
                for hyp in end_hyps:
                    hyp_first = [idx for idx in hyp['hyp'] if idx not in [self.eos, self.null]]
                    hyp_second = [idx for idx in hyp['hyp_bwd'][1:][::-1] if idx not in [self.eos, self.null]]
                    hyp['hyp'] = hyp_first + hyp_second

            # forward second path LM rescoring
            if lm_second is not None:
                self.lm_rescoring(end_hyps, lm_second, lm_weight_second, tag='second')

            # backward secodn path LM rescoring
            if lm_bwd is not None and lm_weight_bwd > 0:
                self.lm_rescoring(end_hyps, lm_bwd, lm_weight_bwd, tag='second_bwd')

            # Sort by score
            end_hyps = sorted(end_hyps, key=lambda x: x['score'], reverse=True)

            if utt_ids is not None:
                logger.info('Utt-id: %s' % utt_ids[b])
            if idx2token is not None:
                logger.info('=' * 200)
                for k in range(len(end_hyps)):
                    if refs_id is not None:
                        logger.info('Ref: %s' % idx2token(refs_id[b]))
                    logger.info('Hyp (fwd): %s' % idx2token(
                        end_hyps[k]['hyp'][1:][::-1] if self.bwd else end_hyps[k]['hyp'][1:]))
                    if self.sync_bidir:
                        if end_hyps[k]['bwd']:
                            logger.info('■■■ backward hypothesis ■■■')
                        logger.info('Hyp (bwd): %s' % idx2token(end_hyps[k]['hyp_bwd'][1:][::-1]))
                    logger.info('log prob (hyp): %.7f' % end_hyps[k]['score'])
                    logger.info('log prob (hyp, att-fwd): %.7f' % (end_hyps[k]['score_attn'] * (1 - ctc_weight)))
                    if self.sync_bidir:
                        logger.info('log prob (hyp, att-bwd): %.7f' %
                                    (end_hyps[k]['score_attn_bwd'] * (1 - ctc_weight)))
                    if ctc_prefix_scorer is not None:
                        logger.info('log prob (hyp, ctc-fwd): %.7f' % (end_hyps[k]['score_ctc'] * ctc_weight))
                        if self.sync_bidir:
                            logger.info('log prob (hyp, ctc-bwd): %.7f' % (end_hyps[k]['score_ctc_bwd'] * ctc_weight))
                    if lm is not None:
                        logger.info('log prob (hyp, first-path lm-fwd): %.7f' % (end_hyps[k]['score_lm'] * lm_weight))
                    if lm_second is not None:
                        logger.info('log prob (hyp, second-path lm-fwd): %.7f' %
                                    (end_hyps[k]['score_lm_second'] * lm_weight_second))
                    if lm_bwd is not None:
                        if self.sync_bidir:
                            logger.info('log prob (hyp, first-path lm-bwd): %.7f' %
                                        (end_hyps[k]['score_lm_bwd'] * lm_weight_bwd))
                        else:
                            logger.info('log prob (hyp, second-path lm-bwd): %.7f' %
                                        (end_hyps[k]['score_lm_second_bwd'] * lm_weight_bwd))
                    logger.info('-' * 50)

            # N-best list
            if self.bwd:
                # Reverse the order
                nbest_hyps_idx += [[np.array(end_hyps[n]['hyp'][1:][::-1]) for n in range(nbest)]]
                aws += [tensor2np(torch.cat(end_hyps[0]['aws'][1:][::-1], dim=2).squeeze(0))]
            else:
                nbest_hyps_idx += [[np.array(end_hyps[n]['hyp'][1:]) for n in range(nbest)]]
                aws += [tensor2np(torch.cat(end_hyps[0]['aws'][1:], dim=2).squeeze(0))]
            scores += [[end_hyps[n]['score_attn'] for n in range(nbest)]]

            # Check <eos>
            eos_flags.append([(end_hyps[n]['hyp'][-1] == self.eos) for n in range(nbest)])

        # Exclude <eos> (<sos> in case of the backward decoder)
        if exclude_eos:
            if self.bwd:
                nbest_hyps_idx = [[nbest_hyps_idx[b][n][1:] if eos_flags[b][n]
                                   else nbest_hyps_idx[b][n] for n in range(nbest)] for b in range(bs)]
            else:
                nbest_hyps_idx = [[nbest_hyps_idx[b][n][:-1] if eos_flags[b][n]
                                   else nbest_hyps_idx[b][n] for n in range(nbest)] for b in range(bs)]

        # Store ASR/LM state
        if len(end_hyps) > 0:
            self.lmstate_final = end_hyps[0]['lmstate']

        return nbest_hyps_idx, aws, scores
