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
from neural_sp.models.modules.initialization import init_with_normal_dist
from neural_sp.models.modules.positinal_embedding import PositionalEncoding
from neural_sp.models.modules.positinal_embedding import XLPositionalEmbedding
from neural_sp.models.modules.transformer import TransformerDecoderBlock
from neural_sp.models.seq2seq.decoders.beam_search import BeamSearch
from neural_sp.models.seq2seq.decoders.ctc import CTC
from neural_sp.models.seq2seq.decoders.ctc import CTCPrefixScore
from neural_sp.models.seq2seq.decoders.decoder_base import DecoderBase
from neural_sp.models.torch_utils import append_sos_eos
from neural_sp.models.torch_utils import compute_accuracy
from neural_sp.models.torch_utils import make_pad_mask
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
        dropout_layer (float): LayerDrop probability for layers
        dropout_head (float): HeadDrop probability for attention heads
        lsm_prob (float): label smoothing probability
        ctc_weight (float):
        ctc_lsm_prob (float): label smoothing probability for CTC
        ctc_fc_list (list):
        backward (bool): decode in the backward order
        global_weight (float):
        mtl_per_batch (bool):
        param_init (str): parameter initialization method
        memory_transformer (bool): TransformerXL decoder
        mem_len (int):
        mocha_chunk_size (int):
        mocha_n_heads_mono (int):
        mocha_n_heads_chunk (int):
        mocha_init_r (int):
        mocha_eps (float):
        mocha_std (float):
        mocha_no_denominator (bool):
        mocha_1dconv (bool): 1dconv for MoChA
        mocha_quantity_loss_weight (float):
        mocha_head_divergence_loss_weight (float):
        latency_metric (str):
        latency_loss_weight (float):
        mocha_first_layer (int):
        external_lm (RNNLM):
        lm_fusion (str):

    """

    def __init__(self, special_symbols,
                 enc_n_units, attn_type, n_heads, n_layers, d_model, d_ff,
                 pe_type, layer_norm_eps, ffn_activation,
                 vocab, tie_embedding,
                 dropout, dropout_emb, dropout_att, dropout_layer, dropout_head,
                 lsm_prob, ctc_weight, ctc_lsm_prob, ctc_fc_list, backward,
                 global_weight, mtl_per_batch, param_init,
                 memory_transformer, mem_len,
                 mocha_chunk_size, mocha_n_heads_mono, mocha_n_heads_chunk,
                 mocha_init_r, mocha_eps, mocha_std,
                 mocha_no_denominator, mocha_1dconv,
                 mocha_quantity_loss_weight, mocha_head_divergence_loss_weight,
                 latency_metric, latency_loss_weight,
                 mocha_first_layer, external_lm, lm_fusion, d_ff_bottleneck_dim):

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

        # for TransformerXL decoder
        self.memory_transformer = memory_transformer
        self.mem_len = mem_len
        if memory_transformer:
            assert pe_type == 'none'

        # for attention plot
        self.aws_dict = {}
        self.data_dict = {}

        # for MMA
        self.attn_type = attn_type
        self.quantity_loss_weight = mocha_quantity_loss_weight
        self._quantity_loss_weight = 0  # for curriculum
        self.mocha_first_layer = mocha_first_layer

        self.headdiv_loss_weight = mocha_head_divergence_loss_weight
        self.latency_metric = latency_metric
        self.latency_loss_weight = latency_loss_weight
        self.ctc_trigger = (self.latency_metric in ['ctc_sync'])
        if self.ctc_trigger:
            assert 0 < self.ctc_weight < 1

        if ctc_weight > 0:
            self.ctc = CTC(eos=self.eos,
                           blank=self.blank,
                           enc_n_units=enc_n_units,
                           vocab=self.vocab,
                           dropout=dropout,
                           lsm_prob=ctc_lsm_prob,
                           fc_list=ctc_fc_list,
                           param_init=0.1,
                           backward=backward)

        if ctc_weight < global_weight:
            # token embedding
            self.embed = nn.Embedding(self.vocab, d_model, padding_idx=self.pad)
            self.pos_enc = PositionalEncoding(d_model, dropout_emb, pe_type, param_init)
            # positional embedding
            self.u = None
            self.v = None
            if memory_transformer:
                self.scale = math.sqrt(d_model)  # for token embedding
                self.dropout_emb = nn.Dropout(p=dropout_emb)  # for token embedding
                self.pos_emb = XLPositionalEmbedding(d_model, dropout_emb)
                if self.mem_len > 0:
                    self.u = nn.Parameter(torch.Tensor(self.n_heads, self.d_model // self.n_heads))
                    self.v = nn.Parameter(torch.Tensor(self.n_heads, self.d_model // self.n_heads))
                    # NOTE: u and v are global parameters
            # self-attention
            self.layers = nn.ModuleList([copy.deepcopy(TransformerDecoderBlock(
                d_model, d_ff, attn_type, n_heads, dropout, dropout_att, dropout_layer,
                layer_norm_eps, ffn_activation, param_init,
                src_tgt_attention=False if lth < mocha_first_layer - 1 else True,
                memory_transformer=memory_transformer,
                mocha_chunk_size=mocha_chunk_size,
                mocha_n_heads_mono=mocha_n_heads_mono,
                mocha_n_heads_chunk=mocha_n_heads_chunk,
                mocha_init_r=mocha_init_r,
                mocha_eps=mocha_eps,
                mocha_std=mocha_std,
                mocha_no_denominator=mocha_no_denominator,
                mocha_1dconv=mocha_1dconv,
                dropout_head=dropout_head,
                lm_fusion=lm_fusion,
                d_ff_bottleneck_dim=d_ff_bottleneck_dim)) for lth in range(n_layers)])
            self.norm_out = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.output = nn.Linear(d_model, self.vocab)
            if tie_embedding:
                self.output.weight = self.embed.weight

            self.lm = external_lm
            if external_lm is not None:
                self.lm_output_proj = nn.Linear(external_lm.output_dim, d_model)

            self.reset_parameters(param_init)

    @staticmethod
    def add_args(parser, args):
        """Add arguments."""
        group = parser.add_argument_group("Transformer decoder")
        # Transformer common
        if not hasattr(args, 'transformer_d_model'):
            group.add_argument('--transformer_d_model', type=int, default=256,
                               help='number of units in the MHA layer')
            group.add_argument('--transformer_d_ff', type=int, default=2048,
                               help='number of units in the FFN layer')
            group.add_argument('--transformer_d_ff_bottleneck_dim', type=int, default=0,
                               help='bottleneck dimension in the FFN layer')
            group.add_argument('--transformer_n_heads', type=int, default=4,
                               help='number of heads in the MHA layer')
            group.add_argument('--transformer_layer_norm_eps', type=float, default=1e-12,
                               help='epsilon value for layer normalization')
            group.add_argument('--transformer_ffn_activation', type=str, default='relu',
                               choices=['relu', 'gelu', 'gelu_accurate', 'glu', 'swish'],
                               help='nonlinear activation for the FFN layer')
            group.add_argument('--transformer_param_init', type=str, default='xavier_uniform',
                               choices=['xavier_uniform', 'pytorch'],
                               help='parameter initializatin')
        # Transformer decoder specific
        group.add_argument('--transformer_attn_type', type=str, default='scaled_dot',
                           choices=['scaled_dot', 'add', 'average',
                                    'mocha', 'sync_bidir', 'sync_bidir_half'],
                           help='type of attention mechasnism for Transformer decoder')
        group.add_argument('--transformer_dec_pe_type', type=str, default='add',
                           choices=['add', 'concat', 'none', '1dconv3L'],
                           help='type of positional encoding for the Transformer decoder')
        group.add_argument('--dropout_dec_layer', type=float, default=0.0,
                           help='LayerDrop probability for Transformer decoder layers')
        group.add_argument('--dropout_head', type=float, default=0.0,
                           help='HeadDrop probability for masking out a head in the Transformer decoder')
        # streaming
        group.add_argument('--mocha_first_layer', type=int, default=1,
                           help='the initial layer to have a MMA function')
        group.add_argument('--mocha_head_divergence_loss_weight', type=float, default=0.0,
                           help='head divergence loss weight for MMA')

        return parser

    def reset_parameters(self, param_init):
        """Initialize parameters."""
        if self.memory_transformer:
            logger.info('===== Initialize %s with normal distribution =====' % self.__class__.__name__)
            for n, p in self.named_parameters():
                if 'conv' in n:
                    continue
                init_with_normal_dist(n, p, std=0.02)

        elif param_init == 'xavier_uniform':
            logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
            # see https://github.com/pytorch/fairseq/blob/master/fairseq/models/transformer.py
            # embedding
            nn.init.normal_(self.embed.weight, mean=0., std=self.d_model**-0.5)
            nn.init.constant_(self.embed.weight[self.pad], 0.)
            # output layer
            nn.init.xavier_uniform_(self.output.weight)
            nn.init.constant_(self.output.bias, 0.)

    def init_memory(self):
        """Initialize memory."""
        if self.device_id >= 0:
            return [torch.empty(0, dtype=torch.float).cuda(self.device_id)
                    for _ in range(self.n_layers)]
        else:
            return [torch.empty(0, dtype=torch.float)
                    for _ in range(self.n_layers)]

    def update_memory(self, memory_prev, hidden_states):
        """Update memory.

        Args:
            memory_prev (list): length `n_layers`, each of which contains [B, L_prev, d_model]`
            hidden_states (list): length `n_layers`, each of which contains [B, L, d_model]`
        Returns:
            new_mems (list): length `n_layers`, each of which contains `[B, mlen, d_model]`

        """
        # if memory_prev is None:
        #     memory_prev = self.init_memory()  # 0-th to L-1-th layer
        assert len(hidden_states) == len(memory_prev)
        mlen = memory_prev[0].size(1) if memory_prev[0].dim() > 1 else 0
        qlen = hidden_states[0].size(1)

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + qlen
            start_idx = max(0, end_idx - self.mem_len)
            for m, h in zip(memory_prev, hidden_states):
                cat = torch.cat([m, h], dim=1)  # `[B, mlen + qlen, d_model]`
                new_mems.append(cat[:, start_idx:end_idx].detach())  # `[B, self.mem_len, d_model]`

        return new_mems

    def forward(self, eouts, elens, ys, task='all',
                teacher_logits=None, recog_params={}, idx2token=None):
        """Forward computation.

        Args:
            eouts (FloatTensor): `[B, T, d_model]`
            elens (IntTensor): `[B]`
            ys (list): length `B`, each of which contains a list of size `[L]`
            task (str): all/ys*/ys_sub*
            teacher_logits (FloatTensor): `[B, L, vocab]`
            recog_params (dict): parameters for MBR training
            idx2token ():
        Returns:
            loss (FloatTensor): `[1]`
            observation (dict):

        """
        observation = {'loss': None, 'loss_att': None, 'loss_ctc': None, 'loss_mbr': None,
                       'acc_att': None, 'ppl_att': None}
        loss = eouts.new_zeros((1,))

        # CTC loss
        trigger_points = None
        if self.ctc_weight > 0 and (task == 'all' or 'ctc' in task):
            forced_align = (self.ctc_trigger and self.training) or self.attn_type == 'triggered_attention'
            loss_ctc, trigger_points = self.ctc(eouts, elens, ys, forced_align=forced_align)
            observation['loss_ctc'] = loss_ctc.item()
            if self.mtl_per_batch:
                loss += loss_ctc
            else:
                loss += loss_ctc * self.ctc_weight

        # XE loss
        if self.global_weight - self.ctc_weight > 0 and (task == 'all' or 'ctc' not in task):
            loss_att, acc_att, ppl_att, losses_auxiliary = self.forward_att(
                eouts, elens, ys, trigger_points=trigger_points)
            observation['loss_att'] = loss_att.item()
            observation['acc_att'] = acc_att
            observation['ppl_att'] = ppl_att
            if 'mocha' in self.attn_type:
                if self._quantity_loss_weight > 0:
                    loss_att += losses_auxiliary['loss_quantity'] * self._quantity_loss_weight
                observation['loss_quantity'] = losses_auxiliary['loss_quantity'].item()
            if self.headdiv_loss_weight > 0:
                loss_att += losses_auxiliary['loss_headdiv'] * self.headdiv_loss_weight
                observation['loss_headdiv'] = losses_auxiliary['loss_headdiv'].item()
            if self.latency_metric:
                observation['loss_latency'] = losses_auxiliary['loss_latency'].item() if self.training else 0
                if self.latency_metric != 'decot' and self.latency_loss_weight > 0:
                    loss_att += losses_auxiliary['loss_latency'] * self.latency_loss_weight
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
            ys (list): length `B`, each of which contains a list of size `[L]`
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
        xmax = eouts.size(1)
        bs, ymax = ys_in.size()[:2]
        mlen = 0
        tgt_mask = (ys_out != self.pad).unsqueeze(1).repeat([1, ymax, 1])
        causal_mask = tgt_mask.new_ones(ymax, ymax).byte()
        causal_mask = torch.tril(causal_mask, diagonal=0 + mlen, out=causal_mask).unsqueeze(0)
        tgt_mask = tgt_mask & causal_mask  # `[B, L (query), L (key)]`

        # Create source-target mask
        src_mask = make_pad_mask(elens, self.device_id).unsqueeze(1).repeat([1, ymax, 1])  # `[B, L, T]`

        # external LM integration
        lmout = None
        if self.lm is not None:
            self.lm.eval()
            with torch.no_grad():
                lmout, lmstate, _ = self.lm.predict(ys_in, None)
            lmout = self.lm_output_proj(lmout)

        out = self.pos_enc(self.embed(ys_in))  # scaled

        mems = self.init_memory()
        pos_embs = None
        if self.memory_transformer:
            out = self.dropout_emb(out)
            # NOTE: TransformerXL does not use positional encoding in the token embedding
            # adopt zero-centered offset
            pos_idxs = torch.arange(mlen - 1, -ymax - 1, -1.0, dtype=torch.float)
            pos_embs = self.pos_emb(pos_idxs, self.device_id)

        hidden_states = [out]
        xy_aws_layers = []
        for lth, (mem, layer) in enumerate(zip(mems, self.layers)):
            out, yy_aws, xy_aws, xy_aws_beta, yy_aws_lm = layer(
                out, tgt_mask, eouts, src_mask, mode='parallel', lmout=lmout,
                pos_embs=pos_embs, memory=mem, u=self.u, v=self.v)
            if lth < self.n_layers - 1:
                hidden_states.append(out)
                # NOTE: outputs from the last layer is not used for momory
            # Attention padding
            if xy_aws is not None and 'mocha' in self.attn_type:
                tgt_mask_v2 = (ys_out != self.pad).unsqueeze(1).unsqueeze(3)  # `[B, 1, L, 1]`
                xy_aws = xy_aws.masked_fill_(tgt_mask_v2.repeat([1, xy_aws.size(1), 1, xmax]) == 0, 0)
                # NOTE: attention padding is quite effective for quantity loss
                xy_aws_layers.append(xy_aws.clone())
            if not self.training:
                if yy_aws is not None:
                    self.aws_dict['yy_aws_layer%d' % lth] = tensor2np(yy_aws)
                if xy_aws is not None:
                    self.aws_dict['xy_aws_layer%d' % lth] = tensor2np(xy_aws)
                if xy_aws_beta is not None:
                    self.aws_dict['xy_aws_beta_layer%d' % lth] = tensor2np(xy_aws_beta)
                if yy_aws_lm is not None:
                    self.aws_dict['yy_aws_lm_layer%d' % lth] = tensor2np(yy_aws_lm)
        logits = self.output(self.norm_out(out))

        # for knowledge distillation
        if return_logits:
            return logits

        # Compute XE loss (+ label smoothing)
        loss, ppl = cross_entropy_lsm(logits, ys_out, self.lsm_prob, self.pad, self.training)
        losses_auxiliary = {}

        # Quantity loss
        losses_auxiliary['loss_quantity'] = 0.
        if 'mocha' in self.attn_type:
            # Average over all heads across all layers
            n_tokens_ref = tgt_mask[:, -1, :].sum(1).float()  # `[B]`
            # NOTE: count <eos> tokens
            n_tokens_pred = sum([torch.abs(aws.sum(3).sum(2).sum(1) / aws.size(1))
                                 for aws in xy_aws_layers])  # `[B]`
            n_tokens_pred /= len(xy_aws_layers)
            losses_auxiliary['loss_quantity'] = torch.mean(torch.abs(n_tokens_pred - n_tokens_ref))

        # Compute token-level accuracy in teacher-forcing
        acc = compute_accuracy(logits, ys_out, self.pad)

        return loss, acc, ppl, losses_auxiliary

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
            n_cols_tmp = 1 if n_heads == 1 else n_cols * max(1, n_heads // 4)
            fig, axes = plt.subplots(max(1, n_heads // n_cols_tmp), n_cols_tmp,
                                     figsize=(20 * max(1, n_heads // 4), 8), squeeze=False)
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
               exclude_eos=False, refs_id=None, utt_ids=None, speakers=None,
               cache_states=True):
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
            cache_states (bool):
        Returns:
            hyps (list): length `B`, each of which contains arrays of size `[L]`
            aw (list): length `B`, each of which contains arrays of size `[L, T]`

        """
        bs, xtime = eouts.size()[:2]
        ys = eouts.new_zeros(bs, 1).fill_(self.eos).long()

        cache = [None] * self.n_layers

        hyps_batch = []
        ylens = torch.zeros(bs).int()
        eos_flags = [False] * bs
        ymax = int(math.floor(xtime * max_len_ratio)) + 1
        for t in range(ymax):
            causal_mask = eouts.new_ones(t + 1, t + 1).byte()
            causal_mask = torch.tril(causal_mask, out=causal_mask).unsqueeze(0)

            new_cache = [None] * self.n_layers
            out = self.pos_enc(self.embed(ys))  # scaled
            for lth, layer in enumerate(self.layers):
                out, _, xy_aws, _, _ = layer(out, causal_mask, eouts, None, cache=cache[lth])
                new_cache[lth] = out

            if cache_states:
                cache = new_cache[:]

            # Pick up 1-best
            y = self.output(self.norm_out(out))[:, -1:].argmax(-1)
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
            if t == ymax - 1:
                break

            ys = torch.cat([ys, y], dim=-1)

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
                    ensmbl_eouts=None, ensmbl_elens=None, ensmbl_decs=[], cache_states=True):
        """Beam search decoding.

        Args:
            eouts (FloatTensor): `[B, T, d_model]`
            elens (IntTensor): `[B]`
            params (dict): hyperparameters for decoding
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
            cache_states (bool): cache decoder states for fast decoding
        Returns:
            nbest_hyps_idx (list): length `B`, each of which contains list of N hypotheses
            aws (list): length `B`, each of which contains arrays of size `[H, L, T]`
            scores (list):

        """
        bs, xmax, _ = eouts.size()
        n_models = len(ensmbl_decs) + 1

        beam_width = params['recog_beam_width']
        assert 1 <= nbest <= beam_width
        ctc_weight = params['recog_ctc_weight']
        max_len_ratio = params['recog_max_len_ratio']
        min_len_ratio = params['recog_min_len_ratio']
        lp_weight = params['recog_length_penalty']
        length_norm = params['recog_length_norm']
        lm_weight = params['recog_lm_weight']
        lm_weight_second = params['recog_lm_second_weight']
        lm_weight_bwd = params['recog_lm_bwd_weight']
        eos_threshold = params['recog_eos_threshold']
        lm_state_carry_over = params['recog_lm_state_carry_over']
        softmax_smoothing = params['recog_softmax_smoothing']
        eps_wait = params['recog_mma_delay_threshold']

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
            ys = eouts.new_zeros(1, 1).fill_(self.eos).long()

            # For joint CTC-Attention decoding
            ctc_prefix_scorer = None
            if ctc_log_probs is not None:
                if self.bwd:
                    ctc_prefix_scorer = CTCPrefixScore(ctc_log_probs[b][::-1], self.blank, self.eos)
                else:
                    ctc_prefix_scorer = CTCPrefixScore(ctc_log_probs[b], self.blank, self.eos)

            if speakers is not None:
                if speakers[b] == self.prev_spk:
                    if lm_state_carry_over and isinstance(lm, RNNLM):
                        lmstate = self.lmstate_final
                self.prev_spk = speakers[b]

            helper = BeamSearch(beam_width, self.eos, ctc_weight, self.device_id)

            end_hyps = []
            ymax = int(math.floor(elens[b] * max_len_ratio)) + 1
            hyps = [{'hyp': [self.eos],
                     'ys': ys,
                     'cache': None,
                     'score': 0.,
                     'score_attn': 0.,
                     'score_ctc': 0.,
                     'score_lm': 0.,
                     'aws': [None],
                     'lmstate': lmstate,
                     'ensmbl_aws':[[None]] * (n_models - 1),
                     'ctc_state': ctc_prefix_scorer.initial_state() if ctc_prefix_scorer is not None else None,
                     'streamable': True,
                     'streaming_failed_point': 1000}]
            streamable_global = True
            for t in range(ymax):
                # batchfy all hypotheses for batch decoding
                cache = [None] * self.n_layers
                if cache_states and t > 0:
                    for lth in range(self.n_layers):
                        cache[lth] = torch.cat([beam['cache'][lth] for beam in hyps], dim=0)
                ys = eouts.new_zeros(len(hyps), t + 1).long()
                for j, beam in enumerate(hyps):
                    ys[j, :] = beam['ys']
                if t > 0:
                    xy_aws_prev = torch.cat([beam['aws'][-1] for beam in hyps], dim=0)
                    # `[B, n_layers, H_mono, 1, klen]`
                else:
                    xy_aws_prev = None

                # Update LM states for shallow fusion
                lmstate, scores_lm = None, None
                if lm is not None:
                    if hyps[0]['lmstate'] is not None:
                        lm_hxs = torch.cat([beam['lmstate']['hxs'] for beam in hyps], dim=1)
                        lm_cxs = torch.cat([beam['lmstate']['cxs'] for beam in hyps], dim=1)
                        lmstate = {'hxs': lm_hxs, 'cxs': lm_cxs}
                    y = ys[:, -1:].clone()  # NOTE: this is important
                    _, lmstate, scores_lm = lm.predict(y, lmstate)

                # for the main model
                causal_mask = eouts.new_ones(t + 1, t + 1).byte()
                causal_mask = torch.tril(causal_mask, out=causal_mask).unsqueeze(0).repeat([ys.size(0), 1, 1])

                out = self.pos_enc(self.embed(ys))  # scaled

                mlen = 0  # TODO: fix later
                if self.memory_transformer:
                    # NOTE: TransformerXL does not use positional encoding in the token embedding
                    mems = self.init_memory()
                    # adopt zero-centered offset
                    pos_idxs = torch.arange(mlen - 1, -(t + 1) - 1, -1.0, dtype=torch.float)
                    pos_embs = self.pos_emb(pos_idxs, self.device_id)
                    out = self.dropout_emb(out)
                    hidden_states = [out]

                n_heads_total = 0
                eouts_b = eouts[b:b + 1, :elens[b]].repeat([ys.size(0), 1, 1])
                new_cache = [None] * self.n_layers
                xy_aws_all_layers = []
                xy_aws = None
                lth_s = self.mocha_first_layer - 1
                for lth, layer in enumerate(self.layers):
                    if self.memory_transformer:
                        out, _, xy_aws, _, _ = layer(
                            out, causal_mask, eouts_b, None,
                            cache=cache[lth],
                            pos_embs=pos_embs, memory=mems[lth], u=self.u, v=self.v)
                        hidden_states.append(out)
                    else:
                        out, _, xy_aws, _, _ = layer(
                            out, causal_mask, eouts_b, None,
                            cache=cache[lth],
                            xy_aws_prev=xy_aws_prev[:, lth - lth_s] if lth >= lth_s and t > 0 else None,
                            eps_wait=eps_wait)

                    new_cache[lth] = out
                    if xy_aws is not None:
                        xy_aws_all_layers.append(xy_aws)
                logits = self.output(self.norm_out(out))
                probs = torch.softmax(logits[:, -1] * softmax_smoothing, dim=1)
                xy_aws_all_layers = torch.stack(xy_aws_all_layers, dim=1)  # `[B, H, n_layers, L, T]`

                # for the ensemble
                ensmbl_new_cache = []
                if n_models > 1:
                    # Ensemble initialization
                    # ensmbl_cache = []
                    # cache_e = [None] * self.n_layers
                    # if cache_states and t > 0:
                    #     for l in range(self.n_layers):
                    #         cache_e[l] = torch.cat([beam['ensmbl_cache'][l] for beam in hyps], dim=0)
                    for i_e, dec in enumerate(ensmbl_decs):
                        out_e = dec.pos_enc(dec.embed(ys))  # scaled
                        eouts_e = ensmbl_eouts[i_e][b:b + 1, :elens[b]].repeat([ys.size(0), 1, 1])
                        new_cache_e = [None] * dec.n_layers
                        for l in range(dec.n_layers):
                            out_e, _, xy_aws_e, _, _ = dec.layers[l](out_e, causal_mask, eouts_e, None,
                                                                     cache=cache[lth])
                            new_cache_e[l] = out_e
                        ensmbl_new_cache.append(new_cache_e)
                        logits_e = dec.output(dec.norm_out(out_e))
                        probs += torch.softmax(logits_e[:, -1] * softmax_smoothing, dim=1)
                        # NOTE: sum in the probability scale (not log-scale)

                # Ensemble in log-scale
                scores_attn = torch.log(probs) / n_models

                new_hyps = []
                for j, beam in enumerate(hyps):
                    # Attention scores
                    total_scores_attn = beam['score_attn'] + scores_attn[j:j + 1]
                    total_scores = total_scores_attn * (1 - ctc_weight)

                    # Add LM score <before> top-K selection
                    if lm is not None:
                        total_scores_lm = beam['score_lm'] + scores_lm[j:j + 1, -1]
                        total_scores += total_scores_lm * lm_weight
                    else:
                        total_scores_lm = eouts.new_zeros(1, self.vocab)

                    total_scores_topk, topk_ids = torch.topk(
                        total_scores, k=beam_width, dim=1, largest=True, sorted=True)

                    # Add length penalty
                    if lp_weight > 0:
                        total_scores_topk += (len(beam['hyp'][1:]) + 1) * lp_weight

                    # Add CTC score
                    new_ctc_states, total_scores_ctc, total_scores_topk = helper.add_ctc_score(
                        beam['hyp'], topk_ids, beam['ctc_state'],
                        total_scores_topk, ctc_prefix_scorer)

                    new_aws = beam['aws'] + [xy_aws_all_layers[j:j + 1, :, :, -1:]]
                    aws_j = torch.cat(new_aws[1:], dim=3)  # `[1, H, n_layers, L, T]`
                    streaming_failed_point = beam['streaming_failed_point']

                    # forward direction
                    for k in range(beam_width):
                        idx = topk_ids[0, k].item()
                        length_norm_factor = len(beam['hyp'][1:]) + 1 if length_norm else 1
                        total_scores_topk /= length_norm_factor

                        if idx == self.eos:
                            # Exclude short hypotheses
                            if len(beam['hyp']) - 1 < elens[b] * min_len_ratio:
                                continue
                            # EOS threshold
                            max_score_no_eos = scores_attn[j, :idx].max(0)[0].item()
                            max_score_no_eos = max(max_score_no_eos, scores_attn[j, idx + 1:].max(0)[0].item())
                            if scores_attn[j, idx].item() <= eos_threshold * max_score_no_eos:
                                continue

                        quantity_rate = 1.
                        if 'mocha' in self.attn_type:
                            n_tokens_hyp_k = t + 1
                            n_quantity_k = aws_j[:, :, :, :n_tokens_hyp_k].int().sum().item()
                            quantity_diff = n_tokens_hyp_k * n_heads_total - n_quantity_k

                            if quantity_diff != 0:
                                if idx == self.eos:
                                    n_tokens_hyp_k -= 1  # NOTE: do not count <eos> for streamability
                                    n_quantity_k = aws_j[:, :, :, :n_tokens_hyp_k].int().sum().item()
                                else:
                                    streamable_global = False
                                if n_tokens_hyp_k * n_heads_total == 0:
                                    quantity_rate = 0
                                else:
                                    quantity_rate = n_quantity_k / (n_tokens_hyp_k * n_heads_total)

                            if beam['streamable'] and not streamable_global:
                                streaming_failed_point = t

                        new_hyps.append(
                            {'hyp': beam['hyp'] + [idx],
                             'ys': torch.cat([beam['ys'], eouts.new_zeros(1, 1).fill_(idx).long()], dim=-1),
                             'cache': [new_cache_l[j:j + 1] for new_cache_l in new_cache] if cache_states else cache,
                             'score': total_scores_topk[0, k].item(),
                             'score_attn': total_scores_attn[0, idx].item(),
                             'score_ctc': total_scores_ctc[k].item(),
                             'score_lm': total_scores_lm[0, idx].item(),
                             'aws': new_aws,
                             'lmstate': {'hxs': lmstate['hxs'][:, j:j + 1],
                                         'cxs': lmstate['cxs'][:, j:j + 1]} if lmstate is not None else None,
                             'ctc_state': new_ctc_states[k] if ctc_prefix_scorer is not None else None,
                             'ensmbl_cache': ensmbl_new_cache,
                             'streamable': streamable_global,
                             'streaming_failed_point': streaming_failed_point,
                             'quantity_rate': quantity_rate})

                # Local pruning
                new_hyps_sorted = sorted(new_hyps, key=lambda x: x['score'], reverse=True)[:beam_width]

                # Remove complete hypotheses
                new_hyps, end_hyps, is_finish = helper.remove_complete_hyp(
                    new_hyps_sorted, end_hyps, prune=True)
                hyps = new_hyps[:]
                if is_finish:
                    break

            # Global pruning
            if len(end_hyps) == 0:
                end_hyps = hyps[:]
            elif len(end_hyps) < nbest and nbest > 1:
                end_hyps.extend(hyps[:nbest - len(end_hyps)])

            # forward second path LM rescoring
            if lm_second is not None:
                self.lm_rescoring(end_hyps, lm_second, lm_weight_second, tag='second')

            # backward secodn path LM rescoring
            if lm_bwd is not None and lm_weight_bwd > 0:
                self.lm_rescoring(end_hyps, lm_bwd, lm_weight_bwd, tag='second_bwd')

            # Sort by score
            end_hyps = sorted(end_hyps, key=lambda x: x['score'], reverse=True)

            for j in range(len(end_hyps[0]['aws'][1:])):
                tmp = end_hyps[0]['aws'][j + 1]
                end_hyps[0]['aws'][j + 1] = tmp.view(1, -1, tmp.size(-2), tmp.size(-1))

            # metrics for streaming infernece
            self.streamable = end_hyps[0]['streamable']
            self.quantity_rate = end_hyps[0]['quantity_rate']
            self.last_success_frame_ratio = None

            if idx2token is not None:
                if utt_ids is not None:
                    logger.info('Utt-id: %s' % utt_ids[b])
                assert self.vocab == idx2token.vocab
                logger.info('=' * 200)
                for k in range(len(end_hyps)):
                    if refs_id is not None:
                        logger.info('Ref: %s' % idx2token(refs_id[b]))
                    logger.info('Hyp: %s' % idx2token(
                        end_hyps[k]['hyp'][1:][::-1] if self.bwd else end_hyps[k]['hyp'][1:]))
                    logger.info('num tokens (hyp): %d' % len(end_hyps[k]['hyp'][1:]))
                    logger.info('log prob (hyp): %.7f' % end_hyps[k]['score'])
                    logger.info('log prob (hyp, att): %.7f' % (end_hyps[k]['score_attn'] * (1 - ctc_weight)))
                    if ctc_prefix_scorer is not None:
                        logger.info('log prob (hyp, ctc): %.7f' % (end_hyps[k]['score_ctc'] * ctc_weight))
                    if lm is not None:
                        logger.info('log prob (hyp, first-path lm): %.7f' % (end_hyps[k]['score_lm'] * lm_weight))
                    if lm_second is not None:
                        logger.info('log prob (hyp, second-path lm): %.7f' %
                                    (end_hyps[k]['score_lm_second'] * lm_weight_second))
                    if lm_bwd is not None:
                        logger.info('log prob (hyp, second-path lm-bwd): %.7f' %
                                    (end_hyps[k]['score_lm_second_bwd'] * lm_weight_bwd))
                    if 'mocha' in self.attn_type:
                        logger.info('streamable: %s' % end_hyps[k]['streamable'])
                        logger.info('streaming failed point: %d' % (end_hyps[k]['streaming_failed_point'] + 1))
                        logger.info('quantity rate [%%]: %.2f' % (end_hyps[k]['quantity_rate'] * 100))
                    logger.info('-' * 50)

                if 'mocha' in self.attn_type and end_hyps[0]['streaming_failed_point'] < 1000:
                    assert not self.streamable
                    aws_last_success = end_hyps[0]['aws'][1:][end_hyps[0]['streaming_failed_point'] - 1]
                    rightmost_frame = max(0, aws_last_success[0, :, 0].nonzero()[:, -1].max().item()) + 1
                    frame_ratio = rightmost_frame * 100 / xmax
                    self.last_success_frame_ratio = frame_ratio
                    logger.info('streaming last success frame ratio: %.2f' % frame_ratio)

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
