#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer language model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import random
import shutil
import torch
import torch.nn as nn

from neural_sp.models.lm.lm_base import LMBase
from neural_sp.models.modules.transformer import PositionalEncoding
from neural_sp.models.modules.transformer import TransformerDecoderBlock
from neural_sp.models.torch_utils import make_pad_mask
from neural_sp.models.torch_utils import repeat
from neural_sp.models.torch_utils import tensor2np
from neural_sp.utils import mkdir_join

import matplotlib
matplotlib.use('Agg')

random.seed(1)

logger = logging.getLogger(__name__)


class TransformerLM(LMBase):
    """Transformer language model."""

    def __init__(self, args, save_path=None):

        super(LMBase, self).__init__()
        logger.info(self.__class__.__name__)

        self.save_path = save_path

        self.d_model = args.transformer_d_model
        self.n_layers = args.n_layers
        self.n_heads = args.transformer_n_heads
        self.lsm_prob = args.lsm_prob

        self.vocab = args.vocab
        self.eos = 2
        self.pad = 3
        # NOTE: reserved in advance

        # for cache
        self.cache_theta = 0.2  # smoothing parameter
        self.cache_lambda = 0.2  # cache weight
        self.cache_ids = []
        self.cache_keys = []
        self.cache_attn = []

        self.embed = nn.Embedding(self.vocab, self.d_model, padding_idx=self.pad)
        self.pos_enc = PositionalEncoding(self.d_model, args.dropout_in, args.transformer_pe_type)
        self.layers = repeat(TransformerDecoderBlock(
            self.d_model, args.transformer_d_ff,
            args.transformer_attn_type, self.n_heads,
            args.dropout_hidden, args.dropout_att,
            args.transformer_layer_norm_eps, args.transformer_ffn_activation,
            args.transformer_param_init,
            src_tgt_attention=False), self.n_layers)
        self.norm_out = nn.LayerNorm(self.d_model, eps=args.transformer_layer_norm_eps)

        self.adaptive_softmax = None
        self.output = None
        if args.adaptive_softmax:
            self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
                self.d_model, self.vocab,
                cutoffs=[round(self.vocab / 15), 3 * round(self.vocab / 15)],
                # cutoffs=[self.vocab // 25, 3 * self.vocab // 5],
                div_value=4.0)
        else:
            self.output = nn.Linear(self.d_model, self.vocab)
            if args.tie_embedding:
                self.output.weight = self.embed.weight

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with Xavier uniform distribution."""
        logging.info('===== Initialize %s =====' % self.__class__.__name__)
        # see https://github.com/pytorch/fairseq/blob/master/fairseq/models/transformer.py
        # embedding
        nn.init.normal_(self.embed.weight, mean=0., std=self.d_model**-0.5)
        nn.init.constant_(self.embed.weight[self.pad], 0)
        # output layer
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.constant_(self.output.bias, 0.)

    def decode(self, ys, ys_prev=None, cache=False):
        """Decode function.

        Args:
            ys (LongTensor): `[B, L]`
            ys_prev (LongTensor): previous tokens
            cahce (bool): concatenate previous tokens
        Returns:
            logits (FloatTensor): `[B, L, vocab]`
            ys_emb (FloatTensor): `[B, L, d_model]` (for ys_prev)
            ys_prev (LongTensor): previous tokens

        """
        # Concatenate previous tokens
        if cache and ys_prev is not None:
            ys = torch.cat([ys_prev, ys], dim=1)
            # NOTE: this is used for ASR decoding

        # Create the self-attention mask
        bs, ymax = ys.size()[:2]
        ylens = torch.IntTensor([ymax] * bs)
        tgt_mask = make_pad_mask(ylens, self.device_id).unsqueeze(1).repeat([1, ymax, 1])
        causal_mask = tgt_mask.new_ones(ymax, ymax).byte()
        causal_mask = torch.tril(causal_mask, out=causal_mask).unsqueeze(0)
        tgt_mask = tgt_mask & causal_mask

        out = self.pos_enc(self.embed(ys.long()))
        for l in range(self.n_layers):
            out, yy_aws, _, _ = self.layers[l](out, tgt_mask)
            if not self.training:
                setattr(self, 'yy_aws_layer%d' % l, tensor2np(yy_aws))
        out = self.norm_out(out)
        if self.adaptive_softmax is None:
            logits = self.output(out)
        else:
            logits = out

        return logits, out, ys

    def plot_attention(self, n_cols=4):
        """Plot attention for each head in all layers."""
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator

        save_path = mkdir_join(self.save_path, 'att_weights')

        # Clean directory
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)

        for l in range(self.n_layers):
            if not hasattr(self, 'yy_aws_layer%d' % l):
                continue

            yy_aws = getattr(self, 'yy_aws_layer%d' % l)

            plt.clf()
            fig, axes = plt.subplots(self.n_heads // n_cols, n_cols, figsize=(20, 8))
            for h in range(self.n_heads):
                if self.n_heads > n_cols:
                    ax = axes[h // n_cols, h % n_cols]
                else:
                    ax = axes[h]
                ax.imshow(yy_aws[-1, h, :, :], aspect="auto")
                ax.grid(False)
                ax.set_xlabel("Input (head%d)" % h)
                ax.set_ylabel("Output (head%d)" % h)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            fig.tight_layout()
            fig.savefig(os.path.join(save_path, 'layer%d.png' % (l)), dvi=500)
            plt.close()
