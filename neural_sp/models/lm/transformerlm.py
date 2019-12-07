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
from neural_sp.models.torch_utils import tensor2np
from neural_sp.utils import mkdir_join

import matplotlib
matplotlib.use('Agg')

random.seed(1)


class TransformerLM(LMBase):
    """Transformer language model."""

    def __init__(self, args, save_path=None):

        super(LMBase, self).__init__()
        logger = logging.getLogger('training')
        logger.info(self.__class__.__name__)

        self.save_path = save_path

        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.pe_type = args.pe_type
        self.n_layers = args.n_layers
        self.attn_n_heads = args.attn_n_heads
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
        self.pos_enc = PositionalEncoding(args.d_model, args.dropout_in, args.pe_type)

        self.layers = nn.ModuleList(
            [TransformerDecoderBlock(args.d_model, args.d_ff,
                                     args.attn_type, args.attn_n_heads,
                                     args.dropout_hidden, args.dropout_att, args.layer_norm_eps,
                                     src_tgt_attention=False)
             for _ in range(self.n_layers)])
        self.norm_out = nn.LayerNorm(args.d_model, eps=args.layer_norm_eps)

        if args.adaptive_softmax:
            self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
                args.d_model, self.vocab,
                cutoffs=[round(self.vocab / 15), 3 * round(self.vocab / 15)],
                # cutoffs=[self.vocab // 25, 3 * self.vocab // 5],
                div_value=4.0)
            self.output = None
        else:
            self.adaptive_softmax = None
            self.output = nn.Linear(self.d_model, self.vocab)
            if args.tie_embedding:
                self.output.weight = self.embed.weight

        # Initialize parameters
        # self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with xavier_uniform style."""
        logger = logging.getLogger('training')
        logger.info('===== Initialize %s =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0.)  # bias
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.))
            elif p.dim() == 2:
                if 'embed' in n:
                    nn.init.normal_(p, mean=0., std=self.d_model**-0.5)
                    logger.info('Initialize %s with %s / %.3f' % (n, 'normal', self.d_model**-0.5))
                else:
                    nn.init.xavier_uniform_(p, 1.0)
                    logger.info('Initialize %s with %s' % (n, 'xavier_uniform'))
            else:
                raise ValueError

    def decode(self, ys, cache=None, is_asr=False):
        """Decode function.

        Args:
            ys (FloatTensor): `[B, L]`
            cache: previous tokens
            is_asr (bool):
        Returns:
            logits (FloatTensor): `[B, L, vocab]`
            ys_emb (FloatTensor): `[B, L, d_model]` (for cache)
            cache: previous tokens

        """
        # Concatenate previous tokens
        if is_asr and cache is not None:
            ys = torch.cat([cache, ys], dim=1)
            # NOTE: this is used for ASR decoding

        # Create the self-attention mask
        bs, ymax = ys.size()[:2]
        ylens = torch.IntTensor([ymax] * bs)
        tgt_mask = make_pad_mask(ylens, self.device_id).unsqueeze(1).repeat([1, ymax, 1])
        subsequent_mask = tgt_mask.new_ones(ymax, ymax).byte()
        subsequent_mask = torch.tril(subsequent_mask, out=subsequent_mask).unsqueeze(0)
        tgt_mask = tgt_mask & subsequent_mask

        ys_emb = self.pos_enc(self.embed(ys.long()))
        for l in range(self.n_layers):
            ys_emb, yy_aws, _ = self.layers[l](ys_emb, tgt_mask)
            if not self.training:
                setattr(self, 'yy_aws_layer%d' % l, tensor2np(yy_aws))
        ys_emb = self.norm_out(ys_emb)
        if self.adaptive_softmax is None:
            logits = self.output(ys_emb)
        else:
            logits = ys_emb

        if is_asr:
            cache = ys

        return logits, ys_emb, cache

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
            fig, axes = plt.subplots(self.attn_n_heads // n_cols, n_cols, figsize=(20, 8))
            for h in range(self.attn_n_heads):
                if self.attn_n_heads > n_cols:
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
