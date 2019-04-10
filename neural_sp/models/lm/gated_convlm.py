#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Gated convolutional neural network language model with Gated Linear Units (GLU)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.base import ModelBase
from neural_sp.models.model_utils import Embedding
from neural_sp.models.model_utils import LinearND
from neural_sp.models.torch_utils import compute_accuracy
from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list


class GLUblock(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super().__init__()

        self.pad_left = nn.ConstantPad1d((kernel_size - 1, 0), 0)
        self.conv = nn.utils.weight_norm(nn.Conv1d(in_channels=in_ch,
                                                   out_channels=out_ch,
                                                   kernel_size=kernel_size),
                                         name='weight')
        self.bias_b = nn.Parameter(torch.zeros(out_ch, 1))
        self.conv_gate = nn.utils.weight_norm(nn.Conv1d(in_channels=in_ch,
                                                        out_channels=out_ch,
                                                        kernel_size=kernel_size),
                                              name='weight')
        self.bias_c = nn.Parameter(torch.zeros(out_ch, 1))

    def forward(self, x):
        """Forward computation.
        Args:
            x (FloatTensor): `[B, in_ch, T]`
        Returns:
            out (FloatTensor): `[B, out_ch, T]`

        """
        residual = x
        x = self.pad_left(x)  # `[B, embed_dim, T+kernel-1]`
        a = self.conv(x) + self.bias_b  # a: `[B, out_ch, T]`
        b = self.conv_gate(x) + self.bias_c  # b: `[B, out_ch, T]`
        x = torch.mul(a, F.sigmoid(b))
        return x + residual


class GatedConvLM(ModelBase):
    """Gated convolutional neural network language model with Gated Linear Units (GLU)."""

    def __init__(self, args):

        super(ModelBase, self).__init__()

        self.emb_dim = args.emb_dim
        self.n_units = args.n_units
        self.n_layers = args.n_layers
        self.tie_embedding = args.tie_embedding
        self.backward = args.backward

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

        args.kernel_size = 5  # 3, 4, 5
        # TODO(hirofumi): make this hyperparameter
        args.channels = [args.n_units] * args.n_layers

        self.embed = Embedding(vocab=self.vocab,
                               emb_dim=args.emb_dim,
                               dropout=args.dropout_emb,
                               ignore_index=self.pad)

        self.glu0 = GLUblock(args.kernel_size, args.emb_dim, args.channels[0])
        glu_layers = OrderedDict()
        for l in range(1, args.n_layers):
            glu_layers['glu%s' % l] = GLUblock(args.kernel_size, args.channels[l - 1], args.channels[l])
        self.glu_layers = nn.Sequential(glu_layers)
        # self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
        #     args.n_units, vs, cutoffs=[round(vs / 25), round(vs / 5)], div_value=4)

        self.output = LinearND(args.channels[-1], self.vocab,
                               dropout=args.dropout_out)
        # NOTE: include bias even when tying weights

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if args.tie_embedding:
            if args.n_units != args.emb_dim:
                raise ValueError('When using the tied flag, n_units must be equal to emb_dim.')
            self.output.fc.weight = self.embed.embed.weight

        # Initialize weight matrices
        self.init_weights(args.param_init, dist=args.param_init_dist)

        # Initialize bias vectors with zero
        self.init_weights(0, dist='constant', keys=['bias'])

    def forward(self, ys, hidden=None, reporter=None, is_eval=False, n_caches=0,
                ylens=[]):
        """Forward computation.

        Args:
            ys (list): A list of length `[B]`, which contains arrays of size `[L]`
            hidden (tuple or list): (h_n, c_n) or (hxs, cxs)
            reporter ():
            is_eval (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
            n_caches (int):
            ylens (list): not used
        Returns:
            loss (FloatTensor): `[1]`
            hidden (tuple or list): (h_n, c_n) or (hxs, cxs)
            reporter ():

        """
        if is_eval:
            self.eval()
            with torch.no_grad():
                loss, hidden, reporter = self._forward(ys, hidden, reporter, n_caches)
        else:
            self.train()
            loss, hidden, reporter = self._forward(ys, hidden, reporter)

        return loss, hidden, reporter

    def _forward(self, ys, hidden, reporter, n_caches=0):
        ys = [np2tensor(np.fromiter(y[::-1], dtype=np.int64) if self.backward else y, self.device_id).long()
              for y in ys]
        ys = pad_list(ys, self.pad)
        ys_in = ys[:, : -1]
        ys_out = ys[:, 1:]

        ys_in = self.encode(ys_in)

        lm_out, hidden = self.decode(ys_in, hidden)
        logits = self.generate(lm_out)

        # Compute XE sequence loss
        if n_caches > 0 and len(self.cache_ids) > 0:
            assert ys_out.size(1) == 1
            assert ys_out.size(0) == 1
            probs = F.softmax(logits, dim=-1)
            cache_probs = torch.zeros_like(probs)

            # Truncate cache
            self.cache_ids = self.cache_ids[-n_caches:]  # list of `[B, 1]`
            self.cache_keys = self.cache_keys[-n_caches:]  # list of `[B, 1, n_units]`

            # Compute inner-product over caches
            cache_attn = F.softmax(self.cache_theta * torch.matmul(
                torch.cat(self.cache_keys, dim=1),
                ys_in.transpose(1, 2)).squeeze(2), dim=1)

            # For visualization
            if len(self.cache_ids) == n_caches:
                self.cache_attn += [cache_attn.cpu().numpy()]
                self.cache_attn = self.cache_attn[-n_caches:]

            # Sum all probabilities
            for offset, idx in enumerate(self.cache_ids):
                cache_probs[:, :, idx] += cache_attn[:, offset]
            probs = (1 - self.cache_lambda) * probs + self.cache_lambda * cache_probs
            loss = -torch.log(probs[:, :, ys_out[:, -1]])
        else:
            loss = F.cross_entropy(logits.view((-1, logits.size(2))),
                                   ys_out.contiguous().view(-1),
                                   ignore_index=self.pad, size_average=True)

        # loss = self.adaptive_softmax(x, ys_out)

        if n_caches > 0:
            # Register to cache
            self.cache_ids += [ys_out[0, -1].item()]
            self.cache_keys += [ys_in]

        # Compute token-level accuracy in teacher-forcing
        acc = compute_accuracy(logits, ys_out, pad=self.pad)

        observation = {'loss.rnnlm': loss.item(),
                       'acc.rnnlm': acc,
                       'ppl.rnnlm': np.exp(loss.item())}

        # Report here
        if reporter is not None:
            is_eval = not self.training
            reporter.add(observation, is_eval)

        return loss, hidden, reporter

    def encode(self, ys):
        """Encode function.

        Args:
            ys (LongTensor):
        Returns:
            ys_emb (FloatTensor): `[B, L, emb_dim]`

        """
        return self.embed(ys)

    def decode(self, ys_emb, hidden=None):
        """Decode function.

        Args:
            ys_emb (FloatTensor): `[B, L, emb_dim]`
            hidden: dummy
        Returns:
            ys_emb (FloatTensor): `[B, L, n_units]`
            hidden: dummy

        """
        bs, max_ylen = ys_emb.size()[:2]

        # embed_dim = in_ch
        ys_emb = self.glu0(ys_emb.transpose(2, 1))  # `[B, out_ch, T]`

        # GLU blocks
        ys_emb = self.glu_layers(ys_emb)  # [B, out_ch, T]
        ys_emb = ys_emb.transpose(2, 1).contiguous()  # `[B, T, out_ch]`

        return ys_emb, hidden

    def generate(self, hidden):
        """Generate function.

        Args:
            hidden (FloatTensor): `[B, T, n_units]`
        Returns:
            logits (FloatTensor): `[B, T, vocab]`

        """
        return self.output(hidden)
