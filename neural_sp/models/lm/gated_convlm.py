#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Gated convolutional neural network language model with Gated Linear Units (GLU)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_sp.models.lm.lm_base import LMBase
from neural_sp.models.modules.embedding import Embedding
from neural_sp.models.modules.linear import LinearND
from neural_sp.models.modules.glu import GLUBlock
from neural_sp.models.torch_utils import compute_accuracy
from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list


class GatedConvLM(LMBase):
    """Gated convolutional neural network language model with Gated Linear Units (GLU)."""

    def __init__(self, args):

        super(LMBase, self).__init__()
        logger = logging.getLogger('training')
        logger.info(self.__class__.__name__)

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

        self.embed = Embedding(vocab=self.vocab,
                               emb_dim=args.emb_dim,
                               dropout=args.dropout_emb,
                               ignore_index=self.pad)

        layers = OrderedDict()

        model_size = args.lm_type.replace('gated_conv_', '')

        if model_size == 'custom':
            layers['conv1'] = GLUBlock(args.kernel_size, args.emb_dim, args.n_units,
                                       bottlececk_dim=args.n_projs,
                                       dropout=args.dropout_hidden)
            for l in range(args.n_layers - 1):
                layers['conv%d' % (l + 2)] = GLUBlock(args.kernel_size, args.n_units, args.n_units,
                                                      bottlececk_dim=args.n_projs,
                                                      dropout=args.dropout_hidden)
            last_dim = args.n_units

        elif model_size == '8':
            layers['conv1'] = GLUBlock(4, args.emb_dim, 900,
                                       dropout=args.dropout_hidden)
            for i in range(1, 8, 1):
                layers['conv2-%d' % i] = GLUBlock(4, 900, 900,
                                                  dropout=args.dropout_hidden)
            last_dim = 900

        elif model_size == '8B':
            layers['conv1'] = GLUBlock(1, args.emb_dim, 512,
                                       dropout=args.dropout_hidden)
            for i in range(1, 4, 1):
                layers['conv2-%d' % i] = GLUBlock(5, 512, 512,
                                                  bottlececk_dim=128,
                                                  dropout=args.dropout_hidden)
            for i in range(1, 4, 1):
                layers['conv3-%d' % i] = GLUBlock(5, 512, 512,
                                                  bottlececk_dim=256,
                                                  dropout=args.dropout_hidden)
            layers['conv4'] = GLUBlock(1, 512, 2048,
                                       bottlececk_dim=1024,
                                       dropout=args.dropout_hidden)
            last_dim = 2048

        elif model_size == '9':
            layers['conv1'] = GLUBlock(4, args.emb_dim, 807,
                                       dropout=args.dropout_hidden)
            for i in range(1, 4, 1):
                layers['conv2-%d-1' % i] = GLUBlock(4, 807, 807,
                                                    dropout=args.dropout_hidden)
                layers['conv2-%d-2' % i] = GLUBlock(4, 807, 807,
                                                    dropout=args.dropout_hidden)
            last_dim = 807

        elif model_size == '13':
            layers['conv1'] = GLUBlock(4, args.emb_dim, 1268,
                                       dropout=args.dropout_hidden)
            for i in range(1, 13, 1):
                layers['conv2-%d' % i] = GLUBlock(4, 1268, 1268,
                                                  dropout=args.dropout_hidden)
            last_dim = 1268

        elif model_size == '14':
            for i in range(1, 4, 1):
                layers['conv1-%d' % i] = GLUBlock(6, args.emb_dim if i == 1 else 850, 850,
                                                  dropout=args.dropout_hidden)
            layers['conv2'] = GLUBlock(1, 850, 850,
                                       dropout=args.dropout_hidden)
            for i in range(1, 5, 1):
                layers['conv3-%d' % i] = GLUBlock(5, 850, 850,
                                                  dropout=args.dropout_hidden)
            layers['conv4'] = GLUBlock(1, 850, 850,
                                       dropout=args.dropout_hidden)
            for i in range(1, 4, 1):
                layers['conv5-%d' % i] = GLUBlock(4, 850, 850,
                                                  dropout=args.dropout_hidden)
            layers['conv6'] = GLUBlock(4, 850, 1024,
                                       dropout=args.dropout_hidden)
            layers['conv7'] = GLUBlock(4, 1024, 2048,
                                       dropout=args.dropout_hidden)
            last_dim = 2048

        elif model_size == '14B':
            layers['conv1'] = GLUBlock(5, args.emb_dim, 512,
                                       dropout=args.dropout_hidden)
            for i in range(1, 4, 1):
                layers['conv2-%d' % i] = GLUBlock(5, 512, 512,
                                                  bottlececk_dim=128,
                                                  dropout=args.dropout_hidden)
            for i in range(1, 4, 1):
                layers['conv3-%d' % i] = GLUBlock(5, 512 if i == 1 else 1024, 1024,
                                                  bottlececk_dim=512,
                                                  dropout=args.dropout_hidden)
            for i in range(1, 7, 1):
                layers['conv4-%d' % i] = GLUBlock(5, 1024 if i == 1 else 2048, 2048,
                                                  bottlececk_dim=1024,
                                                  dropout=args.dropout_hidden)
            layers['conv5'] = GLUBlock(5, 2048, 4096,
                                       bottlececk_dim=1024,
                                       dropout=args.dropout_hidden)
            last_dim = 4096

        else:
            raise NotImplementedError(model_size)

        self.layers = nn.Sequential(layers)

        if args.adaptive_softmax:
            self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
                last_dim, self.vocab,
                # cutoffs=[self.vocab // 10, 3 * self.vocab // 10],
                cutoffs=[self.vocab // 25, self.vocab // 5],
                div_value=4.0)
            self.output = None
        else:
            self.adaptive_softmax = None
            self.output = LinearND(last_dim, self.vocab,
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

        # Initialize parameters
        self.reset_parameters(args.param_init)

    def reset_parameters(self, param_init):
        """Initialize parameters with kaiming_uniform style."""
        logger = logging.getLogger('training')
        logger.info('===== Initialize %s =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.constant_(p, val=0)  # bias
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0))
            elif p.dim() in [2, 4]:
                nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')
                # nn.init.kaiming_normal_(p, mode='fan_in', nonlinearity='relu')
                logger.info('Initialize %s with %s / %.3f' % (n, 'kaiming_uniform', param_init))
            else:
                raise ValueError

    def _forward(self, ys, hidden, reporter, n_caches=0):
        ys = [np2tensor(np.fromiter(y[::-1], dtype=np.int64) if self.backward else y, self.device_id).long()
              for y in ys]
        ys = pad_list(ys, self.pad)
        ys_in = ys[:, : -1]
        ys_out = ys[:, 1:]

        lmout, hidden = self.decode(self.encode(ys_in), hidden)
        if self.adaptive_softmax is None:
            logits = self.generate(lmout)
        else:
            logits = lmout

        # Compute XE sequence loss
        if n_caches > 0 and len(self.cache_ids) > 0:
            assert ys_out.size(1) == 1
            assert ys_out.size(0) == 1
            if self.adaptive_softmax is None:
                probs = F.softmax(logits, dim=-1)
            else:
                probs = self.adaptive_softmax.log_prob(logits).exp()
            cache_probs = probs.new_zeros(probs.size())

            # Truncate cache
            self.cache_ids = self.cache_ids[-n_caches:]  # list of `[B, 1]`
            self.cache_keys = self.cache_keys[-n_caches:]  # list of `[B, 1, n_units]`

            # Compute inner-product over caches
            cache_attn = F.softmax(self.cache_theta * torch.matmul(
                torch.cat(self.cache_keys, dim=1), lmout.transpose(2, 1)).squeeze(2), dim=1)

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
            if self.adaptive_softmax is None:
                loss = F.cross_entropy(logits.view((-1, logits.size(2))),
                                       ys_out.contiguous().view(-1),
                                       ignore_index=self.pad, size_average=True)
            else:
                loss = self.adaptive_softmax(logits.view((-1, logits.size(2))),
                                             ys_out.contiguous().view(-1)).loss

        if n_caches > 0:
            # Register to cache
            self.cache_ids += [ys_out[0, -1].item()]
            self.cache_keys += [lmout]

        # Compute token-level accuracy in teacher-forcing
        if self.adaptive_softmax is None:
            acc = compute_accuracy(logits, ys_out, pad=self.pad)
        else:
            acc = compute_accuracy(self.adaptive_softmax.log_prob(
                logits.view((-1, logits.size(2)))), ys_out, pad=self.pad)

        observation = {'loss.lm': loss.item(),
                       'acc.lm': acc,
                       'ppl.lm': np.exp(loss.item())}

        # Report here
        if reporter is not None:
            is_eval = not self.training
            reporter.add(observation, is_eval)

        return loss, hidden, reporter

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

        # NOTE: consider embed_dim as in_ch
        ys_emb = ys_emb.unsqueeze(3)
        ys_emb = self.layers(ys_emb.transpose(2, 1))  # [B, out_ch, T, 1]
        ys_emb = ys_emb.transpose(2, 1).contiguous()  # `[B, T, out_ch, 1]`
        ys_emb = ys_emb.squeeze(3)

        return ys_emb, hidden
