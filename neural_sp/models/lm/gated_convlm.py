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
from neural_sp.models.modules.embedding import Embedding
from neural_sp.models.modules.linear import LinearND
from neural_sp.models.torch_utils import compute_accuracy
from neural_sp.models.torch_utils import np2tensor
from neural_sp.models.torch_utils import pad_list


class GLUBlock(nn.Module):
    """GLU block.

    Args:
        kernel_size (int): kernel size
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        bottlececk_dim (int): dimension of the bottleneck layers for computational efficiency
        dropout (float): dropout probability

    """

    def __init__(self, kernel_size, in_ch, out_ch, bottlececk_dim=0, dropout=0.0):
        super().__init__()

        self.conv_residual = None
        if in_ch != out_ch:
            self.conv_residual = nn.utils.weight_norm(nn.Conv2d(in_channels=in_ch,
                                                                out_channels=out_ch,
                                                                kernel_size=(1, 1)),
                                                      name='weight', dim=0)

        self.pad_left = nn.ConstantPad2d((0, 0, kernel_size - 1, 0), 0)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_gate = nn.Dropout(p=dropout)

        if bottlececk_dim == 0:
            self.conv_in = lambda x: x
            self.conv = nn.utils.weight_norm(nn.Conv2d(in_channels=in_ch,
                                                       out_channels=out_ch,
                                                       kernel_size=(kernel_size, 1)),
                                             name='weight', dim=0)
            self.conv_out = lambda x: x

            self.conv_gate_in = lambda x: x
            self.conv_gate = nn.utils.weight_norm(nn.Conv2d(in_channels=in_ch,
                                                            out_channels=out_ch,
                                                            kernel_size=(kernel_size, 1)),
                                                  name='weight', dim=0)
            self.conv_gate_out = lambda x: x

        elif bottlececk_dim > 0:
            self.conv_in = nn.utils.weight_norm(nn.Conv2d(in_channels=in_ch,
                                                          out_channels=bottlececk_dim,
                                                          kernel_size=(1, 1)),
                                                name='weight', dim=0)
            self.conv = nn.utils.weight_norm(nn.Conv2d(in_channels=bottlececk_dim,
                                                       out_channels=bottlececk_dim,
                                                       kernel_size=(kernel_size, 1)),
                                             name='weight', dim=0)
            self.conv_out = nn.utils.weight_norm(nn.Conv2d(in_channels=bottlececk_dim,
                                                           out_channels=out_ch,
                                                           kernel_size=(1, 1)),
                                                 name='weight', dim=0)

            self.conv_gate_in = nn.utils.weight_norm(nn.Conv2d(in_channels=in_ch,
                                                               out_channels=bottlececk_dim,
                                                               kernel_size=(1, 1)),
                                                     name='weight', dim=0)
            self.conv_gate = nn.utils.weight_norm(nn.Conv2d(in_channels=bottlececk_dim,
                                                            out_channels=bottlececk_dim,
                                                            kernel_size=(kernel_size, 1)),
                                                  name='weight', dim=0)
            self.conv_gate_out = nn.utils.weight_norm(nn.Conv2d(in_channels=bottlececk_dim,
                                                                out_channels=out_ch,
                                                                kernel_size=(1, 1)),
                                                      name='weight', dim=0)

    def forward(self, x):
        """Forward computation.
        Args:
            x (FloatTensor): `[B, in_ch, T]`
        Returns:
            out (FloatTensor): `[B, out_ch, T]`

        """
        residual = x
        if self.conv_residual is not None:
            residual = self.conv_residual(residual)
        x = self.pad_left(x)  # `[B, embed_dim, T+kernel-1, 1]`
        a = self.conv_out(self.conv(self.conv_in(x)))  # `[B, out_ch, T ,1]`
        a = self.dropout(a)
        b = self.conv_gate_out(self.conv_gate(self.conv_gate_in(x)))  # `[B, out_ch, T, 1]`
        b = self.dropout_gate(b)

        x = torch.mul(a, F.sigmoid(b))
        x += residual
        return x


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

        self.embed = Embedding(vocab=self.vocab,
                               emb_dim=args.emb_dim,
                               dropout=args.dropout_emb,
                               ignore_index=self.pad)

        glu_layers = OrderedDict()

        # model = 'small'
        # model = '8'
        # model = '13'
        # model = '14'
        model = '14B'

        if model == 'small':
            glu_layers['conv1-1'] = GLUBlock(4, args.emb_dim, 600, bottlececk_dim=15)
            glu_layers['conv2-1'] = GLUBlock(4, 600, 600, bottlececk_dim=30)
            glu_layers['conv3-1'] = GLUBlock(4, 600, 600, bottlececk_dim=30)
            glu_layers['conv4-1'] = GLUBlock(4, 600, 600, bottlececk_dim=30)
            glu_layers['conv5-1'] = GLUBlock(4, 600, 600, bottlececk_dim=30)
            last_dim = 600

        elif model == '8':
            glu_layers['conv1-1'] = GLUBlock(4, args.emb_dim, 900)
            for i in range(1, 8, 1):
                glu_layers['conv2-%d' % i] = GLUBlock(4, 900, 900)
            last_dim = 900

        elif model == '13':
            glu_layers['conv1-1'] = GLUBlock(4, args.emb_dim, 1268)
            for i in range(1, 13, 1):
                glu_layers['conv2-%d' % i] = GLUBlock(4, 1268, 1268)
            last_dim = 1268

        elif model == '14':
            for i in range(1, 4, 1):
                glu_layers['conv1-%d' % i] = GLUBlock(6, args.emb_dim if i == 1 else 850, 850)
            glu_layers['conv2-1'] = GLUBlock(1, 850, 850)
            for i in range(1, 5, 1):
                glu_layers['conv3-%d' % i] = GLUBlock(5, 850, 850)
            glu_layers['conv4-1'] = GLUBlock(1, 850, 850)
            for i in range(1, 4, 1):
                glu_layers['conv5-%d' % i] = GLUBlock(4, 850, 850)
            glu_layers['conv6-1'] = GLUBlock(4, 850, 1024)
            glu_layers['conv7-1'] = GLUBlock(4, 1024, 2048)
            last_dim = 2048

        elif model == '14B':
            glu_layers['conv1-1'] = GLUBlock(5, args.emb_dim, 512)
            for i in range(1, 4, 1):
                glu_layers['conv2-%d' % i] = GLUBlock(5, 512, 512, bottlececk_dim=128)
            for i in range(1, 4, 1):
                glu_layers['conv3-%d' % i] = GLUBlock(5, 512 if i == 1 else 1024, 1024, bottlececk_dim=512)
            for i in range(1, 7, 1):
                glu_layers['conv4-%d' % i] = GLUBlock(5, 1024 if i == 1 else 2048, 2048, bottlececk_dim=1024)
            glu_layers['conv5-1'] = GLUBlock(5, 2048, 4096, bottlececk_dim=1024)
            last_dim = 4096
        else:
            raise NotImplementedError(model)

        self.glu_layers = nn.Sequential(glu_layers)

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
        self.reset_parameters(args.param_init, dist=args.param_init_dist)

        # Initialize bias vectors with zero
        self.reset_parameters(0, dist='constant', keys=['bias'])

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

        lmout, hidden = self.decode(ys_in, hidden)
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
            self.cache_keys += [ys_in]

        # Compute token-level accuracy in teacher-forcing
        if self.adaptive_softmax is None:
            acc = compute_accuracy(logits, ys_out, pad=self.pad)
        else:
            acc = 0

        observation = {'loss.lm': loss.item(),
                       'acc.lm': acc,
                       'ppl.lm': np.exp(loss.item())}

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

        # NOTE: embed_dim = in_ch
        ys_emb = ys_emb.unsqueeze(3)
        ys_emb = self.glu_layers(ys_emb.transpose(2, 1))  # [B, out_ch, T, 1]
        ys_emb = ys_emb.transpose(2, 1).contiguous()  # `[B, T, out_ch]`
        ys_emb = ys_emb.squeeze(3)

        return ys_emb, hidden

    def generate(self, hidden):
        """Generate function.

        Args:
            hidden (FloatTensor): `[B, T, n_units]`
        Returns:
            logits (FloatTensor): `[B, T, vocab]`

        """
        return self.output(hidden)
