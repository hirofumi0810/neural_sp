#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Gated convolutional neural network language model with Gated Linear Units (GLU)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import logging
import torch.nn as nn

from neural_sp.models.lm.lm_base import LMBase
from neural_sp.models.modules.glu import ConvGLUBlock

logger = logging.getLogger(__name__)


class GatedConvLM(LMBase):
    """Gated convolutional neural network language model with Gated Linear Units (GLU)."""

    def __init__(self, args, save_path=None):

        super(LMBase, self).__init__()
        logger.info(self.__class__.__name__)

        self.lm_type = args.lm_type
        self.save_path = save_path

        self.emb_dim = args.emb_dim
        self.n_units = args.n_units
        self.n_layers = args.n_layers
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

        self.embed = nn.Embedding(self.vocab, args.emb_dim, padding_idx=self.pad)
        self.dropout_embed = nn.Dropout(p=args.dropout_in)

        model_size = args.lm_type.replace('gated_conv_', '')

        blocks = OrderedDict()
        dropout = args.dropout_hidden
        if model_size == 'custom':
            blocks['conv1'] = ConvGLUBlock(args.kernel_size, args.emb_dim, args.n_units,
                                           bottlececk_dim=args.n_projs,
                                           dropout=dropout)
            for l in range(args.n_layers - 1):
                blocks['conv%d' % (l + 2)] = ConvGLUBlock(args.kernel_size, args.n_units, args.n_units,
                                                          bottlececk_dim=args.n_projs,
                                                          dropout=dropout)
            last_dim = args.n_units

        elif model_size == '8':
            blocks['conv1'] = ConvGLUBlock(4, args.emb_dim, 900, dropout=dropout)
            for i in range(1, 8, 1):
                blocks['conv2-%d' % i] = ConvGLUBlock(4, 900, 900, dropout=dropout)
            last_dim = 900

        elif model_size == '8B':
            blocks['conv1'] = ConvGLUBlock(1, args.emb_dim, 512, dropout=dropout)
            for i in range(1, 4, 1):
                blocks['conv2-%d' % i] = ConvGLUBlock(5, 512, 512,
                                                      bottlececk_dim=128,
                                                      dropout=dropout)
            for i in range(1, 4, 1):
                blocks['conv3-%d' % i] = ConvGLUBlock(5, 512, 512,
                                                      bottlececk_dim=256,
                                                      dropout=dropout)
            blocks['conv4'] = ConvGLUBlock(1, 512, 2048,
                                           bottlececk_dim=1024,
                                           dropout=dropout)
            last_dim = 2048

        elif model_size == '9':
            blocks['conv1'] = ConvGLUBlock(4, args.emb_dim, 807, dropout=dropout)
            for i in range(1, 4, 1):
                blocks['conv2-%d-1' % i] = ConvGLUBlock(4, 807, 807, dropout=dropout)
                blocks['conv2-%d-2' % i] = ConvGLUBlock(4, 807, 807, dropout=dropout)
            last_dim = 807

        elif model_size == '13':
            blocks['conv1'] = ConvGLUBlock(4, args.emb_dim, 1268, dropout=dropout)
            for i in range(1, 13, 1):
                blocks['conv2-%d' % i] = ConvGLUBlock(4, 1268, 1268, dropout=dropout)
            last_dim = 1268

        elif model_size == '14':
            for i in range(1, 4, 1):
                blocks['conv1-%d' % i] = ConvGLUBlock(6, args.emb_dim if i == 1 else 850, 850,
                                                      dropout=dropout)
            blocks['conv2'] = ConvGLUBlock(1, 850, 850, dropout=dropout)
            for i in range(1, 5, 1):
                blocks['conv3-%d' % i] = ConvGLUBlock(5, 850, 850, dropout=dropout)
            blocks['conv4'] = ConvGLUBlock(1, 850, 850, dropout=dropout)
            for i in range(1, 4, 1):
                blocks['conv5-%d' % i] = ConvGLUBlock(4, 850, 850, dropout=dropout)
            blocks['conv6'] = ConvGLUBlock(4, 850, 1024, dropout=dropout)
            blocks['conv7'] = ConvGLUBlock(4, 1024, 2048, dropout=dropout)
            last_dim = 2048

        elif model_size == '14B':
            blocks['conv1'] = ConvGLUBlock(5, args.emb_dim, 512, dropout=dropout)
            for i in range(1, 4, 1):
                blocks['conv2-%d' % i] = ConvGLUBlock(5, 512, 512,
                                                      bottlececk_dim=128,
                                                      dropout=dropout)
            for i in range(1, 4, 1):
                blocks['conv3-%d' % i] = ConvGLUBlock(5, 512 if i == 1 else 1024, 1024,
                                                      bottlececk_dim=512,
                                                      dropout=dropout)
            for i in range(1, 7, 1):
                blocks['conv4-%d' % i] = ConvGLUBlock(5, 1024 if i == 1 else 2048, 2048,
                                                      bottlececk_dim=1024,
                                                      dropout=dropout)
            blocks['conv5'] = ConvGLUBlock(5, 2048, 4096,
                                           bottlececk_dim=1024,
                                           dropout=dropout)
            last_dim = 4096

        else:
            raise NotImplementedError(model_size)

        self.blocks = nn.Sequential(blocks)

        if args.adaptive_softmax:
            self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
                last_dim, self.vocab,
                # cutoffs=[self.vocab // 10, 3 * self.vocab // 10],
                cutoffs=[self.vocab // 25, self.vocab // 5],
                div_value=4.0)
            self.output = None
        else:
            self.adaptive_softmax = None
            self.output = nn.Linear(last_dim, self.vocab)
            if args.tie_embedding:
                if args.n_units != args.emb_dim:
                    raise ValueError('When using the tied flag, n_units must be equal to emb_dim.')
                self.output.weight = self.embed.weight

        self.reset_parameters(args.param_init)

    @staticmethod
    def add_args(parser, args):
        """Add arguments."""
        group = parser.add_argument_group("GatedConv LM")
        group.add_argument('--n_units', type=int, default=1024,
                           help='number of units in each layer')
        group.add_argument('--kernel_size', type=int, default=4,
                           help='kernel size for GatedConvLM')
        return parser

    def reset_parameters(self, param_init):
        """Initialize parameters with kaiming_uniform style."""
        logger.info('===== Initialize %s =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0.)  # bias
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', 0.))
            elif p.dim() in [2, 4]:
                nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')
                # nn.init.kaiming_normal_(p, mode='fan_in', nonlinearity='relu')
                logger.info('Initialize %s with %s / %.3f' % (n, 'kaiming_uniform', param_init))
            else:
                raise ValueError(n)

    def decode(self, ys, state=None, mems=None, incremental=False):
        """Decode function.

        Args:
            ys (LongTensor): `[B, L]`
            state: dummy interfance for RNNLM
            cache: dummy interfance for TransformerLM/TransformerXL
            incremental: dummy interfance for TransformerLM/TransformerXL
        Returns:
            logits (FloatTensor): `[B, L, vocab]`
            out (FloatTensor): `[B, L, d_model]` (for cache)
            new_cache: dummy interfance for RNNLM/TransformerLM/TransformerXL
            new_mems: dummy interfance for TransformerXL

        """
        out = self.dropout_embed(self.embed(ys.long()))
        bs, max_ylen = out.size()[:2]

        # NOTE: consider embed_dim as in_ch
        out = out.unsqueeze(3)
        out = self.blocks(out.transpose(2, 1))  # [B, out_ch, T, 1]
        out = out.transpose(2, 1).contiguous()  # `[B, T, out_ch, 1]`
        out = out.squeeze(3)
        if self.adaptive_softmax is None:
            logits = self.output(out)
        else:
            logits = out

        return logits, out, None
