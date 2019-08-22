# ! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torch.nn as nn

from neural_sp.models.modules.linear import Linear


class CIF(nn.Module):
    """docstring for CIF."""

    def __init__(self, enc_dim, conv_out_channels, conv_kernel_size,
                 threshold=0.9):
        super(CIF, self).__init__()

        self.threshold = threshold
        self.channel = conv_out_channels
        self.n_heads = 1

        self.conv = nn.Conv1d(in_channels=enc_dim,
                              out_channels=conv_out_channels,
                              kernel_size=conv_kernel_size * 2 + 1,
                              stride=1,
                              padding=conv_kernel_size)
        self.proj = Linear(conv_out_channels, 1)

    def forward(self, eouts, elens, ylens=None, max_len=200):
        """

        Args:
            eouts (FloatTensor): `[B, T, enc_dim]`
            elens (IntTensor): `[B]`
            ylens (IntTensor): `[B]`
            max_len (int): the maximum length of target sequence
        Returns:
            eouts_fired (FloatTensor): `[B, T, enc_dim]`
            alpha (FloatTensor): `[B, T]`
            aws (FloatTensor): `[B, 1 (head), L. T]`

        """
        bs, xtime, enc_dim = eouts.size()

        # 1d conv
        conv_feat = self.conv(eouts.transpose(2, 1))  # `[B, channel, kmax]`
        conv_feat = conv_feat.transpose(2, 1)
        alpha = torch.sigmoid(self.proj(conv_feat)).squeeze(2)  # `[B, kmax]`

        # normalization
        if ylens is not None:
            alpha_norm = alpha / alpha.sum(1).unsqueeze(1) * ylens.unsqueeze(1)
        else:
            alpha_norm = alpha

        if ylens is not None:
            max_len = ylens.max().int()
        eouts_fired = eouts.new_zeros(bs, max_len + 1, enc_dim)
        aws = eouts.new_zeros(bs, 1, max_len + 1, xtime)
        n_tokens = torch.zeros(bs, dtype=torch.int32)
        state = eouts.new_zeros(bs, self.channel)
        alpha_accum = eouts.new_zeros(bs)
        for t in range(xtime):
            alpha_accum += alpha_norm[:, t]

            for b in range(bs):
                # skip the padding region
                if t > elens[b] - 1:
                    continue
                # skip all-fired utterance
                if ylens is not None and n_tokens[b] >= ylens[b].item():
                    continue
                if alpha_accum[b] >= self.threshold:
                    # fire
                    ak1 = 1 - alpha_accum[b]
                    ak2 = alpha_norm[b, t] - ak1
                    aws[b, 0, n_tokens[b], t] += ak1
                    eouts_fired[b, n_tokens[b]] = state[b] + ak1 * eouts[b, t]
                    n_tokens[b] += 1
                    # Carry over to the next frame
                    state[b] = ak2 * eouts[b, t]
                    alpha_accum[b] = ak2
                    aws[b, 0, n_tokens[b], t] += ak2
                else:
                    # Carry over to the next frame
                    state[b] += alpha_norm[b, t] * eouts[b, t]
                    aws[b, 0, n_tokens[b], t] += alpha_norm[b, t]

            # tail of target sequence
            if ylens is None and t == elens[b] - 1:
                if alpha_accum[b] >= 0.5:
                    n_tokens[b] += 1
                    eouts_fired[b, n_tokens[b]] = state[b]

        # truncate
        eouts_fired = eouts_fired[:, :max_len]
        aws = aws[:, :max_len]

        return eouts_fired, alpha, aws
