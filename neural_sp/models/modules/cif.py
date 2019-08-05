# ! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torch.nn as nn

from neural_sp.models.modules.linear import Linear
from neural_sp.models.torch_utils import pad_list


class CIF(nn.Module):
    """docstring for CIF."""

    def __init__(self, enc_dim, conv_out_channels, conv_kernel_size,
                 threshold=0.9):
        super(CIF, self).__init__()

        self.threshold = threshold
        self.channel = conv_out_channels

        self.conv = nn.Conv1d(in_channels=enc_dim,
                              out_channels=conv_out_channels,
                              kernel_size=conv_kernel_size * 2 + 1,
                              stride=1,
                              padding=conv_kernel_size)
        self.proj = Linear(conv_out_channels, 1)

    def forward(self, eouts, elens, ylens=None):
        """

        Args:
            eouts (FloatTensor): `[B, T, enc_dim]`
            elens (IntTensor): `[B]`
            ylens ():
        Returns:
            eouts_fired (FloatTensor):
            alpha (FloatTensor): `[B, T]`

        """
        bs, time, enc_dim = eouts.size()

        # 1d conv
        conv_feat = self.conv(eouts.transpose(2, 1))  # `[B, channel, kmax]`
        conv_feat = conv_feat.transpose(2, 1)
        alpha = torch.sigmoid(self.proj(conv_feat)).squeeze(2)  # `[B, kmax]`

        eouts_fired = []
        for b in range(bs):
            # normalization
            if ylens is not None:
                alpha_norm = alpha[b] / alpha[b].sum() * ylens[b]
            else:
                alpha_norm = alpha[b]

            alpha_residual = 0
            state_residual = eouts.new_zeros(self.channel)
            state_accum_b_list = []
            for t in range(elens[b]):
                alpha_accum = alpha_norm[t] + alpha_residual
                if alpha_accum >= self.threshold:
                    # fire
                    alpha_accum = 1  # this is import for the next frame
                    alpha_residual = alpha_norm[t] - (1 - alpha_residual)
                    state_accum = state_residual + (1 - alpha_residual) * eouts[b, t]
                    state_accum_b_list.append(state_accum.unsqueeze(0))
                    state_residual = alpha_residual * eouts[b, t]
                else:
                    # Carry over to the next frame
                    alpha_residual = alpha_accum
                    state_accum = state_residual + alpha_norm[t] * eouts[b, t]
                    state_residual = state_accum

                # tail of target sequence
                if ylens is None:
                    if alpha_accum >= 0.5:
                        state_accum_b_list.append(state_accum.unsqueeze(0))

            # padding for batch training
            if ylens is not None:
                for _ in range(ylens[b].item() - len(state_accum_b_list)):
                    state_accum_b_list.append(eouts.new_zeros(1, self.channel))
            eouts_fired.append(torch.cat(state_accum_b_list, dim=0))

        eouts_fired = pad_list(eouts_fired, 0.0)
        return eouts_fired, alpha
