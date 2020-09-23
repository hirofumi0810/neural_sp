#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Subsampling layers."""

import math
import torch
import torch.nn as nn

from neural_sp.models.seq2seq.encoders.conv import update_lens_1d


class ConcatSubsampler(nn.Module):
    """Subsample by concatenating successive input frames."""

    def __init__(self, subsampling_factor, n_units):
        super(ConcatSubsampler, self).__init__()

        self.subsampling_factor = subsampling_factor
        if subsampling_factor > 1:
            self.proj = nn.Linear(n_units * subsampling_factor, n_units)

    def forward(self, xs, xlens, batch_first=True):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, F]` or `[T, B, F]`
            xlens (IntTensor): `[B]` (on CPU)
            batch_first (bool): operate batch-first tensor
        Returns:
            xs (FloatTensor): `[B, T', F']` or `[T', B, F']`
            xlens (IntTensor): `[B]` (on CPU)

        """
        if self.subsampling_factor == 1:
            return xs, xlens

        if batch_first:
            xs = xs.transpose(1, 0).contiguous()

        xs = [torch.cat([xs[t - r:t - r + 1] for r in range(self.subsampling_factor - 1, -1, -1)], dim=-1)
              for t in range(xs.size(0)) if (t + 1) % self.subsampling_factor == 0]
        xs = torch.cat(xs, dim=0)
        # NOTE: Exclude the last frames if the length is not divisible
        xs = torch.relu(self.proj(xs))

        if batch_first:
            xs = xs.transpose(1, 0)

        xlens = [max(1, i.item() // self.subsampling_factor) for i in xlens]
        xlens = torch.IntTensor(xlens)
        return xs, xlens


class Conv1dSubsampler(nn.Module):
    """Subsample by 1d convolution and max-pooling."""

    def __init__(self, subsampling_factor, n_units, conv_kernel_size=5):
        super(Conv1dSubsampler, self).__init__()

        assert conv_kernel_size % 2 == 1, "Kernel size should be odd for 'same' conv."
        self.subsampling_factor = subsampling_factor
        if subsampling_factor > 1:
            self.conv1d = nn.Conv1d(in_channels=n_units,
                                    out_channels=n_units,
                                    kernel_size=conv_kernel_size,
                                    stride=1,
                                    padding=(conv_kernel_size - 1) // 2)
            self.pool = nn.MaxPool1d(kernel_size=subsampling_factor,
                                     stride=subsampling_factor,
                                     padding=0,
                                     ceil_mode=True)

    def forward(self, xs, xlens, batch_first=True):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, F]` or `[T, B, F]`
            xlens (IntTensor): `[B]` (on CPU)
            batch_first (bool): operate batch-first tensor
        Returns:
            xs (FloatTensor): `[B, T', F']` or `[T', B, F']`
            xlens (IntTensor): `[B]` (on CPU)

        """
        if self.subsampling_factor == 1:
            return xs, xlens

        if batch_first:
            xs = torch.relu(self.conv1d(xs.transpose(2, 1)))
            xs = self.pool(xs).transpose(2, 1).contiguous()
        else:
            xs = torch.relu(self.conv1d(xs.permute(1, 2, 0)))
            xs = self.pool(xs).permute(2, 0, 1).contiguous()

        xlens = update_lens_1d(xlens, self.pool)
        return xs, xlens


class DropSubsampler(nn.Module):
    """Subsample by droping input frames."""

    def __init__(self, subsampling_factor):
        super(DropSubsampler, self).__init__()

        self.subsampling_factor = subsampling_factor

    def forward(self, xs, xlens, batch_first=True):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, F]` or `[T, B, F]`
            xlens (IntTensor): `[B]` (on CPU)
        Returns:
            xs (FloatTensor): `[B, T', F']` or `[T', B, F']`
            xlens (IntTensor): `[B]` (on CPU)

        """
        if self.subsampling_factor == 1:
            return xs, xlens

        if batch_first:
            xs = xs[:, ::self.subsampling_factor]
        else:
            xs = xs[::self.subsampling_factor]

        xlens = [max(1, math.ceil(i.item() / self.subsampling_factor)) for i in xlens]
        xlens = torch.IntTensor(xlens)
        return xs, xlens


class AddSubsampler(nn.Module):
    """Subsample by summing input frames."""

    def __init__(self, subsampling_factor):
        super(AddSubsampler, self).__init__()

        self.subsampling_factor = subsampling_factor
        assert subsampling_factor <= 2

    def forward(self, xs, xlens, batch_first=True):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, F]` or `[T, B, F]`
            xlens (IntTensor): `[B]` (on CPU)
            batch_first (bool): operate batch-first tensor
        Returns:
            xs (FloatTensor): `[B, T', F']` or `[T', B, F']`
            xlens (IntTensor): `[B]` (on CPU)

        """
        if self.subsampling_factor == 1:
            return xs, xlens

        if batch_first:
            bs, xmax, idim = xs.size()
            xs_even = xs[:, ::self.subsampling_factor]
            if xmax % 2 == 0:
                xs_odd = xs[:, 1::self.subsampling_factor]
            else:
                xs_odd = torch.cat([xs, xs.new_zeros(bs, 1, idim)], dim=1)[:, 1::self.subsampling_factor]
        else:
            xmax, bs, idim = xs.size()
            xs_even = xs[::self.subsampling_factor]
            if xmax % 2 == 0:
                xs_odd = xs[1::self.subsampling_factor]
            else:
                xs_odd = torch.cat([xs, xs.new_zeros(1, bs, idim)], dim=0)[1::self.subsampling_factor]

        xs = xs_odd + xs_even

        xlens = [max(1, math.ceil(i.item() / self.subsampling_factor)) for i in xlens]
        xlens = torch.IntTensor(xlens)
        return xs, xlens


class MaxpoolSubsampler(nn.Module):
    """Subsample by max-pooling input frames."""

    def __init__(self, subsampling_factor):
        super(MaxpoolSubsampler, self).__init__()

        self.subsampling_factor = subsampling_factor
        if subsampling_factor > 1:
            self.pool = nn.MaxPool1d(kernel_size=subsampling_factor,
                                     stride=subsampling_factor,
                                     padding=0,
                                     ceil_mode=True)

    def forward(self, xs, xlens, batch_first=True):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, F]` or `[T, B, F]`
            xlens (IntTensor): `[B]` (on CPU)
            batch_first (bool): operate batch-first tensor
        Returns:
            xs (FloatTensor): `[B, T', F']` or `[T', B, F']`
            xlens (IntTensor): `[B]` (on CPU)

        """
        if self.subsampling_factor == 1:
            return xs, xlens

        if batch_first:
            xs = self.pool(xs.transpose(2, 1)).transpose(2, 1).contiguous()
        else:
            xs = self.pool(xs.permute(1, 2, 0)).permute(2, 0, 1).contiguous()

        xlens = update_lens_1d(xlens, self.pool)
        return xs, xlens
