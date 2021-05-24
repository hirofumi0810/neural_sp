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

        self.factor = subsampling_factor
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
        if self.factor == 1:
            return xs, xlens

        if batch_first:
            xs = xs.transpose(1, 0).contiguous()

        xs = [torch.cat([xs[t - r:t - r + 1] for r in range(self.factor - 1, -1, -1)], dim=-1)
              for t in range(xs.size(0)) if (t + 1) % self.factor == 0]
        xs = torch.cat(xs, dim=0)
        # NOTE: Exclude the last frames if the length is not divisible
        xs = torch.relu(self.proj(xs))

        if batch_first:
            xs = xs.transpose(1, 0)

        xlens = [max(1, i.item() // self.factor) for i in xlens]
        xlens = torch.IntTensor(xlens)
        return xs, xlens


class Conv1dSubsampler(nn.Module):
    """Subsample by stride in 1d convolution."""

    def __init__(self, subsampling_factor, n_units, kernel_size=3):
        super(Conv1dSubsampler, self).__init__()

        assert kernel_size % 2 == 1, "Kernel size should be odd for 'same' conv."
        self.factor = subsampling_factor
        if subsampling_factor > 1:
            self.conv1d = nn.Conv1d(in_channels=n_units,
                                    out_channels=n_units,
                                    kernel_size=kernel_size,
                                    stride=subsampling_factor,
                                    padding=(kernel_size - 1) // 2)

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
        if self.factor == 1:
            return xs, xlens

        if batch_first:
            xs = self.conv1d(xs.transpose(2, 1))
            xs = xs.transpose(2, 1).contiguous()
        else:
            xs = self.conv1d(xs.permute(1, 2, 0))
            xs = xs.permute(2, 0, 1).contiguous()
        xs = torch.relu(xs)

        xlens = update_lens_1d(xlens, self.conv1d)
        return xs, xlens


class DropSubsampler(nn.Module):
    """Subsample by dropping input frames."""

    def __init__(self, subsampling_factor):
        super(DropSubsampler, self).__init__()

        self.factor = subsampling_factor

    def forward(self, xs, xlens, batch_first=True):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, F]` or `[T, B, F]`
            xlens (IntTensor): `[B]` (on CPU)
        Returns:
            xs (FloatTensor): `[B, T', F']` or `[T', B, F']`
            xlens (IntTensor): `[B]` (on CPU)

        """
        if self.factor == 1:
            return xs, xlens

        if batch_first:
            xs = xs[:, ::self.factor]
        else:
            xs = xs[::self.factor]

        xlens = [max(1, math.ceil(i.item() / self.factor)) for i in xlens]
        xlens = torch.IntTensor(xlens)
        return xs, xlens


class AddSubsampler(nn.Module):
    """Subsample by summing input frames."""

    def __init__(self, subsampling_factor):
        super(AddSubsampler, self).__init__()

        self.factor = subsampling_factor
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
        if self.factor == 1:
            return xs, xlens

        if batch_first:
            bs, xmax, idim = xs.size()
            xs_even = xs[:, ::self.factor]
            if xmax % 2 == 0:
                xs_odd = xs[:, 1::self.factor]
            else:
                xs_odd = torch.cat([xs, xs.new_zeros(bs, 1, idim)], dim=1)[:, 1::self.factor]
        else:
            xmax, bs, idim = xs.size()
            xs_even = xs[::self.factor]
            if xmax % 2 == 0:
                xs_odd = xs[1::self.factor]
            else:
                xs_odd = torch.cat([xs, xs.new_zeros(1, bs, idim)], dim=0)[1::self.factor]

        xs = xs_odd + xs_even

        xlens = [max(1, math.ceil(i.item() / self.factor)) for i in xlens]
        xlens = torch.IntTensor(xlens)
        return xs, xlens


class MaxPoolSubsampler(nn.Module):
    """Subsample by max-pooling input frames."""

    def __init__(self, subsampling_factor):
        super(MaxPoolSubsampler, self).__init__()

        self.factor = subsampling_factor
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
        if self.factor == 1:
            return xs, xlens

        if batch_first:
            xs = self.pool(xs.transpose(2, 1)).transpose(2, 1).contiguous()
        else:
            xs = self.pool(xs.permute(1, 2, 0)).permute(2, 0, 1).contiguous()

        xlens = update_lens_1d(xlens, self.pool)
        return xs, xlens


class MeanPoolSubsampler(nn.Module):
    """Subsample by mean-pooling input frames."""

    def __init__(self, subsampling_factor):
        super(MeanPoolSubsampler, self).__init__()

        self.factor = subsampling_factor
        if subsampling_factor > 1:
            self.pool = nn.AvgPool1d(kernel_size=subsampling_factor,
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
        if self.factor == 1:
            return xs, xlens

        if batch_first:
            xs = self.pool(xs.transpose(2, 1)).transpose(2, 1).contiguous()
        else:
            xs = self.pool(xs.permute(1, 2, 0)).permute(2, 0, 1).contiguous()

        xlens = update_lens_1d(xlens, self.pool)
        return xs, xlens
