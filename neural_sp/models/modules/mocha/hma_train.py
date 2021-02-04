# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Monotonic attention in MoChA at training time."""

import logging
import torch

logger = logging.getLogger(__name__)


def parallel_monotonic_attention(e_ma, aw_prev, trigger_points, eps, noise_std,
                                 no_denom, decot, lookahead):
    """Efficient monotonic attention in MoChA at training time.

    Args:
        e_ma (FloatTensor): `[B, H_ma, qlen, klen]`
        aw_prev (FloatTensor): `[B, H_ma, qlen, klen]`
        trigger_points (IntTensor): `[B, qlen]`
        eps (float): epsilon parameter to avoid zero division
        noise_std (float): standard deviation for Gaussian noise
        no_denom (bool): set the denominator to 1 in the alpha recurrence
        decot (bool): delay constrainted training (DeCoT)
        lookahead (int): lookahead frames for DeCoT
    Returns:
        alpha (FloatTensor): `[B, H_ma, qlen, klen]`
        p_choose (FloatTensor): `[B, H_ma, qlen, klen]`

    """
    bs, H_ma, qlen, klen = e_ma.size()
    aw_prev = aw_prev[:, :, :, :klen]

    if decot:
        aw_prev_pad = aw_prev.new_zeros(bs, H_ma, qlen, klen)
        aw_prev_pad[:, :, :, :aw_prev.size(3)] = aw_prev
        aw_prev = aw_prev_pad

    bs, H_ma, qlen, klen = e_ma.size()
    p_choose = torch.sigmoid(add_gaussian_noise(e_ma, noise_std))  # `[B, H_ma, qlen, klen]`
    alpha = []
    # safe_cumprod computes cumprod in logspace with numeric checks
    cumprod_1mp_choose = safe_cumprod(1 - p_choose, eps=eps)  # `[B, H_ma, qlen, klen]`
    # Compute recurrence relation solution
    for i in range(qlen):
        denom = 1 if no_denom else torch.clamp(
            cumprod_1mp_choose[:, :, i:i + 1], min=eps, max=1.0)
        cumsum_in = aw_prev / denom  # `[B, H_ma, 1, klen]`

        monotonic = False
        if monotonic and i > 0:
            cumsum_in = torch.cat([denom.new_zeros(bs, H_ma, 1, 1),
                                   cumsum_in[:, :, :, 1:]], dim=-1)

        aw_prev = p_choose[:, :, i:i + 1] * cumprod_1mp_choose[:, :, i:i + 1] * \
            torch.cumsum(cumsum_in, dim=-1)  # `[B, H_ma, 1, klen]`
        # Mask the right part from the trigger point
        if decot:
            assert trigger_points is not None
            for b in range(bs):
                aw_prev[b, :, :, trigger_points[b, i:i + 1] + lookahead + 1:] = 0
        alpha.append(aw_prev)

    alpha = torch.cat(alpha, dim=2) if qlen > 1 else alpha[-1]  # `[B, H_ma, qlen, klen]`
    return alpha, p_choose


def add_gaussian_noise(x, std):
    """Add Gaussian noise to encourage discreteness.

    Args:
        x (FloatTensor): `[B, H_ma, qlen, klen]`
        std (float): standard deviation
    Returns:
        x (FloatTensor): `[B, H_ma, qlen, klen]`

    """
    noise = x.new_zeros(x.size()).normal_(std=std)
    return x + noise


def safe_cumprod(x, eps):
    """Numerically stable cumulative product by cumulative sum in log-space.

    Args:
        x (FloatTensor): `[B, H, qlen, klen]`
    Returns:
        x (FloatTensor): `[B, H, qlen, klen]`

    """
    return torch.exp(exclusive_cumsum(torch.log(torch.clamp(x, min=eps, max=1.0))))


def exclusive_cumsum(x):
    """Exclusive cumulative summation [a, b, c] => [0, a, a + b].

    Args:
        x (FloatTensor): `[B, H, qlen, klen]`
    Returns:
        x (FloatTensor): `[B, H, qlen, klen]`

    """
    return torch.cumsum(torch.cat([x.new_zeros(x.size(0), x.size(1), x.size(2), 1),
                                   x[:, :, :, :-1]], dim=-1), dim=-1)
