# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Monotonic attention in MoChA at test time."""

import logging
import torch

logger = logging.getLogger(__name__)


def hard_monotonic_attention(e_ma, aw_prev, eps_wait):
    """Monotonic attention in MoChA at test time.

    Args:
        e_ma (FloatTensor): `[B, H_ma, qlen, klen]`
        aw_prev (FloatTensor): `[B, H_ma, qlen, klen]`
        eps_wait (int): wait time delay for head-synchronous decoding in MMA
    Returns:
        alpha (FloatTensor): `[B, H_ma, qlen, klen]`
        p_choose (FloatTensor): `[B, H_ma, qlen, klen]`

    """
    bs, H_ma, qlen, klen = e_ma.size()
    assert qlen == 1
    assert e_ma.size(-1) == aw_prev.size(-1)

    aw_prev = aw_prev[:, :, :, -klen:]
    # assert aw_prev.sum() > 0

    p_threshold = 0.5
    p_choose = (torch.sigmoid(e_ma[:, :, 0:1]) >= p_threshold).float()

    # Attend when monotonic energy is above threshold (Sigmoid > p_threshold)
    # Remove any probabilities before the index chosen at the last time step
    p_choose *= torch.cumsum(
        aw_prev[:, :, 0:1, -e_ma.size(3):], dim=-1)  # `[B, H_ma, 1 (qlen), klen]`

    # Now, use exclusive cumprod to remove probabilities after the first
    # chosen index, like so:
    # p_choose                        = [0, 0, 0, 1, 1, 0, 1, 1]
    # 1 - p_choose                    = [1, 1, 1, 0, 0, 1, 0, 0]
    # exclusive_cumprod(1 - p_choose) = [1, 1, 1, 1, 0, 0, 0, 0]
    # alpha: product of above         = [0, 0, 0, 1, 0, 0, 0, 0]
    alpha = p_choose * exclusive_cumprod(1 - p_choose)  # `[B, H_ma, 1 (qlen), klen]`

    if eps_wait > 0:
        for b in range(bs):
            # no boundary until the last frame for all heads
            if alpha[b].sum() == 0:
                continue

            leftmost = alpha[b, :, -1].nonzero()[:, -1].min().item()
            rightmost = alpha[b, :, -1].nonzero()[:, -1].max().item()
            for h in range(H_ma):
                # no bondary at the h-th head
                if alpha[b, h, -1].sum().item() == 0:
                    alpha[b, h, -1, min(rightmost, leftmost + eps_wait)] = 1
                    continue

                # surpass acceptable latency
                if alpha[b, h, -1].nonzero()[:, -1].min().item() >= leftmost + eps_wait:
                    alpha[b, h, -1, :] = 0  # reset
                    alpha[b, h, -1, leftmost + eps_wait] = 1

    return alpha, p_choose


def exclusive_cumprod(x):
    """Exclusive cumulative product [a, b, c] => [1, a, a * b].

        Args:
            x (FloatTensor): `[B, H, qlen, klen]`
        Returns:
            x (FloatTensor): `[B, H, qlen, klen]`

    """
    return torch.cumprod(torch.cat([x.new_ones(x.size(0), x.size(1), x.size(2), 1),
                                    x[:, :, :, :-1]], dim=-1), dim=-1)
