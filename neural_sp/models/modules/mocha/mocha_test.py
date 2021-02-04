# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Chunkwise attention in MoChA at test time."""

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


def hard_chunkwise_attention(alpha, u, mask, chunk_size, H_ca,
                             sharpening_factor, share_chunkwise_attention):
    """Chunkwise attention in MoChA at test time.

    Args:
        alpha (FloatTensor): `[B, H_ma, qlen, klen]`
        u (FloatTensor): `[B, (H_ma*)H_ca, qlen, klen]`
        mask (ByteTensor): `[B, qlen, klen]`
        chunk_size (int): window size for chunkwise attention
        H_ca (int): number of chunkwise attention heads
        sharpening_factor (float): sharping factor for beta calculation
        share_chunkwise_attention (int): share CA heads among MA heads
    Returns:
        beta (FloatTensor): `[B, H_ma * H_ca, qlen, klen]`

    """
    bs, H_ma, qlen, klen = alpha.size()
    assert (u.size(2) == qlen) and (u.size(3) == klen), (u.size(), alpha.size())
    alpha = alpha.unsqueeze(2)   # `[B, H_ma, 1, qlen, klen]`
    u = u.unsqueeze(1)  # `[B, 1, (H_ma*)H_ca, qlen, klen]`
    if H_ca > 1:
        alpha = alpha.repeat([1, 1, H_ca, 1, 1])
    if H_ma > 1:
        if share_chunkwise_attention:
            u = u.repeat([1, H_ma, 1, 1, 1])
        else:
            u = u.view(bs, H_ma, H_ca, qlen, klen)

    mask = alpha.clone().byte()  # `[B, H_ma, H_ca, qlen, klen]`
    for b in range(bs):
        for h in range(H_ma):
            if alpha[b, h, 0, 0].sum() > 0:
                boundary = alpha[b, h, 0, 0].nonzero()[:, -1].min().item()
                if chunk_size == -1:
                    # infinite lookback attention
                    mask[b, h, :, 0, 0:boundary + 1] = 1
                else:
                    mask[b, h, :, 0, max(0, boundary - chunk_size + 1):boundary + 1] = 1

    NEG_INF = float(np.finfo(torch.tensor(0, dtype=u.dtype).numpy().dtype).min)
    u = u.masked_fill(mask == 0, NEG_INF)
    beta = torch.softmax(u, dim=-1)
    return beta.view(bs, -1, qlen, klen)
