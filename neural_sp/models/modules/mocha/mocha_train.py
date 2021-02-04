# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Chunkwise attention in MoChA at training time."""

import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def soft_chunkwise_attention(alpha, u, mask, chunk_size, H_ca,
                             sharpening_factor, share_chunkwise_attention):
    """Chunkwise attention in MoChA at training time.

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
    alpha = alpha.unsqueeze(2)  # `[B, H_ma, 1, qlen, klen]`
    u = u.unsqueeze(1)  # `[B, 1, (H_ma*)H_ca, qlen, klen]`
    if H_ca > 1:
        alpha = alpha.repeat([1, 1, H_ca, 1, 1])
    if H_ma > 1 and not share_chunkwise_attention:
        u = u.view(bs, H_ma, H_ca, qlen, klen)
    # Shift logits to avoid overflow
    u -= torch.max(u, dim=-1, keepdim=True)[0]
    # Limit the range for numerical stability
    softmax_exp = torch.clamp(torch.exp(u), min=1e-5)
    # Compute chunkwise softmax denominators
    if chunk_size == -1:
        # infinite lookback attention
        # inner_items = alpha * sharpening_factor / torch.cumsum(softmax_exp, dim=-1)
        # beta = softmax_exp * torch.cumsum(inner_items.flip(dims=[-1]), dim=-1).flip(dims=[-1])
        # beta = beta.masked_fill(mask.unsqueeze(1), 0)
        # beta = beta / beta.sum(dim=-1, keepdim=True)

        softmax_denominators = torch.cumsum(softmax_exp, dim=-1)
        # Compute \beta_{i, :}. emit_probs are \alpha_{i, :}.
        beta = softmax_exp * moving_sum(alpha * sharpening_factor / softmax_denominators,
                                        back=0, forward=klen - 1)
    else:
        softmax_denominators = moving_sum(softmax_exp,
                                          back=chunk_size - 1, forward=0)
        # Compute \beta_{i, :}. emit_probs are \alpha_{i, :}.
        beta = softmax_exp * moving_sum(alpha * sharpening_factor / softmax_denominators,
                                        back=0, forward=chunk_size - 1)
    return beta.view(bs, -1, qlen, klen)


def moving_sum(x, back, forward):
    """Compute the moving sum of x over a chunk_size with the provided bounds.

    Args:
        x (FloatTensor): `[B, H_ma, H_ca, qlen, klen]`
        back (int): number of lookback frames
        forward (int): number of lookahead frames

    Returns:
        x_sum (FloatTensor): `[B, H_ma, H_ca, qlen, klen]`

    """
    bs, n_heads_mono, n_heads_chunk, qlen, klen = x.size()
    x = x.reshape(-1, klen)
    # Moving sum is computed as a carefully-padded 1D convolution with ones
    x_padded = F.pad(x, pad=[back, forward])  # `[B * H_ma * H_ca * qlen, back + klen + forward]`
    # Add a "channel" dimension
    x_padded = x_padded.unsqueeze(1)
    # Construct filters
    filters = x.new_ones(1, 1, back + forward + 1)
    x_sum = F.conv1d(x_padded, filters)
    x_sum = x_sum.squeeze(1).view(bs, n_heads_mono, n_heads_chunk, qlen, -1)
    return x_sum
