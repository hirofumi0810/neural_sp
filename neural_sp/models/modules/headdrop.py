# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""HeadDrop regularization."""

import random

random.seed(1)


def headdrop(aws, n_heads, dropout):
    """HeadDrop regularization.

        Args:
            aws (FloatTensor): `[B, H, qlen, klen]`
            n_heads (int): number of attention heads
            dropout (float): HeadDrop probability
        Returns:
            aws (FloatTensor): `[B, H, qlen, klen]`

    """
    n_effective_heads = n_heads
    head_mask = aws.new_ones(aws.size()).byte()
    for h in range(n_heads):
        if random.random() < dropout:
            head_mask[:, h] = 0
            n_effective_heads -= 1
    aws = aws.masked_fill_(head_mask == 0, 0)
    # Normalization
    if n_effective_heads > 0:
        aws = aws * (n_heads / n_effective_heads)
    return aws
