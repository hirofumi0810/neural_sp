# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Continuous integrate and fire (CIF)."""

import logging
import torch
import torch.nn as nn

from neural_sp.models.modules.initialization import init_with_xavier_uniform
from neural_sp.models.torch_utils import make_pad_mask

logger = logging.getLogger(__name__)


class CIF(nn.Module):
    """Continuous integrate and fire (CIF).

    Args:
        enc_dim (int): dimension of encoder outputs
        window (int): kernel size of 1dconv
        threshold (int): boundary threshold (equivalent to beta in the paper)
        param_init (int): parameter initialization method
        layer_norm_eps (int): epsilon value for layer normalization

    """

    def __init__(self, enc_dim, window, threshold=1.0,
                 param_init='', layer_norm_eps=1e-12):

        super().__init__()

        self.enc_dim = enc_dim
        self.beta = threshold
        assert (window - 1) % 2 == 0, 'window must be the odd number.'

        self.conv1d = nn.Conv1d(in_channels=enc_dim,
                                out_channels=enc_dim,
                                kernel_size=window,
                                stride=1,
                                padding=(window - 1) // 2)
        self.norm = nn.LayerNorm(enc_dim, eps=layer_norm_eps)
        self.proj = nn.Linear(enc_dim, 1)

        if param_init == 'xavier_uniform':
            self.reset_parameters()
        else:
            logger.info('Parameter initialization is skipped.')

    def reset_parameters(self):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' %
                    self.__class__.__name__)
        for n, p in self.named_parameters():
            init_with_xavier_uniform(n, p)

    def forward(self, eouts, elens, ylens=None, mode='parallel', streaming=False):
        """Forward pass.

        Args:
            eouts (FloatTensor): `[B, T, enc_dim]`
            elens (IntTensor): `[B]`
            ylens (IntTensor): `[B]`
            mode (str): parallel/incremental
            streaming: dummy interface for streaming attention
        Returns:
            cv (FloatTensor): `[B, L, enc_dim]`
            aws (FloatTensor): `[B, L, T]`
            attn_state (dict): dummy interface
                alpha (FloatTensor): `[B, T]`

        """
        bs, xmax, enc_dim = eouts.size()
        attn_state = {}

        # 1d conv
        conv_feat = self.conv1d(eouts.transpose(2, 1)).transpose(2, 1)  # `[B, T, enc_dim]`
        conv_feat = torch.relu(self.norm(conv_feat))
        alpha = torch.sigmoid(self.proj(conv_feat)).squeeze(2)  # `[B, T]`

        # normalization
        if mode == 'parallel':
            # padding
            assert ylens is not None
            device = eouts.device
            ylens = ylens.to(device)
            mask = make_pad_mask(elens.to(device))
            alpha = alpha.clone().masked_fill_(mask == 0, 0)

            alpha_norm = alpha / alpha.sum(1, keepdim=True) * ylens.float().unsqueeze(1)
            ymax = int(ylens.max().item())
        elif mode == 'incremental':
            alpha_norm = alpha  # infernece time
            ymax = 1
            if bs > 1:
                raise NotImplementedError('Batch mode is not supported.')
                # TODO(hirofumi0810): support batch mode
        else:
            raise ValueError(mode)

        cv = eouts.new_zeros(bs, ymax + 1, enc_dim)
        aws = eouts.new_zeros(bs, ymax + 1, xmax)
        n_tokens = torch.zeros(bs, dtype=torch.int64)
        state = eouts.new_zeros(bs, self.enc_dim)
        alpha_accum = eouts.new_zeros(bs)
        for j in range(xmax):
            alpha_accum_prev = alpha_accum
            alpha_accum += alpha_norm[:, j]

            if mode == 'parallel' and (alpha_accum >= self.beta).sum() == 0:
                # No boundary is located in all utterances in mini-batch
                # Carry over to the next frame
                state += alpha_norm[:, j, None] * eouts[:, j]
                aws[:, n_tokens, j] += alpha_norm[:, j]
            else:
                for b in range(bs):
                    # skip the padding region
                    if j > elens[b] - 1:
                        continue

                    # skip all-fired utterance
                    if mode == 'parallel' and n_tokens[b].item() >= ylens[b]:
                        continue

                    if alpha_accum[b] < self.beta:
                        # No boundary is located
                        # Carry over to the next frame
                        state[b] += alpha_norm[b, j, None] * eouts[b, j]
                        aws[b, n_tokens[b], j] += alpha_norm[b, j]

                        # tail handling
                        if mode == 'incremental' and j == elens[b] - 1:
                            if alpha_accum[b] >= 0.5:
                                n_tokens[b] += 1
                                cv[b, n_tokens[b]] = state[b]
                            break
                    else:
                        # A boundary is located
                        ak1 = 1 - alpha_accum_prev[b]
                        ak2 = alpha_norm[b, j] - ak1
                        cv[b, n_tokens[b]] = state[b] + ak1 * eouts[b, j]
                        aws[b, n_tokens[b], j] += ak1
                        n_tokens[b] += 1
                        # Carry over to the next frame
                        state[b] = ak2 * eouts[b, j]
                        alpha_accum[b] = ak2
                        aws[b, n_tokens[b], j] += ak2

                        if mode == 'incremental':
                            break

                if mode == 'incremental' and n_tokens[0] >= 1:
                    break
                    # TODO(hirofumi0810): support batch mode

        # truncate
        cv = cv[:, :ymax]
        aws = aws[:, :ymax]
        attn_state['alpha'] = alpha

        return cv, aws, attn_state
