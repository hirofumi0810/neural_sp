# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Single-head attention layer."""

import numpy as np
import torch
import torch.nn as nn


class AttentionMechanism(nn.Module):
    """Single-head attention layer.

    Args:
        kdim (int): dimension of key
        qdim (int): dimension of query
        atype (str): type of attention mechanisms
        adim: (int) dimension of attention space
        sharpening_factor (float): sharpening factor in the softmax layer
            for attention weights
        sigmoid_smoothing (bool): replace the softmax layer for attention weights
            with the sigmoid function
        conv_out_channels (int): number of channels of conv outputs.
            This is used for location-based attention.
        conv_kernel_size (int): size of kernel.
            This must be the odd number.
        dropout (float): dropout probability for attention weights
        lookahead (int): lookahead frames for triggered attention

    """

    def __init__(self, kdim, qdim, adim, atype,
                 sharpening_factor=1, sigmoid_smoothing=False,
                 conv_out_channels=10, conv_kernel_size=201, dropout=0.,
                 lookahead=2):

        super().__init__()

        assert conv_kernel_size % 2 == 1, "Kernel size should be odd for 'same' conv."
        self.atype = atype
        self.adim = adim
        self.sharpening_factor = sharpening_factor
        self.sigmoid_smoothing = sigmoid_smoothing
        self.n_heads = 1
        self.lookahead = lookahead
        self.reset()

        # attention dropout applied after the softmax layer
        self.dropout = nn.Dropout(p=dropout)

        if atype == 'no':
            raise NotImplementedError
            # NOTE: sequence-to-sequence without attention (use the last state as a context vector)

        elif atype in ['add', 'triggered_attention']:
            self.w_key = nn.Linear(kdim, adim)
            self.w_query = nn.Linear(qdim, adim, bias=False)
            self.v = nn.Linear(adim, 1, bias=False)

        elif atype == 'location':
            self.w_key = nn.Linear(kdim, adim)
            self.w_query = nn.Linear(qdim, adim, bias=False)
            self.w_conv = nn.Linear(conv_out_channels, adim, bias=False)
            self.conv = nn.Conv2d(in_channels=1,
                                  out_channels=conv_out_channels,
                                  kernel_size=(1, conv_kernel_size),
                                  stride=1,
                                  padding=(0, (conv_kernel_size - 1) // 2),
                                  bias=False)
            self.v = nn.Linear(adim, 1, bias=False)

        elif atype == 'dot':
            self.w_key = nn.Linear(kdim, adim, bias=False)
            self.w_query = nn.Linear(qdim, adim, bias=False)

        elif atype == 'luong_dot':
            assert kdim == qdim
            # NOTE: no additional parameters

        elif atype == 'luong_general':
            self.w_key = nn.Linear(kdim, qdim, bias=False)

        elif atype == 'luong_concat':
            self.w = nn.Linear(kdim + qdim, adim, bias=False)
            self.v = nn.Linear(adim, 1, bias=False)

        else:
            raise ValueError(atype)

    def reset(self):
        self.key = None
        self.mask = None

    def forward(self, key, value, query, mask=None, aw_prev=None,
                cache=False, mode='', trigger_points=None, streaming=False):
        """Forward pass.

        Args:
            key (FloatTensor): `[B, klen, kdim]`
            klens (IntTensor): `[B]`
            value (FloatTensor): `[B, klen, vdim]`
            query (FloatTensor): `[B, 1, qdim]`
            mask (ByteTensor): `[B, qlen, klen]`
            aw_prev (FloatTensor): `[B, 1 (H), 1 (qlen), klen]`
            cache (bool): cache key and mask
            mode: dummy interface for MoChA/MMA
            trigger_points (IntTensor): `[B]`
            streaming: dummy interface for streaming attention
        Returns:
            cv (FloatTensor): `[B, 1, vdim]`
            aw (FloatTensor): `[B, 1 (H), 1 (qlen), klen]`
            attn_state (dict): dummy interface

        """
        bs, klen = key.size()[:2]
        qlen = query.size(1)
        attn_state = {}

        if aw_prev is None:
            aw_prev = key.new_zeros(bs, 1, klen)
        else:
            aw_prev = aw_prev.squeeze(1)  # remove head dimension

        # Pre-computation of encoder-side features for computing scores
        if self.key is None or not cache:
            if self.atype in ['add', 'triggered_attention',
                              'location', 'dot', 'luong_general']:
                self.key = self.w_key(key)
            else:
                self.key = key
            self.mask = mask
            if mask is not None:
                assert self.mask.size() == (bs, 1, klen), (self.mask.size(), (bs, 1, klen))

        # for batch beam search decoding
        if self.key.size(0) != query.size(0):
            self.key = self.key[0: 1, :, :].repeat([query.size(0), 1, 1])

        if self.atype == 'no':
            raise NotImplementedError

        elif self.atype in ['add', 'triggered_attention']:
            tmp = self.key.unsqueeze(1) + self.w_query(query).unsqueeze(2)
            e = self.v(torch.tanh(tmp)).squeeze(3)

        elif self.atype == 'location':
            conv_feat = self.conv(aw_prev.unsqueeze(1)).squeeze(2)  # `[B, ch, klen]`
            conv_feat = conv_feat.transpose(2, 1).contiguous().unsqueeze(1)  # `[B, 1, klen, ch]`
            tmp = self.key.unsqueeze(1) + self.w_query(query).unsqueeze(2)
            e = self.v(torch.tanh(tmp + self.w_conv(conv_feat))).squeeze(3)

        elif self.atype == 'dot':
            e = torch.bmm(self.w_query(query), self.key.transpose(2, 1))

        elif self.atype in ['luong_dot', 'luong_general']:
            e = torch.bmm(query, self.key.transpose(2, 1))

        elif self.atype == 'luong_concat':
            query = query.repeat([1, klen, 1])
            e = self.v(torch.tanh(self.w(torch.cat([self.key, query], dim=-1)))).transpose(2, 1)
        assert e.size() == (bs, qlen, klen), (e.size(), (bs, qlen, klen))

        NEG_INF = float(np.finfo(torch.tensor(0, dtype=e.dtype).numpy().dtype).min)

        # Mask the right part from the trigger point
        if self.atype == 'triggered_attention':
            assert trigger_points is not None
            for b in range(bs):
                e[b, :, trigger_points[b] + self.lookahead + 1:] = NEG_INF

        # Compute attention weights, context vector
        if self.mask is not None:
            e = e.masked_fill_(self.mask == 0, NEG_INF)
        if self.sigmoid_smoothing:
            aw = torch.sigmoid(e) / torch.sigmoid(e).sum(-1).unsqueeze(-1)
        else:
            aw = torch.softmax(e * self.sharpening_factor, dim=-1)
        aw = self.dropout(aw)
        cv = torch.bmm(aw, value)

        return cv, aw.unsqueeze(1), attn_state
