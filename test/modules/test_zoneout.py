#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for Zoneout."""

import importlib
import pytest
import torch


def make_args(**kwargs):
    args = dict(
        zoneout_prob_h=0,
        zoneout_prob_c=0,
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "rnn_type, args",
    [
        ('lstm', {'zoneout_prob_h': 0.1}),
        ('lstm', {'zoneout_prob_c': 0.1}),
        ('gru', {'zoneout_prob_h': 0.1}),
        ('gru', {'zoneout_prob_c': 0.1}),
    ]
)
def test_forward(rnn_type, args):
    args = make_args(**args)

    batch_size = 4
    cell_size = 32

    xs = torch.FloatTensor(batch_size, cell_size)
    hxs = torch.zeros(batch_size, cell_size)
    cxs = torch.zeros(batch_size, cell_size) if rnn_type == 'lstm' else None

    if rnn_type == 'lstm':
        cell = torch.nn.LSTMCell(cell_size, cell_size)
    elif rnn_type == 'gru':
        cell = torch.nn.GRUCell(cell_size, cell_size)
    else:
        raise ValueError(rnn_type)
    args['cell'] = cell

    module = importlib.import_module('neural_sp.models.modules.zoneout')
    zoneout_cell = module.ZoneoutCell(**args)

    if rnn_type == 'lstm':
        h, c = zoneout_cell(xs, (hxs, cxs))
        assert h.size() == (batch_size, cell_size)
        assert c.size() == (batch_size, cell_size)
    elif rnn_type == 'gru':
        h = zoneout_cell(xs, hxs)
        assert h.size() == (batch_size, cell_size)
    else:
        raise ValueError(rnn_type)
