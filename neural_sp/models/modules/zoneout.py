# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Zoneout regularization."""

import torch.nn as nn


class ZoneoutCell(nn.Module):

    def __init__(self, cell, zoneout_prob_h, zoneout_prob_c):

        super().__init__()

        self.cell = cell
        self.hidden_size = cell.hidden_size

        if not isinstance(cell, nn.RNNCellBase):
            raise TypeError("The cell is not a LSTMCell or GRUCell!")

        if isinstance(cell, nn.LSTMCell):
            self.prob = (zoneout_prob_h, zoneout_prob_c)
        else:
            self.prob = zoneout_prob_h

    def forward(self, inputs, state):
        """Forward pass.

        Args:
            inputs (FloatTensor): `[B, input_dim]'
            state (tuple or FloatTensor):
        Returns:
            state (tuple or FloatTensor):

        """
        return self.zoneout(state, self.cell(inputs, state), self.prob)

    def zoneout(self, state, next_state, prob):
        if isinstance(state, tuple):
            return (self.zoneout(state[0], next_state[0], prob[0]),
                    self.zoneout(state[1], next_state[1], prob[1]))
        mask = state.new(state.size()).bernoulli_(prob)
        if self.training:
            return mask * next_state + (1 - mask) * state
        else:
            return prob * next_state + (1 - prob) * state


def zoneout_wrapper(cell, zoneout_prob_h=0, zoneout_prob_c=0):
    if zoneout_prob_h > 0 or zoneout_prob_c > 0:
        return ZoneoutCell(cell, zoneout_prob_h, zoneout_prob_c)
    else:
        return cell
