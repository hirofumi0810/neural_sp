#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from chainer import Variable


def np2var(inputs):
    """
    Args:
        inputs (np.ndarray): A tensor of size `[B, T, input_size]`
    Returns:
        Variable of size `[T, B, input_size]`
    """
    return Variable(inputs, requires_grad=False)


def np2varlist_(inputs):
    """
    Args:
        inputs (np.ndarray): A tensor of size `[B, T, input_size]`
    Returns:
        var_list (list): list of character.Variable of size `[T, input_size]`
            Note that len(var_list) == B.
    """
    assert len(inputs.shape) == 3

    var_list = []
    for i_batch in range(inputs.shape[0]):
        var_list.append(Variable(inputs[i_batch], requires_grad=False))
    # volatile??

    return var_list
