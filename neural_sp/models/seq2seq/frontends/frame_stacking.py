# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Frame stacking."""

import numpy as np


def stack_frame(x, n_stacks, n_skips, dtype=np.float32):
    """Stack & skip some frames. This implementation is based on

       https://arxiv.org/abs/1507.06947.
           Sak, Haşim, et al.
           "Fast and accurate recurrent neural network acoustic models for speech recognition."
           arXiv preprint arXiv:1507.06947 (2015).

    Args:
        x (np.ndarray): `[T, input_dim]`
        n_stacks (int): the number of frames to stack
        n_skips (int): the number of frames to skip
        dtype ():
    Returns:
        stacked_feat (np.ndarray): `[floor(T / n_skips), input_dim * n_stacks]`

    """
    if n_stacks == 1 and n_skips == 1:
        return x
    if n_stacks < n_skips:
        raise ValueError('n_skips must be less than n_stacks.')
    assert isinstance(x, np.ndarray), 'x should be np.ndarray.'

    T, input_dim = x.shape
    T_new = T // n_skips if T % n_stacks == 0 else (T // n_skips) + 1

    stacked_feat = np.zeros((T_new, input_dim * n_stacks), dtype=dtype)
    stack_count = 0
    stack = []
    for t, frame_t in enumerate(x):
        if t == len(x) - 1:  # final frame
            # Stack the final frame
            stack.append(frame_t)

            while stack_count != int(T_new):
                # Concatenate stacked frames
                for i in range(len(stack)):
                    stacked_feat[stack_count][input_dim * i:input_dim * (i + 1)] = stack[i]
                stack_count += 1

                # Delete some frames to skip
                for _ in range(n_skips):
                    if len(stack) != 0:
                        stack.pop(0)

        elif len(stack) < n_stacks:  # first & middle frames
            # Stack some frames until stack is filled
            stack.append(frame_t)

        if len(stack) == n_stacks:
            # Concatenate stacked frames
            for i in range(n_stacks):
                stacked_feat[stack_count][input_dim * i:input_dim * (i + 1)] = stack[i]
            stack_count += 1

            # Delete some frames to skip
            for _ in range(n_skips):
                stack.pop(0)

    return stacked_feat
