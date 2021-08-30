# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Utility functions."""

import copy
import numpy as np
import torch


def repeat(module, n_layers):
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(n_layers)])


def tensor2np(x):
    """Convert torch.Tensor to np.ndarray.

    Args:
        x (torch.Tensor):
    Returns:
        np.ndarray

    """
    if x is None:
        return x
    return x.cpu().detach().numpy()


def tensor2scalar(x):
    """Convert torch.Tensor to a scalar value.

    Args:
        x (torch.Tensor):
    Returns:
        scaler

    """
    if isinstance(x, float):
        return x
    return x.cpu().detach().item()


def np2tensor(array, device=None):
    """Convert form np.ndarray to torch.Tensor.

    Args:
        array (np.ndarray): tensor of any sizes
    Returns:
        tensor (torch.Tensor):

    """
    tensor = torch.from_numpy(array).to(device)
    return tensor


def pad_list(xs, pad_value=0., pad_left=False):
    """Convert list of Tensors to a single Tensor with padding.

    Args:
        xs (List[Tensor]): length `[B]`, which contains Tensors of size `[T, input_size]`
        pad_value (float):
        pad_left (bool):
    Returns:
        xs_pad (FloatTensor): `[B, T, input_size]`

    """
    bs = len(xs)
    max_time = max(x.size(0) for x in xs)
    xs_pad = xs[0].new_zeros(bs, max_time, * xs[0].size()[1:]).fill_(pad_value)
    for b in range(bs):
        if len(xs[b]) == 0:
            continue
        if pad_left:
            xs_pad[b, -xs[b].size(0):] = xs[b]
        else:
            xs_pad[b, :xs[b].size(0)] = xs[b]
    return xs_pad


def make_pad_mask(seq_lens):
    """Make mask for padding.

    Args:
        seq_lens (IntTensor): `[B]`
    Returns:
        mask (IntTensor): `[B, T]`

    """
    bs = seq_lens.size(0)
    max_time = seq_lens.max()
    seq_range = torch.arange(0, max_time, dtype=torch.int32, device=seq_lens.device)
    seq_range = seq_range.unsqueeze(0).expand(bs, max_time)
    mask = seq_range < seq_lens.unsqueeze(-1)
    return mask


def append_sos_eos(ys, sos, eos, pad, device, bwd=False, replace_sos=False):
    """Append <sos> and <eos> and return padded sequences.

    Args:
        ys (List[List]): length `[B]`, which contains a list of size `[L]`
        sos (int): index for <sos>
        eos (int): index for <eos>
        pad (int): index for <pad>
        device ():
        bwd (bool): reverse ys for backward reference
        replace_sos (bool): replace <sos> with the special token
    Returns:
        ys_in (LongTensor): `[B, L]`
        ys_out (LongTensor): `[B, L]`
        ylens (IntTensor): `[B]`

    """
    _eos = torch.zeros(1, dtype=torch.int64, device=device).fill_(eos)
    ys = [np2tensor(np.fromiter(y[::-1] if bwd else y, dtype=np.int64),
                    device) for y in ys]
    if replace_sos:
        ylens = np2tensor(np.fromiter([y[1:].size(0) + 1 for y in ys], dtype=np.int32))  # +1 for <eos>
        ys_in = pad_list([y for y in ys], pad)
        ys_out = pad_list([torch.cat([y[1:], _eos], dim=0) for y in ys], pad)
    else:
        _sos = torch.zeros(1, dtype=torch.int64, device=device).fill_(sos)
        ylens = np2tensor(np.fromiter([y.size(0) + 1 for y in ys], dtype=np.int32))  # +1 for <eos>
        ys_in = pad_list([torch.cat([_sos, y], dim=0) for y in ys], pad)
        ys_out = pad_list([torch.cat([y, _eos], dim=0) for y in ys], pad)
    return ys_in, ys_out, ylens


def compute_accuracy(logits, ys_ref, pad):
    """Compute teacher-forcing accuracy.

    Args:
        logits (FloatTensor): `[B, T, vocab]`
        ys_ref (LongTensor): `[B, T]`
        pad (int): index for padding
    Returns:
        acc (float): teacher-forcing accuracy

    """
    pad_pred = logits.view(ys_ref.size(0), ys_ref.size(1), logits.size(-1)).argmax(2)
    mask = ys_ref != pad
    numerator = torch.sum(pad_pred.masked_select(mask) == ys_ref.masked_select(mask))
    denominator = torch.sum(mask)
    acc = float(numerator) * 100 / float(denominator)
    return acc
