#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Functions for computing edit distance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import Levenshtein as lev  # TODO(hirofumi): install
import numpy as np


def compute_per(ref, hyp, normalize=False):
    """Compute Phone Error Rate.

    Args:
        ref (list): phones in the reference transcript
        hyp (list): phones in the predicted transcript
        normalize (bool, optional): if True, divide by the length of ref
    Returns:
        per (float): Phone Error Rate between ref and hyp

    """
    # Build mapping of phone to index
    phone_set = set(ref + hyp)
    phone2char = dict(zip(phone_set, range(len(phone_set))))

    # Map phones to a single char array
    # NOTE: Levenshtein packages only accepts strings
    phones_ref = [chr(phone2char[p]) for p in ref]
    phones_hyp = [chr(phone2char[p]) for p in hyp]

    per = lev.distance(''.join(phones_ref), ''.join(phones_hyp))
    if normalize:
        per /= len(ref)
    return per * 100


def compute_cer(ref, hyp, normalize=False):
    """Compute Character Error Rate.

    Args:
        ref (str): a sentence without spaces
        hyp (str): a sentence without spaces
        normalize (bool, optional): if True, divide by the length of ref
    Returns:
        cer (float): Character Error Rate between ref and hyp

    """
    cer = lev.distance(hyp, ref)
    if normalize:
        cer /= len(list(ref))
    return cer * 100


def compute_wer(ref, hyp, normalize=False):
    """Compute Word Error Rate.

        [Reference]
            https://martin-thoma.com/word-error-rate-calculation/
    Args:
        ref (list): words in the reference transcript
        hyp (list): words in the predicted transcript
        normalize (bool, optional): if True, divide by the length of ref
    Returns:
        wer (float): Word Error Rate between ref and hyp
        n_sub (int): the number of substitution
        n_ins (int): the number of insertion
        n_del (int): the number of deletion

    """
    # Initialisation
    d = np.zeros((len(ref) + 1) * (len(hyp) + 1), dtype=np.uint16)
    d = d.reshape((len(ref) + 1, len(hyp) + 1))
    for i in range(len(ref) + 1):
        for j in range(len(hyp) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # Computation
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                sub_tmp = d[i - 1][j - 1] + 1
                ins_tmp = d[i][j - 1] + 1
                del_tmp = d[i - 1][j] + 1
                d[i][j] = min(sub_tmp, ins_tmp, del_tmp)

    wer = d[len(ref)][len(hyp)]

    # Find out the manipulation steps
    x = len(ref)
    y = len(hyp)
    error_list = []
    while True:
        if x == 0 and y == 0:
            break
        else:
            if x > 0 and y > 0:
                if d[x][y] == d[x - 1][y - 1] and ref[x - 1] == hyp[y - 1]:
                    error_list.append("C")
                    x = x - 1
                    y = y - 1
                elif d[x][y] == d[x][y - 1] + 1:
                    error_list.append("I")
                    y = y - 1
                elif d[x][y] == d[x - 1][y - 1] + 1:
                    error_list.append("S")
                    x = x - 1
                    y = y - 1
                else:
                    error_list.append("D")
                    x = x - 1
            elif x == 0 and y > 0:
                if d[x][y] == d[x][y - 1] + 1:
                    error_list.append("I")
                    y = y - 1
                else:
                    error_list.append("D")
                    x = x - 1
            elif y == 0 and x > 0:
                error_list.append("D")
                x = x - 1
            else:
                raise ValueError

    n_sub = error_list.count("S")
    n_ins = error_list.count("I")
    n_del = error_list.count("D")
    n_cor = error_list.count("C")

    assert wer == (n_sub + n_ins + n_del)
    assert n_cor == (len(ref) - n_sub - n_del)

    if normalize:
        wer /= len(ref)

    return wer * 100, n_sub * 100, n_ins * 100, n_del * 100


def wer_align(ref, hyp, normalize=False, double_byte=False):
    """Compute Word Error Rate.

        [Reference]
            https://github.com/zszyellow/WER-in-python
    Args:
        ref (list): words in the reference transcript
        hyp (list): words in the predicted transcript
        normalize (bool, optional): if True, divide by the length of ref
        double_byte (bool):
    Returns:
        wer (float): Word Error Rate between ref and hyp
        n_sub (int): the number of substitution error
        n_ins (int): the number of insertion error
        n_del (int): the number of deletion error

    """
    space_char = "　" if double_byte else " "
    s_char = "Ｓ" if double_byte else "S"
    i_char = "Ｉ" if double_byte else "I"
    d_char = "Ｄ" if double_byte else "D"

    # Build the matrix
    d = np.zeros((len(ref) + 1) * (len(hyp) + 1),
                 dtype=np.uint8).reshape((len(ref) + 1, len(hyp) + 1))
    for i in range(len(ref) + 1):
        for j in range(len(hyp) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                n_sub = d[i - 1][j - 1] + 1
                n_ins = d[i][j - 1] + 1
                n_del = d[i - 1][j] + 1
                d[i][j] = min(n_sub, n_ins, n_del)
    wer = float(d[len(ref)][len(hyp)])

    # Find out the manipulation steps
    x = len(ref)
    y = len(hyp)
    error_list = []
    while True:
        if x == 0 and y == 0:
            break
        else:
            if x > 0 and y > 0:
                if d[x][y] == d[x - 1][y - 1] and ref[x - 1] == hyp[y - 1]:
                    error_list.append("C")
                    x = x - 1
                    y = y - 1
                elif d[x][y] == d[x][y - 1] + 1:
                    error_list.append("I")
                    y = y - 1
                elif d[x][y] == d[x - 1][y - 1] + 1:
                    error_list.append("S")
                    x = x - 1
                    y = y - 1
                else:
                    error_list.append("D")
                    x = x - 1
            elif x == 0 and y > 0:
                if d[x][y] == d[x][y - 1] + 1:
                    error_list.append("I")
                    y = y - 1
                else:
                    error_list.append("D")
                    x = x - 1
            elif y == 0 and x > 0:
                error_list.append("D")
                x = x - 1
            else:
                raise ValueError
    error_list = error_list[::-1]

    # Print the result in aligned way
    print("REF: ", end='')
    for i in range(len(error_list)):
        if error_list[i] == "I":
            count = 0
            for j in range(i):
                if error_list[j] == "D":
                    count += 1
            index = i - count
            print(space_char * (len(hyp[index])), end=' ')
        elif error_list[i] == "S":
            count1 = 0
            for j in range(i):
                if error_list[j] == "I":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if error_list[j] == "D":
                    count2 += 1
            index2 = i - count2
            if len(ref[index1]) < len(hyp[index2]):
                print(ref[index1] + space_char *
                      (len(hyp[index2]) - len(ref[index1])), end=' ')
            else:
                print(ref[index1], end=' ')
        else:
            count = 0
            for j in range(i):
                if error_list[j] == "I":
                    count += 1
            index = i - count
            print(ref[index], end=' ')

    print("\nHYP: ", end='')
    for i in range(len(error_list)):
        if error_list[i] == "D":
            count = 0
            for j in range(i):
                if error_list[j] == "I":
                    count += 1
            index = i - count
            print(space_char * (len(ref[index])), end=' ')
        elif error_list[i] == "S":
            count1 = 0
            for j in range(i):
                if error_list[j] == "I":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if error_list[j] == "D":
                    count2 += 1
            index2 = i - count2
            if len(ref[index1]) > len(hyp[index2]):
                print(hyp[index2] + space_char *
                      (len(ref[index1]) - len(hyp[index2])), end=' ')
            else:
                print(hyp[index2], end=' ')
        else:
            count = 0
            for j in range(i):
                if error_list[j] == "D":
                    count += 1
            index = i - count
            print(hyp[index], end=' ')

    print("\nEVA: ", end='')
    for i in range(len(error_list)):
        if error_list[i] == "D":
            count = 0
            for j in range(i):
                if error_list[j] == "I":
                    count += 1
            index = i - count
            print(d_char + space_char * (len(ref[index]) - 1), end=' ')
        elif error_list[i] == "I":
            count = 0
            for j in range(i):
                if error_list[j] == "D":
                    count += 1
            index = i - count
            print(i_char + space_char * (len(hyp[index]) - 1), end=' ')
        elif error_list[i] == "S":
            count1 = 0
            for j in range(i):
                if error_list[j] == "I":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if error_list[j] == "D":
                    count2 += 1
            index2 = i - count2
            if len(ref[index1]) > len(hyp[index2]):
                print(s_char + space_char * (len(ref[index1]) - 1), end=' ')
            else:
                print(s_char + space_char * (len(hyp[index2]) - 1), end=' ')
        else:
            count = 0
            for j in range(i):
                if error_list[j] == "I":
                    count += 1
            index = i - count
            print(space_char * (len(ref[index])), end=' ')

    n_sub = error_list.count("S")
    n_ins = error_list.count("I")
    n_del = error_list.count("D")
    n_cor = error_list.count("C")

    assert wer == (n_sub + n_ins + n_del)
    assert n_cor == (len(ref) - n_sub - n_del)

    if normalize:
        wer /= len(ref)

    return wer * 100, n_sub * 100, n_ins * 100, n_del * 100
