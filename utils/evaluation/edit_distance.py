#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions for computing edit distance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import Levenshtein as lev


def compute_per(ref, hyp, normalize=False):
    """Compute Phone Error Rate.
    Args:
        ref (list): phones in the reference transcript
        hyp (list): phones in the predicted transcript
        normalize (bool, optional): if True, divide by the length of str_true
    Returns:
        per (float): Phone Error Rate between str_true and str_pred
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
    return per


def compute_cer(ref, hyp, normalize=False):
    """Compute Character Error Rate.
    Args:
        ref (string): a sentence without spaces
        hyp (string): a sentence without spaces
        normalize (bool, optional): if True, divide by the length of ref
    Returns:
        cer (float): Character Error Rate between ref and hyp
    """
    cer = lev.distance(hyp, ref)
    if normalize:
        cer /= len(list(ref))
    return cer


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
        sub (int): the number of substitution
        ins (int): the number of insertion
        dele (int): the number of deletion
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
            if d[x][y] == d[x - 1][y - 1] and ref[x - 1] == hyp[y - 1]:
                error_list.append("C")
                x = x - 1
                y = y - 1
            elif d[x][y] == d[x][y - 1] + 1:
                error_list.append("I")
                x = x
                y = y - 1
            elif d[x][y] == d[x - 1][y - 1] + 1:
                error_list.append("S")
                x = x - 1
                y = y - 1
            else:
                error_list.append("D")
                x = x - 1
                y = y

    sub = error_list.count("S")
    ins = error_list.count("I")
    dele = error_list.count("D")
    corr = error_list.count("C")

    assert wer == (sub + ins + dele)
    assert corr == (len(ref) - sub - dele)

    if normalize:
        wer /= len(ref)

    return wer, sub, ins, dele


def wer_align(ref, hyp):
    """Compute Word Error Rate.
        [Reference]
            https://github.com/zszyellow/WER-in-python
    Args:
        ref (list): words in the reference transcript
        hyp (list): words in the predicted transcript
    Returns:
        sub (int): the number of substitution error
        ins (int): the number of insertion error
        dele (int): the number of deletion error
    """
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
                sub = d[i - 1][j - 1] + 1
                ins = d[i][j - 1] + 1
                dele = d[i - 1][j] + 1
                d[i][j] = min(sub, ins, dele)
    result = float(d[len(ref)][len(hyp)]) / len(ref) * 100
    result = str("%.2f" % result) + "%"

    # Find out the manipulation steps
    x = len(ref)
    y = len(hyp)
    error_list = []
    while True:
        if x == 0 and y == 0:
            break
        else:
            if d[x][y] == d[x - 1][y - 1] and ref[x - 1] == hyp[y - 1]:
                error_list.append("C")
                x = x - 1
                y = y - 1
            elif d[x][y] == d[x][y - 1] + 1:
                error_list.append("I")
                x = x
                y = y - 1
            elif d[x][y] == d[x - 1][y - 1] + 1:
                error_list.append("S")
                x = x - 1
                y = y - 1
            else:
                error_list.append("D")
                x = x - 1
                y = y
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
            print(" " * (len(hyp[index])), end=' ')
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
                print(ref[index1] + " " *
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
            print(" " * (len(ref[index])), end=' ')
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
                print(hyp[index2] + " " *
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
            print("D" + " " * (len(ref[index]) - 1), end=' ')
        elif error_list[i] == "I":
            count = 0
            for j in range(i):
                if error_list[j] == "D":
                    count += 1
            index = i - count
            print("I" + " " * (len(hyp[index]) - 1), end=' ')
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
                print("S" + " " * (len(ref[index1]) - 1), end=' ')
            else:
                print("S" + " " * (len(hyp[index2]) - 1), end=' ')
        else:
            count = 0
            for j in range(i):
                if error_list[j] == "I":
                    count += 1
            index = i - count
            print(" " * (len(ref[index])), end=' ')

    print("\nWER: " + result)

    sub = error_list.count("S")
    ins = error_list.count("I")
    dele = error_list.count("D")

    return sub, ins, dele
