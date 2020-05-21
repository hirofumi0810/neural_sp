#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Make a dictionary file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import codecs
from distutils.util import strtobool
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('text', type=str,
                    help='path to text file')
parser.add_argument('--unit', type=str,
                    choices=['word', 'wp', 'char', 'phone', 'word_char'],
                    help='token units')
parser.add_argument('--vocab_size', type=int, nargs='?',
                    help='the size of vocabulary for word and wordpiece.')
parser.add_argument('--remove_word_boundary', action='store_false',
                    help='remove all whitespaces in the transcriptions')
parser.add_argument('--nlsyms', type=str, default=False, nargs='?',
                    help='path to non-linguistic symbols, e.g., <NOISE> etc.')
parser.add_argument('--speed_perturb', type=strtobool, default=False,
                    help='use speed perturbation.')
args = parser.parse_args()


# TODO(hirofumi): python sentencepiece shows different behaviors from bash command.

def main():

    nlsyms = []
    if args.nlsyms:
        with codecs.open(args.nlsyms, 'r', encoding="utf-8") as f:
            for line in f:
                nlsyms.append(line.strip())

    if args.unit == 'wp':
        raise ValueError("Use spm_encode in the bash script instead of text2dict.py.")

    word_dict = {}
    token_set = set([])
    with codecs.open(args.text, 'r', encoding="utf-8") as f:
        pbar = tqdm(total=len(codecs.open(args.text, 'r', encoding="utf-8").readlines()))
        for line in f:
            line = line.strip()

            if args.speed_perturb and 'sp1.0' not in line:
                pbar.update(1)
                continue

            words = line.split()[1:]
            if '' in words:
                words.remove('')

            # Remove special tokens
            for nlsym in nlsyms:
                # Include in the dictionary to sort by frequency
                if args.unit in ['word', 'word_char']:
                    if nlsym not in word_dict.keys():
                        word_dict[nlsym] = words.count(nlsym)
                    else:
                        word_dict[nlsym] += words.count(nlsym)

                while True:
                    if nlsym in words:
                        words.remove(nlsym)
                    else:
                        break

            text = ' '.join(words)

            if args.unit in ['word', 'word_char']:
                for w in words:
                    # Count word frequency
                    if w not in word_dict.keys():
                        word_dict[w] = 1
                    else:
                        word_dict[w] += 1

                if args.unit == 'word_char':
                    token_set |= set(list(text))

            elif args.unit == 'char':
                # Remove whitespaces
                if args.remove_word_boundary:
                    text = text.replace(' ', '')

                token_set |= set(list(text))

            elif args.unit == 'phone':
                token_set |= set(words)

            else:
                raise ValueError(args.unit)
            pbar.update(1)

    if args.unit in ['word']:
        token_list = sorted(list(word_dict.keys()),
                            key=lambda x: word_dict[x],
                            reverse=True)[:args.vocab_size]
        # NOTE: nlsyms are already included in the word_dict

    elif args.unit == 'word_char':
        word_char_list = sorted(list(word_dict.keys()),
                                key=lambda x: word_dict[x],
                                reverse=True)[:args.vocab_size] + list(token_set)
        token_list = sorted(list(set(word_char_list)))
        # NOTE: nlsyms are already included in the word_dict

    elif args.unit in ['char']:
        token_list = sorted(nlsyms) + sorted(list(token_set))

    elif args.unit == 'phone':
        token_list = sorted(list(token_set))
        # NOTE: nlsyms are already included in the phone set

    for t in token_list:
        print('%s' % t)


if __name__ == '__main__':
    main()
