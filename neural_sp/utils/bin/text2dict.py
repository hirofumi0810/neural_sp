#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Make dataset CSV files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sentencepiece as spm
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('text', type=str,
                    help='path to text file')
parser.add_argument('--unit', type=str, choices=['word', "bpe", 'char', "phone"],
                    help='token units')
parser.add_argument('--vocab_size', type=int,
                    help='the size of vocabulary for word and bpe.')
parser.add_argument('--remove_word_boundary', action='store_false',
                    help='')
parser.add_argument('--nlsyms', type=str, default=False,
                    help='path to non-linguistic symbols, e.g., <NOISE> etc.')
parser.add_argument('--bpe_model_type', type=str, choices=['unigram', 'bpe'],
                    help='')
parser.add_argument('--bpe_model', type=str,
                    help='')
args = parser.parse_args()


def main():

    nlsyms = []
    if args.nlsyms:
        with open(args.nlsyms, 'r') as f:
            for line in f:
                nlsyms.append(line.strip().encode('utf-8'))

    if args.unit == 'bpe':
        # TODO: CSJ
        # words = list(map(lambda x: x.split('+')[0], words[1:]))

        spm.SentencePieceTrainer.Train('--input=' + args.text +
                                       ' --vocab_size=' + str(args.vocab_size) +
                                       ' --model_type=' + args.bpe_model_type +
                                       ' --model_prefix=' + args.bpe_model +
                                       ' --input_sentence_size=100000000')
    else:
        word_dict = {}
        word2phone = {}
        token_set = set([])
        with open(args.text, 'r') as f:
            pbar = tqdm(total=len(open(args.text).readlines()))
            for line in f:
                line = unicode(line, 'utf-8').strip()

                # Remove special tokens
                for token in nlsyms:
                    line = line.replace(token, '')

                words = line.split()
                if '' in words:
                    words.remove('')

                # for CSJ
                words = list(map(lambda x: x.split('+')[0], words[1:]))
                text = ' '.join(words)

                if args.unit == 'word':
                    for w in words:
                        # Count word frequency
                        if w not in word_dict.keys():
                            word_dict[w] = 1
                        else:
                            word_dict[w] += 1
                        token_set.add(w)

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

    if args.unit == 'word':
        word_list = sorted(nlsyms) + sorted(list(word_dict.keys()),
                                            key=lambda x: word_dict[x],
                                            reverse=True)[:args.vocab_size]
        for w in word_list:
            print('%s' % w.encode('utf-8'))

    elif args.unit == 'bpe':
        raise NotImplementedError()

    elif args.unit == 'char':
        for c in sorted(nlsyms) + sorted(list(token_set)):
            print('%s' % c.encode('utf-8'))

    elif args.unit == 'phone':
        for p in sorted(list(token_set)):
            print('%s' % p.encode('utf-8'))


if __name__ == '__main__':
    main()
