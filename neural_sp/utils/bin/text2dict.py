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
parser.add_argument('--unit', type=str, choices=['word', "wp", 'char', "phone", "word_char"],
                    help='token units')
parser.add_argument('--vocab_size', type=int, nargs='?',
                    help='the size of vocabulary for word and wordpiece.')
parser.add_argument('--remove_word_boundary', action='store_false',
                    help='remove all whitespaces in the transcriptions')
parser.add_argument('--nlsyms', type=str, default=False,
                    help='path to non-linguistic symbols, e.g., <NOISE> etc.')
parser.add_argument('--wp_type', type=str, default='unigram', nargs='?',
                    choices=['unigram', 'bpe'],
                    help='')
parser.add_argument('--wp_model', type=str, default=False, nargs='?',
                    help='prefix of the wordpiece model')
args = parser.parse_args()


# TODO(hirofumi): python sentencepiece shows different behaviors from bash command.

def main():

    nlsyms = []
    if args.nlsyms:
        with open(args.nlsyms, 'r') as f:
            for line in f:
                nlsyms.append(unicode(line, 'utf-8').strip())

    if args.unit == 'wp':
        spm.SentencePieceTrainer.Train('--input=' + args.text +
                                       ' --user_defined_symbols=' + ','.join(nlsyms) +
                                       ' --vocab_size=' + str(args.vocab_size) +
                                       ' --model_type=' + args.wp_type +
                                       ' --model_prefix=' + args.wp_model +
                                       ' --input_sentence_size=100000000')
        sp = spm.SentencePieceProcessor()
        sp.Load(args.wp_model + '.model')

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

            words = line.split()[1:]
            if '' in words:
                words.remove('')
            text = ' '.join(words)

            if args.unit in ['word', 'word_char']:
                for w in words:
                    # Count word frequency
                    if w not in word_dict.keys():
                        word_dict[w] = 1
                    else:
                        word_dict[w] += 1

            elif args.unit == 'wp':
                token_set |= set(sp.EncodeAsPieces(text))

            elif args.unit in ['char', 'word_char']:
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
        token_list = sorted(nlsyms) + sorted(list(word_dict.keys()),
                                             key=lambda x: word_dict[x],
                                             reverse=True)[:args.vocab_size]

    elif args.unit == 'word_char':
        word_char_list = sorted(list(word_dict.keys()),
                                key=lambda x: word_dict[x],
                                reverse=True)[:args.vocab_size]
        word_char_list += sorted(list(token_set))
        token_list = sorted(nlsyms) + sorted(list(set(word_char_list)))

    elif args.unit == 'wp':
        token_list = sorted(nlsyms) + sorted(list(token_set))

    elif args.unit == 'char':
        token_list = sorted(nlsyms) + sorted(list(token_set))

    elif args.unit == 'phone':
        token_list = sorted(list(token_set))

    for t in token_list:
        print('%s' % t.encode('utf-8'))


if __name__ == '__main__':
    main()
