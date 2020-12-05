#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Map transcription to phone sequence with a lexicon."""

import argparse
import codecs
import re
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--text', type=str,
                    help='text file')
parser.add_argument('--lexicon', type=str, default='',
                    help='path to lexicon')
parser.add_argument('--unk', type=str, default='NSN', nargs='?',
                    help='phone corresponding to unknown marks <unk> in lexicon')
args = parser.parse_args()


def main():

    word2phone = {}
    with codecs.open(args.lexicon, 'r', encoding="utf-8") as f:
        for line in f:
            word = line.strip().split(' ')[0]
            word = word.split('+')[0]  # for CSJ
            word = word.lower()  # for Librispeech
            phone_seq = ' '.join(line.strip().split(' ')[1:])
            word2phone[word] = phone_seq

    utt_count = 0
    with codecs.open(args.text, 'r', encoding="utf-8") as f:
        pbar = tqdm(total=len(codecs.open(args.text, 'r', encoding="utf-8").readlines()))
        for line in f:
            # Remove successive spaces
            line = re.sub(r'[\s]+', ' ', line.strip())
            utt_id = line.split(' ')[0]
            words = line.split(' ')[1:]
            if '' in words:
                words.remove('')

            phones = []
            for w in words:
                if w in word2phone:
                    phones += word2phone[w].split()
                else:
                    phones += [args.unk]
            text_phone = ' '.join(phones)

            print('%s %s' % (utt_id, text_phone))
            utt_count += 1
            pbar.update(1)


if __name__ == '__main__':
    main()
