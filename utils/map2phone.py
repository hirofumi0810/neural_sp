#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Map transcription to phone sequence with a lexicon."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import re
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--text', type=str,
                    help='text file')
parser.add_argument('--lexicon', type=str, default='',
                    help='path to lexicon')
args = parser.parse_args()


def main():

    word2phone = {}
    with open(args.lexicon, 'r') as f:
        for line in f:
            word = unicode(line, 'utf-8').strip().split(' ')[0]
            word = word.split('+')[0]  # for CSJ
            phone_seq = ' '.join(unicode(line, 'utf-8').strip().split(' ')[1:])
            word2phone[word] = phone_seq

    utt_count = 0
    with open(args.text, 'r') as f:
        pbar = tqdm(total=len(open(args.text).readlines()))
        for line in f:
            # Remove succesive spaces
            line = re.sub(r'[\s]+', ' ', unicode(line, 'utf-8').strip())
            utt_id = line.split(' ')[0]
            words = line.split(' ')[1:]
            if '' in words:
                words.remove('')

            phones = []
            for w in words:
                phones += word2phone[w].split()
            text_phone = ' '.join(phones)

            print('%s %s' % (utt_id.encode('utf-8'), text_phone.encode('utf-8')))
            utt_count += 1
            pbar.update(1)


if __name__ == '__main__':
    main()
