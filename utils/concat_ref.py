#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Concatenate references for the same speaker."""

import argparse
import codecs
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('text', type=str,
                    help='path to text file')
parser.add_argument('utt2spk', type=str,
                    help='path to utt2spk file')
args = parser.parse_args()


def main():

    utt2spk = {}
    with codecs.open(args.utt2spk, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            utt_id, speaker_id = line.split()
            speaker_id = speaker_id.split('-')[0]  # for TEDLIUM2
            utt2spk[utt_id] = speaker_id

    refereces = {}
    with codecs.open(args.text, 'r', encoding="utf-8") as f:
        pbar = tqdm(total=len(codecs.open(args.text, 'r', encoding="utf-8").readlines()))
        for line in f:
            line = line.strip()

            utt_id = line.split()[0]
            speaker_id = utt2spk[utt_id]

            words = line.split()[1:]
            if '' in words:
                words.remove('')
            text = ' '.join(words)

            if speaker_id not in refereces.keys():
                refereces[speaker_id] = text
            else:
                refereces[speaker_id] += ' <eos> ' + text

            pbar.update(1)

    for k, v in refereces.items():
        print('%s %s' % (k, v))


if __name__ == '__main__':
    main()
