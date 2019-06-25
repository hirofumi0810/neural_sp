#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Remove filler and disfluency based on POS tag."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import codecs
import re
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('text', type=str,
                    help='path to text file')
args = parser.parse_args()


repeat_2gram_exception = ['bye bye']


def main():

    with codecs.open(args.text, 'r', encoding="utf-8") as f:
        pbar = tqdm(total=len(open(args.text).readlines()))
        for line in f:
            line = line.strip()

            utt_id = line.split()[0]
            text = ' '.join(line.split()[1:])

            # Remove repeats words (without interventing words) at first
            w_prev = ''
            w_prev_2gram = []
            words = []
            for w in text.split(' '):
                if w_prev + ' ' + w in ' '.join(w_prev_2gram):
                    words = words[:-1]  # remove the last word
                elif w != w_prev or (w_prev + ' ' + w in repeat_2gram_exception):
                    words.append(w)

                w_prev_2gram.append(w)
                w_prev_2gram = w_prev_2gram[-3:]  # pruning
                w_prev = w
            text = ' '.join(words)

            # Remove noise
            text = re.sub(r'\[noise\]', ' ', text)
            text = re.sub(r'\[laughter\]', ' ', text)
            text = re.sub(r'\[vocalized-noise\]', ' ', text)

            # Remove filled pause (filler and backchannel)
            for w in ['uh-huh', 'um-hum', 'mhm', 'mmhm', 'mm-hm', 'mm-huh', 'huh-uh',
                      'uhhuh', 'uhuh',
                      'uh', 'um', 'eh', 'mm', 'hm', 'ah', 'huh', 'ha', 'er', 'oof', 'hee', 'ach', 'eee', 'ew',
                      'you know', 'i mean']:
                text = re.sub(r'\A%s\s+' % w, ' ', text)  # start
                text = re.sub(r'\s+%s\Z' % w, ' ', text)  # end
                text = re.sub(r'\A%s\Z' % w, ' ', text)  # single
                text = re.sub(r'\s+%s\s+' % w, ' ', text)  # middle

            # Remove fragment (partial words)
            text = re.sub(r'\A([^\s]+-)\s+', ' ', text)  # start
            text = re.sub(r'\s+([^\s]+-)\Z', ' ', text)  # end
            text = re.sub(r'\A([^\s]+-)\Z', ' ', text)  # single
            text = re.sub(r'\s+([^\s]+-)\s+', ' ', text)  # middle

            text = re.sub(r'\A(-[^\s]+)\s+', ' ', text)  # start
            text = re.sub(r'\s+(-[^\s]+)\Z', ' ', text)  # end
            text = re.sub(r'\A(-[^\s]+)\Z', ' ', text)  # single
            text = re.sub(r'\s+(-[^\s]+)\s+', ' ', text)  # middle

            # Remove conseccutive spaces
            text = re.sub(r'[\s]+', ' ', text)

            # Remove the first and last spaces
            if len(text) > 0 and text[0] == ' ':
                text = text[1:]
            if len(text) > 0 and text[-1] == ' ':
                text = text[:-1]

            if len(text) > 0:
                print(utt_id + ' ' + text)
            pbar.update(1)


if __name__ == '__main__':
    main()
