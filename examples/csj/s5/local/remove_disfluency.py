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


def main():

    with codecs.open(args.text, 'r', encoding="utf-8") as f:
        pbar = tqdm(total=len(open(args.text).readlines()))
        for line in f:
            line = line.strip()

            utt_id = line.split()[0]
            words = line.split()[1:]

            # Remove filler and disfluency
            text = ' '.join(['' if '言いよどみ' in w or '感動詞' in w else w for w in words])

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
