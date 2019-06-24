#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Remove POS tag."""

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

            # Remove POS tag
            text = ' '.join(list(map(lambda x: x.split('+')[0], words)))

            # Remove <sp> (short pause)
            text = text.replace('<sp>', '')

            # Remove conseccutive spaces
            text = re.sub(r'[\s]+', ' ', text)

            # Remove the first and last spaces
            if len(text) > 0 and text[0] == ' ':
                text = text[1:]
            if len(text) > 0 and text[-1] == ' ':
                text = text[:-1]

            print(utt_id + ' ' + text)
            pbar.update(1)


if __name__ == '__main__':
    main()
