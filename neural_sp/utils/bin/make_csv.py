#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from distutils.util import strtobool
import kaldi_io
import os
import re
import sentencepiece as spm
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--feat', type=str,
                    help='feat.scp file')
parser.add_argument('--utt2num_frames', type=str,
                    help='utt2num_frames file')
parser.add_argument('--dict', type=str,
                    help='dictionary file')
parser.add_argument('--text', type=str,
                    help='text file')
parser.add_argument('--unit', type=str, choices=['word', "wordpiece", 'char', "phone"],
                    help='token units')
parser.add_argument('--remove_word_boundary', type=strtobool, default=False,
                    help='')
parser.add_argument('--is_test', type=strtobool, default=False)
parser.add_argument('--unk', type=str, default='<unk>',
                    help='<unk> token')
parser.add_argument('--space', type=str, default='<space>',
                    help='<space> token')
parser.add_argument('--nlsyms', type=str, default='', nargs='?',
                    help='path to non-linguistic symbols, e.g., <NOISE> etc.')
parser.add_argument('--wp_model', type=str, default=False, nargs='?',
                    help='prefix of the wordpiece model')
args = parser.parse_args()


def main():

    nlsyms = []
    if args.nlsyms:
        with open(args.nlsyms, 'r') as f:
            for line in f:
                nlsyms.append(unicode(line, 'utf-8').strip())

    utt2feat = {}
    with open(args.feat, 'r') as f:
        for line in f:
            utt_id, feat_path = line.strip().split(' ')
            utt2feat[utt_id] = feat_path

    utt2frame = {}
    with open(args.utt2num_frames, 'r') as f:
        for line in f:
            utt_id, x_len = line.strip().split(' ')
            utt2frame[utt_id] = int(x_len)

    token2id = {}
    with open(args.dict, 'r') as f:
        for line in f:
            token, id = unicode(line, 'utf-8').strip().split(' ')
            token2id[token] = str(id)

    if args.unit == 'wordpiece' and not args.is_test:
        sp = spm.SentencePieceProcessor()
        sp.Load(args.wp_model + '.model')

    print(',utt_id,feat_path,x_len,x_dim,text,token_id,y_len,y_dim')

    x_dim = None
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

            # for CSJ
            words = list(map(lambda x: x.split('+')[0], words))

            text = ' '.join(words)
            feat_path = utt2feat[utt_id]
            x_len = utt2frame[utt_id]

            if not os.path.isfile(feat_path.split(':')[0]):
                raise ValueError('There is no file: %s' % feat_path)

            # Convert strings into the corresponding indices
            if args.is_test:
                token_id = ''
                y_len = 1
                # NOTE; skip test sets for OOV issues
            else:
                token_ids = []
                if args.unit == 'word':
                    for w in words:
                        if w in token2id.keys():
                            token_ids.append(token2id[w])
                        else:
                            # Replace with <unk>
                            token_ids.append(token2id[args.unk])

                elif args.unit == 'wordpiece':
                    token_ids = list(map(str, sp.EncodeAsIds(text)))

                elif args.unit == 'char':
                    for i,  w in enumerate(words):
                        if w in nlsyms:
                            token_ids.append(token2id[w])
                        else:
                            for c in list(w):
                                if c in token2id.keys():
                                    token_ids.append(c)
                                else:
                                    # Replace with <unk>
                                    token_ids.append(token2id[args.unk])

                        # Remove whitespaces
                        if not args.remove_word_boundary:
                            if i < len(words) - 1:
                                token_ids.append(token2id[args.space])

                elif args.unit == 'phone':
                    for p in words:
                        token_ids.append(token2id[p])

                else:
                    raise ValueError(args.unit)
                token_id = ' '.join(token_ids)
                y_len = len(token_ids)

            if x_dim is None:
                x_dim = kaldi_io.read_mat(feat_path).shape[-1]
            y_dim = len(token2id.keys())

            print("%d,%s,%s,%d,%d,\"%s\",%s,%d,%d" %
                  (utt_count, utt_id.encode('utf-8'), feat_path, x_len, x_dim,
                   text.encode('utf-8'), token_id.encode('utf-8'), y_len, y_dim))
            utt_count += 1
            pbar.update(1)


if __name__ == '__main__':
    main()
