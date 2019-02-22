#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Make a dataset csv file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import codecs
from distutils.util import strtobool
import kaldi_io
import os
import re
import sentencepiece as spm
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--feat', type=str, default='', nargs='?',
                    help='feat.scp file')
parser.add_argument('--utt2num_frames', type=str,
                    help='utt2num_frames file')
parser.add_argument('--dict', type=str,
                    help='dictionary file')
parser.add_argument('--text', type=str,
                    help='text file')
parser.add_argument('--unit', type=str, choices=['word', "wp", 'char', "phone", "word_char"],
                    help='token units')
parser.add_argument('--remove_space', type=strtobool, default=False,
                    help='')
parser.add_argument('--unk', type=str, default='<unk>',
                    help='<unk> token')
parser.add_argument('--space', type=str, default='<space>',
                    help='<space> token')
parser.add_argument('--nlsyms', type=str, default='', nargs='?',
                    help='path to non-linguistic symbols, e.g., [noise] etc.')
parser.add_argument('--wp_model', type=str, default=False, nargs='?',
                    help='prefix of the wordpiece model')
args = parser.parse_args()


def main():

    nlsyms = []
    if args.nlsyms:
        with codecs.open(args.nlsyms, 'r', encoding="utf-8") as f:
            for line in f:
                nlsyms.append(line.strip())

    utt2feat = {}
    utt2frame = {}
    if args.feat:
        with codecs.open(args.feat, 'r', encoding="utf-8") as f:
            for line in f:
                utt_id, feat_path = line.strip().split(' ')
                utt2feat[utt_id] = feat_path

        with codecs.open(args.utt2num_frames, 'r', encoding="utf-8") as f:
            for line in f:
                utt_id, xlen = line.strip().split(' ')
                utt2frame[utt_id] = int(xlen)

    token2id = {}
    with codecs.open(args.dict, 'r', encoding="utf-8") as f:
        for line in f:
            token, id = line.strip().split(' ')
            token2id[token] = str(id)

    id2token = {}
    with codecs.open(args.dict, 'r', encoding="utf-8") as f:
        for line in f:
            token, id = line.strip().split(' ')
            id2token[str(id)] = token

    if args.unit == 'wp':
        sp = spm.SentencePieceProcessor()
        sp.Load(args.wp_model + '.model')

    print(',utt_id,feat_path,xlen,x_dim,text,token_id,ylen,ydim')

    x_dim = None
    utt_count = 0
    with codecs.open(args.text, 'r', encoding="utf-8") as f:
        pbar = tqdm(total=len(codecs.open(args.text, 'r', encoding="utf-8").readlines()))
        for line in f:
            # Remove succesive spaces
            line = re.sub(r'[\s]+', ' ', line.strip())
            utt_id = line.split(' ')[0]
            words = line.split(' ')[1:]
            if '' in words:
                words.remove('')

            text = ' '.join(words)
            if args.feat:
                feat_path = utt2feat[utt_id]
                xlen = utt2frame[utt_id]

                if not os.path.isfile(feat_path.split(':')[0]):
                    raise ValueError('There is no file: %s' % feat_path)
            else:
                # dummy for LM
                feat_path = ''
                xlen = 0

            # Convert strings into the corresponding indices
            token_ids = []
            if args.unit in ['word', 'word_char']:
                for w in words:
                    if w in token2id.keys():
                        token_ids.append(token2id[w])
                    else:
                        # Replace with <unk>
                        if args.unit == 'word_char':
                            for c in list(w):
                                if c in token2id.keys():
                                    token_ids.append(token2id[c])
                                else:
                                    token_ids.append(token2id[args.unk])
                        else:
                            token_ids.append(token2id[args.unk])

            elif args.unit == 'wp':
                wps = sp.EncodeAsPieces(text)
                for wp in wps:
                    if wp in token2id.keys():
                        token_ids.append(token2id[wp])
                    else:
                        # Replace with <unk>
                        token_ids.append(token2id[args.unk])
            elif args.unit == 'char':
                for i,  w in enumerate(words):
                    if w in nlsyms:
                        token_ids.append(token2id[w])
                    else:
                        for c in list(w):
                            if c in token2id.keys():
                                token_ids.append(token2id[c])
                            else:
                                # Replace with <unk>
                                token_ids.append(token2id[args.unk])

                    # Remove whitespaces
                    if not args.remove_space:
                        if i < len(words) - 1:
                            token_ids.append(token2id[args.space])

            elif args.unit == 'phone':
                for p in words:
                    token_ids.append(token2id[p])

            else:
                raise ValueError(args.unit)
            token_id = ' '.join(token_ids)
            ylen = len(token_ids)

            if x_dim is None:
                if args.feat:
                    x_dim = kaldi_io.read_mat(feat_path).shape[-1]
                else:
                    x_dim = 0
            ydim = len(token2id.keys())

            print('\"%d\",\"%s\",\"%s\",\"%d\",\"%d\",\"%s\",\"%s\",\"%d\",\"%d\"' %
                  (utt_count, utt_id, feat_path, xlen, x_dim, text, token_id, ylen, ydim))
            utt_count += 1
            pbar.update(1)


if __name__ == '__main__':
    main()
