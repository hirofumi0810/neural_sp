#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Make a dataset tsv file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import codecs
from distutils.util import strtobool
import kaldiio
import os
import re
import sentencepiece as spm
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--feat', type=str, default='', nargs='?',
                    help='feats.scp file')
parser.add_argument('--utt2num_frames', type=str, nargs='?',
                    help='utt2num_frames file')
parser.add_argument('--utt2spk', type=str, nargs='?',
                    help='utt2spk file')
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
parser.add_argument('--wp_nbest', type=int, default=1, nargs='?',
                    help='')
parser.add_argument('--update', action='store_true',
                    help='')
args = parser.parse_args()


def main():

    nlsyms = []
    if args.nlsyms:
        with codecs.open(args.nlsyms, 'r', encoding="utf-8") as f:
            for line in f:
                nlsyms.append(line.strip())

    utt2featpath = {}
    if args.feat:
        with codecs.open(args.feat, 'r', encoding="utf-8") as f:
            for line in f:
                utt_id, feat_path = line.strip().split(' ')
                utt2featpath[utt_id] = feat_path

    utt2num_frames = {}
    if args.utt2num_frames and os.path.isfile(args.utt2num_frames):
        with codecs.open(args.utt2num_frames, 'r', encoding="utf-8") as f:
            for line in f:
                utt_id, xlen = line.strip().split(' ')
                utt2num_frames[utt_id] = int(xlen)

    utt2spk = {}
    if args.utt2spk and os.path.isfile(args.utt2spk):
        with codecs.open(args.utt2spk, 'r', encoding="utf-8") as f:
            for line in f:
                utt_id, speaker = line.strip().split(' ')
                utt2spk[str(utt_id)] = speaker

    token2idx = {}
    idx2token = {}
    with codecs.open(args.dict, 'r', encoding="utf-8") as f:
        for line in f:
            token, idx = line.strip().split(' ')
            token2idx[token] = str(idx)
            idx2token[str(idx)] = token

    if args.unit == 'wp':
        sp = spm.SentencePieceProcessor()
        sp.Load(args.wp_model + '.model')

    if not args.update:
        print('utt_id\tspeaker\tfeat_path\txlen\txdim\ttext\ttoken_id\tylen\tydim\tprev_utt')

    xdim = None
    pbar = tqdm(total=len(codecs.open(args.text, 'r', encoding="utf-8").readlines()))

    # Sort by 1.session and 2.onset
    if 'swbd' in args.text and not args.update:
        lines = [line.strip() for line in codecs.open(args.text, 'r', encoding="utf-8")]
        lines = sorted(lines, key=lambda x: (str(utt2spk[x.split(' ')[0]]).split('-')[0],
                                             int(x.split(' ')[0].split('_')[-1].split('-')[0])))
    else:
        lines = codecs.open(args.text, 'r', encoding="utf-8")

    for line in lines:
        # Remove succesive spaces
        line = re.sub(r'[\s]+', ' ', line.strip())
        utt_id = str(line.split(' ')[0])
        words = line.split(' ')[1:]
        if '' in words:
            words.remove('')
        text = ' '.join(words)

        if args.feat:
            feat_path = utt2featpath[utt_id]
            if utt_id in utt2num_frames.keys():
                xlen = utt2num_frames[utt_id]
            else:
                xlen = kaldiio.load_mat(feat_path).shape[-2]
            speaker = utt2spk[utt_id]

            if not os.path.isfile(feat_path.split(':')[0]):
                raise ValueError('There is no file: %s' % feat_path)
        else:
            # dummy for LM
            feat_path = ''
            xlen = 0
            speaker = ''

        # Convert strings into the corresponding indices
        token_ids = []
        if args.unit in ['word', 'word_char']:
            for w in words:
                if w in token2idx.keys():
                    token_ids.append(token2idx[w])
                else:
                    # Replace with <unk>
                    if args.unit == 'word_char':
                        for c in list(w):
                            if c in token2idx.keys():
                                token_ids.append(token2idx[c])
                            else:
                                token_ids.append(token2idx[args.unk])
                    else:
                        token_ids.append(token2idx[args.unk])

        elif args.unit == 'wp':
            # Remove space before the first special symbol
            wps = sp.EncodeAsPieces(text)
            if wps[0] == '▁' and wps[1][0] == '<':
                wps = wps[1:]

            for wp in wps:
                if wp in token2idx.keys():
                    token_ids.append(token2idx[wp])
                else:
                    # Replace with <unk>
                    token_ids.append(token2idx[args.unk])

        elif args.unit == 'char':
            for i, w in enumerate(words):
                if w in nlsyms:
                    token_ids.append(token2idx[w])
                else:
                    for c in list(w):
                        if c in token2idx.keys():
                            token_ids.append(token2idx[c])
                        else:
                            # Replace with <unk>
                            token_ids.append(token2idx[args.unk])

                # Remove whitespaces
                if not args.remove_space:
                    if i < len(words) - 1:
                        token_ids.append(token2idx[args.space])

        elif args.unit == 'phone':
            for p in words:
                token_ids.append(token2idx[p])

        else:
            raise ValueError(args.unit)
        token_id = ' '.join(token_ids)
        ylen = len(token_ids)

        if xdim is None:
            if args.feat:
                xdim = kaldiio.load_mat(feat_path).shape[-1]
            else:
                xdim = 0
        ydim = len(token2idx.keys())

        print('%s\t%s\t%s\t%d\t%d\t%s\t%s\t%d\t%d' %
              (utt_id, speaker, feat_path, xlen, xdim, text, token_id, ylen, ydim))

        # data augmentation for wordpiece
        if args.unit == 'wp' and args.wp_nbest > 1:
            raise NotImplementedError

            for wp_i in sp.NBestEncodeAsPieces(text, args.wp_nbest)[1:]:
                if wp_i in token2idx.keys():
                    token_ids = token2idx[wp_i]
                else:
                    # Replace with <unk>
                    token_ids = token2idx[args.unk]

                token_id = ' '.join(token_ids)
                ylen = len(token_ids)

                print('%s\t%s\t%s\t%d\t%d\t%s\t%s\t%d\t%d' %
                      (utt_id, speaker, feat_path, xlen, xdim, text, token_id, ylen, ydim))

        pbar.update(1)


if __name__ == '__main__':
    main()
