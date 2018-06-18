#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate texts by the hierarchical ASR model (WSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import argparse
from distutils.util import strtobool

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.wsj.s5.exp.dataset.load_dataset_hierarchical import Dataset
from utils.io.labels.word import Word2char
from utils.config import load_config
from utils.evaluation.edit_distance import wer_align
from utils.evaluation.resolving_unk import resolve_unk
from utils.evaluation.normalization import normalize

parser = argparse.ArgumentParser()
parser.add_argument('--data_save_path', type=str,
                    help='path to saved data')
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')
parser.add_argument('--beam_width', type=int, default=1,
                    help='the size of beam in the main task')
parser.add_argument('--beam_width_sub', type=int, default=1,
                    help='the size of beam in the sub task')
parser.add_argument('--length_penalty', type=float, default=0,
                    help='length penalty')
parser.add_argument('--coverage_penalty', type=float, default=0,
                    help='coverage penalty')
parser.add_argument('--resolving_unk', type=strtobool, default=False)
parser.add_argument('--a2c_oracle', type=strtobool, default=False)
parser.add_argument('--joint_decoding', type=strtobool, default=False)
parser.add_argument('--score_sub_weight', type=float, default=0)

MAX_DECODE_LEN_WORD = 32
MIN_DECODE_LEN_WORD = 2
MAX_DECODE_LEN_CHAR = 199
MIN_DECODE_LEN_CHAR = 10


def main():

    args = parser.parse_args()

    # Load a config file
    config = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Load dataset
    dataset = Dataset(data_save_path=args.data_save_path,
                      input_freq=config['input_freq'],
                      use_delta=config['use_delta'],
                      use_double_delta=config['use_double_delta'],
                      # data_type='test_dev93',
                      data_type='test_eval92',
                      data_size=config['data_size'],
                      label_type=config['label_type'],
                      label_type_sub=config['label_type_sub'],
                      batch_size=args.eval_batch_size,
                      sort_utt=False, reverse=False, tool=config['tool'])
    config['num_classes'] = dataset.num_classes
    config['num_classes_sub'] = dataset.num_classes_sub

    # Load model
    model = load(model_type=config['model_type'],
                 config=config,
                 backend=config['backend'])

    # Restore the saved parameters
    model.load_checkpoint(save_path=args.model_path, epoch=args.epoch)

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    sys.stdout = open(join(args.model_path, 'decode.txt'), 'w')

    word2char = Word2char(dataset.vocab_file_path,
                          dataset.vocab_file_path_sub)

    for batch, is_new_epoch in dataset:
        # Decode
        if model.model_type == 'nested_attention':
            if args.a2c_oracle:
                if dataset.is_test:
                    max_label_num = 0
                    for b in range(len(batch['xs'])):
                        if max_label_num < len(list(batch['ys_sub'][b])):
                            max_label_num = len(list(batch['ys_sub'][b]))

                    ys_sub = []
                    for b in range(len(batch['xs'])):
                        indices = dataset.char2idx(batch['ys_sub'][b])
                        ys_sub += [indices]
                        # NOTE: transcript is seperated by space('_')
            else:
                ys_sub = batch['ys_sub']

            best_hyps, aw, best_hyps_sub, aw_sub, _, perm_idx = model.decode(
                batch['xs'],
                beam_width=args.beam_width,
                max_decode_len=MAX_DECODE_LEN_WORD,
                min_decode_len=MIN_DECODE_LEN_WORD,
                beam_width_sub=args.beam_width_sub,
                max_decode_len_sub=MAX_DECODE_LEN_CHAR,
                min_decode_len_sub=MIN_DECODE_LEN_CHAR,
                length_penalty=args.length_penalty,
                coverage_penalty=args.coverage_penalty,
                teacher_forcing=args.a2c_oracle,
                ys_sub=ys_sub)
        else:
            best_hyps, aw, perm_idx = model.decode(
                batch['xs'],
                beam_width=args.beam_width,
                max_decode_len=MAX_DECODE_LEN_WORD,
                min_decode_len=MIN_DECODE_LEN_WORD,
                length_penalty=args.length_penalty,
                coverage_penalty=args.coverage_penalty)
            best_hyps_sub, aw_sub, _ = model.decode(
                batch['xs'],
                beam_width=args.beam_width_sub,
                max_decode_len=MAX_DECODE_LEN_CHAR,
                min_decode_len=MIN_DECODE_LEN_CHAR,
                length_penalty=args.length_penalty,
                coverage_penalty=args.coverage_penalty,
                task_index=1)

        if model.model_type == 'hierarchical_attention' and args.joint_decoding:
            best_hyps_joint, aw_joint, best_hyps_sub_joint, aw_sub_joint, _ = model.decode(
                batch['xs'],
                beam_width=args.beam_width,
                max_decode_len=MAX_DECODE_LEN_WORD,
                min_decode_len=MIN_DECODE_LEN_WORD,
                length_penalty=args.length_penalty,
                coverage_penalty=args.coverage_penalty,
                joint_decoding=args.joint_decoding,
                space_index=dataset.char2idx('_')[0],
                oov_index=dataset.word2idx('OOV')[0],
                word2char=word2char,
                idx2word=dataset.idx2word,
                idx2char=dataset.idx2char,
                score_sub_weight=args.score_sub_weight)

        ys = [batch['ys'][i] for i in perm_idx]
        ys_sub = [batch['ys_sub'][i] for i in perm_idx]

        for b in range(len(batch['xs'])):
            # Reference
            if dataset.is_test:
                str_ref = ys[b]
                str_ref_sub = ys_sub[b]
                # NOTE: transcript is seperated by space('_')
            else:
                str_ref = dataset.idx2word(ys[b])
                str_ref_sub = dataset.idx2char(ys_sub[b])

            # Hypothesis
            str_hyp = dataset.idx2word(best_hyps[b])
            str_hyp_sub = dataset.idx2char(best_hyps_sub[b])
            if model.model_type == 'hierarchical_attention' and args.joint_decoding:
                str_hyp_joint = dataset.idx2word(best_hyps_joint[b])
                str_hyp_joint_sub = dataset.idx2char(best_hyps_sub_joint[b])

            # Resolving UNK
            if 'OOV' in str_hyp and args.resolving_unk:
                str_hyp_no_unk = resolve_unk(
                    str_hyp, best_hyps_sub[b], aw[b], aw_sub[b], dataset.idx2char)
            if model.model_type == 'hierarchical_attention' and args.joint_decoding:
                if 'OOV' in str_hyp_joint and args.resolving_unk:
                    str_hyp_no_unk_joint = resolve_unk(
                        str_hyp_joint, best_hyps_sub_joint[b], aw_joint[b], aw_sub_joint[b], dataset.idx2char)

            print('----- wav: %s -----' % batch['input_names'][b])

            str_hyp = normalize(str_hyp)
            str_hyp_sub = normalize(str_hyp_sub)

            if 'OOV' in str_hyp and args.resolving_unk:
                str_hyp_no_unk = normalize(str_hyp_no_unk)

            wer = wer_align(ref=str_ref.split('_'),
                            hyp=str_hyp.split('_'),
                            normalize=True)[0]
            print('\nWER (main)  : %.3f %%' % wer)
            if dataset.label_type_sub == 'character_wb':
                wer_sub = wer_align(ref=str_ref_sub.split('_'),
                                    hyp=str_hyp_sub.split('_'),
                                    normalize=True,
                                    japanese=True)[0]
                print('\nWER (sub)   : %.3f %%' % wer_sub)
            else:
                cer = wer_align(ref=list(str_ref.replace('_', '')),
                                hyp=list(str_hyp_sub.replace('_', '')),
                                normalize=True)[0]
                print('\nCER (sub)   : %.3f %%' % cer)
            if 'OOV' in str_hyp and args.resolving_unk:
                wer_no_unk = wer_align(
                    ref=str_ref.split('_'),
                    hyp=str_hyp_no_unk.replace('*', '').split('_'),
                    normalize=True)[0]
                print('\nWER (no UNK): %.3f %%' % wer_no_unk)

            if model.model_type == 'hierarchical_attention' and args.joint_decoding:
                print('===== joint decoding =====')
                str_hyp_joint = normalize(str_hyp_joint)
                str_hyp_joint_sub = normalize(str_hyp_joint_sub)

                if 'OOV' in str_hyp_joint and args.resolving_unk:
                    str_hyp_no_unk_joint = normalize(str_hyp_no_unk_joint)

                wer_joint = wer_align(ref=str_ref.split('_'),
                                      hyp=str_hyp_joint.split('_'),
                                      normalize=True)[0]
                print('\nWER (main)  : %.3f %%' % wer_joint)
                if 'OOV' in str_hyp_joint and args.resolving_unk:
                    wer_no_unk_joint = wer_align(
                        ref=str_ref.split('_'),
                        hyp=str_hyp_no_unk_joint.replace('*', '').split('_'),
                        normalize=True)[0]
                    print('\nWER (no UNK): %.3f %%' % wer_no_unk_joint)

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
