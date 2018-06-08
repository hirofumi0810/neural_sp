#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Decode the hierarchical model's outputs (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import argparse
import re

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.csj.s5.exp.dataset.load_dataset_hierarchical import Dataset
from utils.io.labels.word import Word2char
from utils.config import load_config
from utils.evaluation.edit_distance import compute_wer
from utils.evaluation.resolving_unk import resolve_unk

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
                    help='length penalty in beam search decoding')
parser.add_argument('--coverage_penalty', type=float, default=0,
                    help='coverage penalty in beam search decoding')

parser.add_argument('--resolving_unk', type=bool, default=False)
parser.add_argument('--a2c_oracle', type=bool, default=False)
parser.add_argument('--joint_decoding', choices=[None, 'onepass', 'rescoring'],
                    default=None)
parser.add_argument('--score_sub_weight', type=float, default=0)

MAX_DECODE_LEN_WORD = 100
MIN_DECODE_LEN_WORD = 1
MAX_DECODE_LEN_CHAR = 200
MIN_DECODE_LEN_CHAR = 1


def main():

    args = parser.parse_args()

    # Load a config file (.yml)
    params = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Load dataset
    dataset = Dataset(
        data_save_path=args.data_save_path,
        backend=params['backend'],
        input_freq=params['input_freq'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        data_type='eval1',
        # data_type='eval2',
        # data_type='eval3',
        data_size=params['data_size'],
        label_type=params['label_type'], label_type_sub=params['label_type_sub'],
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False, reverse=False, tool=params['tool'])
    params['num_classes'] = dataset.num_classes
    params['num_classes_sub'] = dataset.num_classes_sub

    # Load model
    model = load(model_type=params['model_type'],
                 params=params,
                 backend=params['backend'])

    # Restore the saved parameters
    model.load_checkpoint(save_path=args.model_path, epoch=args.epoch)

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    # sys.stdout = open(join(model.model_dir, 'decode.txt'), 'w')

    ######################################################################

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
                beam_width_sub=args.beam_width_sub,
                max_decode_len=MAX_DECODE_LEN_WORD,
                max_decode_len_sub=MAX_DECODE_LEN_CHAR,
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

        if model.model_type == 'hierarchical_attention' and args.joint_decoding is not None:
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
            ##############################
            # Reference
            ##############################
            if dataset.is_test:
                str_ref = ys[b]
                str_ref_sub = ys_sub[b]
                # NOTE: transcript is seperated by space('_')
            else:
                # Convert from list of index to string
                str_ref = dataset.idx2word(ys[b])
                str_ref_sub = dataset.idx2char(ys_sub[b])

            ##############################
            # Hypothesis
            ##############################
            # Convert from list of index to string
            str_hyp = dataset.idx2word(best_hyps[b])
            str_hyp_sub = dataset.idx2char(best_hyps_sub[b])
            if model.model_type == 'hierarchical_attention' and args.joint_decoding is not None:
                str_hyp_joint = dataset.idx2word(best_hyps_joint[b])
                str_hyp_sub_joint = dataset.idx2char(best_hyps_sub_joint[b])

            ##############################
            # Resolving UNK
            ##############################
            if 'OOV' in str_hyp and args.resolving_unk:
                str_hyp_no_unk = resolve_unk(
                    str_hyp, best_hyps_sub[b], aw[b], aw_sub[b], dataset.idx2char)
            if model.model_type == 'hierarchical_attention' and args.joint_decoding is not None:
                if 'OOV' in str_hyp_joint and args.resolving_unk:
                    str_hyp_no_unk_joint = resolve_unk(
                        str_hyp_joint, best_hyps_sub_joint[b], aw_joint[b], aw_sub_joint[b], dataset.idx2char)

            print('----- wav: %s -----' % batch['input_names'][b])
            print('Ref         : %s' % str_ref.replace('_', ' '))
            print('Hyp (main)  : %s' % str_hyp.replace('_', ' '))
            print('Hyp (sub)   : %s' % str_hyp_sub.replace('_', ' '))
            if 'OOV' in str_hyp and args.resolving_unk:
                print('Hyp (no UNK): %s' % str_hyp_no_unk.replace('_', ' '))
            if model.model_type == 'hierarchical_attention' and args.joint_decoding is not None:
                print('===== joint decoding =====')
                print('Hyp (main)  : %s' % str_hyp_joint.replace('_', ' '))
                print('Hyp (sub)   : %s' % str_hyp_sub_joint.replace('_', ' '))
                if 'OOV' in str_hyp_joint and args.resolving_unk:
                    print('Hyp (no UNK): %s' %
                          str_hyp_no_unk_joint.replace('_', ' '))

            try:
                wer, _, _, _ = compute_wer(
                    ref=str_ref.split('_'),
                    hyp=re.sub(r'(.*)_>(.*)', r'\1', str_hyp).split('_'),
                    normalize=True)
                print('WER (main)  : %.3f %%' % (wer * 100))
                if dataset.label_type_sub == 'character_wb':
                    wer_sub, _, _, _ = compute_wer(
                        ref=str_ref_sub.split('_'),
                        hyp=re.sub(r'(.*)>(.*)', r'\1',
                                   str_hyp_sub).split('_'),
                        normalize=True)
                    print('WER (sub)   : %.3f %%' % (wer_sub * 100))
                else:
                    cer, _, _, _ = compute_wer(
                        ref=list(str_ref_sub.replace('_', '')),
                        hyp=list(re.sub(r'(.*)>(.*)', r'\1',
                                        str_hyp_sub).replace('_', '')),
                        normalize=True)
                    print('CER (sub)   : %.3f %%' % (cer * 100))
                if 'OOV' in str_hyp and args.resolving_unk:
                    wer_no_unk, _, _, _ = compute_wer(
                        ref=str_ref.split('_'),
                        hyp=re.sub(r'(.*)_>(.*)', r'\1',
                                   str_hyp_no_unk.replace('*', '')).split('_'),
                        normalize=True)
                    print('WER (no UNK): %.3f %%' % (wer_no_unk * 100))

                if model.model_type == 'hierarchical_attention' and args.joint_decoding is not None:
                    print('===== joint decoding =====')
                    wer_joint, _, _, _ = compute_wer(
                        ref=str_ref.split('_'),
                        hyp=re.sub(r'(.*)_>(.*)', r'\1',
                                   str_hyp_joint).split('_'),
                        normalize=True)
                    print('WER (main)  : %.3f %%' % (wer_joint * 100))
                    if 'OOV' in str_hyp_joint and args.resolving_unk:
                        wer_no_unk_joint, _, _, _ = compute_wer(
                            ref=str_ref.split('_'),
                            hyp=re.sub(r'(.*)_>(.*)', r'\1',
                                       str_hyp_no_unk_joint.replace('*', '')).split('_'),
                            normalize=True)
                        print('WER (no UNK): %.3f %%' %
                              (wer_no_unk_joint * 100))

            except:
                print('--- skipped ---')
            print('\n')

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
