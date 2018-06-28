#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate texts by the ASR model on the modular training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import argparse
from distutils.util import strtobool

sys.path.append(abspath('../../../'))
from src.models.load_model import load
from src.dataset.loader import Dataset as Dataset_a2p
from src.dataset.loader_p2w import Dataset as Dataset_p2w
from src.utils.config import load_config
from src.utils.evaluation.edit_distance import wer_align
from src.utils.evaluation.normalization import normalize, normalize_swbd, GLM

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str,
                    help='the name of corpus')
parser.add_argument('--data_type', type=str,
                    help='the type of data (ex. train, dev etc.)')
parser.add_argument('--data_save_path', type=str,
                    help='path to saved data')
parser.add_argument('--model_path_a2p', type=str,
                    help='path to the model to evaluate (A2P)')
parser.add_argument('--model_path_p2w', type=str,
                    help='path to the model to evaluate (P2W)')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')
parser.add_argument('--beam_width_a2p', type=int, default=1,
                    help='the size of beam (A2P)')
parser.add_argument('--beam_width_p2w', type=int, default=1,
                    help='the size of beam (P2W)')
parser.add_argument('--length_penalty', type=float, default=0,
                    help='length penalty')
parser.add_argument('--coverage_penalty', type=float, default=0,
                    help='coverage penalty')
parser.add_argument('--rnnlm_weight', type=float, default=0,
                    help='the weight of RNNLM score')
parser.add_argument('--rnnlm_path', default=None, type=str, nargs='?',
                    help='path to the RMMLM')
parser.add_argument('--stdout', type=strtobool, default=False)
args = parser.parse_args()

# corpus depending
if args.corpus == 'csj':
    MAX_DECODE_LEN_WORD = 100
    MIN_DECODE_LEN_WORD = 1
    MAX_DECODE_LEN_RATIO_WORD = 1
    MIN_DECODE_LEN_RATIO_WORD = 0

    MAX_DECODE_LEN_CHAR = 200
    MIN_DECODE_LEN_CHAR = 1
    MAX_DECODE_LEN_RATIO_CHAR = 1
    MIN_DECODE_LEN_RATIO_CHAR = 0.2

    MAX_DECODE_LEN_PHONE = 200
    MIN_DECODE_LEN_PHONE = 1
    MAX_DECODE_LEN_RATIO_PHONE = 1
    MIN_DECODE_LEN_RATIO_PHONE = 0
elif args.corpus == 'swbd':
    MAX_DECODE_LEN_WORD = 100
    MIN_DECODE_LEN_WORD = 1
    MAX_DECODE_LEN_RATIO_WORD = 1
    MIN_DECODE_LEN_RATIO_WORD = 0

    MAX_DECODE_LEN_CHAR = 300
    MIN_DECODE_LEN_CHAR = 1
    MAX_DECODE_LEN_RATIO_CHAR = 1
    MIN_DECODE_LEN_RATIO_CHAR = 0.2

    MAX_DECODE_LEN_PHONE = 300
    MIN_DECODE_LEN_PHONE = 1
    MAX_DECODE_LEN_RATIO_PHONE = 1
    MIN_DECODE_LEN_RATIO_PHONE = 0
elif args.corpus == 'librispeech':
    MAX_DECODE_LEN_WORD = 200
    MIN_DECODE_LEN_WORD = 1
    MAX_DECODE_LEN_RATIO_WORD = 1
    MIN_DECODE_LEN_RATIO_WORD = 0

    MAX_DECODE_LEN_CHAR = 600
    MIN_DECODE_LEN_CHAR = 1
    MAX_DECODE_LEN_RATIO_CHAR = 1
    MIN_DECODE_LEN_RATIO_CHAR = 0.2
elif args.corpus == 'wsj':
    MAX_DECODE_LEN_WORD = 32
    MIN_DECODE_LEN_WORD = 2
    MAX_DECODE_LEN_RATIO_WORD = 1
    MIN_DECODE_LEN_RATIO_WORD = 0

    MAX_DECODE_LEN_CHAR = 199
    MIN_DECODE_LEN_CHAR = 10
    MAX_DECODE_LEN_RATIO_CHAR = 1
    MIN_DECODE_LEN_RATIO_CHAR = 0.2
    # NOTE:
    # dev93 (char): 10-199
    # test_eval92 (char): 16-195
    # dev93 (word): 2-32
    # test_eval92 (word): 3-30
elif args.corpus == 'timit':
    MAX_DECODE_LEN_PHONE = 71
    MIN_DECODE_LEN_PHONE = 13
    MAX_DECODE_LEN_RATIO_PHONE = 1
    MIN_DECODE_LEN_RATIO_PHONE = 0
    # NOTE*
    # dev: 13-71
    # test: 13-69
else:
    raise ValueError(args.corpus)


def main():

    # Load a ASR config file
    config_a2p = load_config(
        join(args.model_path_a2p, 'config.yml'), is_eval=True)
    config_p2w = load_config(
        join(args.model_path_p2w, 'config.yml'), is_eval=True)

    # Load dataset
    dataset_a2p = Dataset_a2p(
        corpus=args.corpus,
        data_save_path=args.data_save_path,
        input_freq=config_a2p['input_freq'],
        use_delta=config_a2p['use_delta'],
        use_double_delta=config_a2p['use_double_delta'],
        data_size=config_a2p['data_size'] if 'data_size' in config_a2p.keys(
        ) else '',
        data_type=args.data_type,
        label_type=config_a2p['label_type'],
        batch_size=args.eval_batch_size,
        sort_utt=False, reverse=False, tool=config_a2p['tool'])
    config_a2p['num_classes'] = dataset_a2p.num_classes
    dataset_p2w = Dataset_p2w(
        corpus=args.corpus,
        data_save_path=args.data_save_path,
        data_type=args.data_type,
        data_size=config_p2w['data_size'],
        label_type_in=config_p2w['label_type_in'],
        label_type=config_p2w['label_type'],
        batch_size=args.eval_batch_size,
        sort_utt=False, reverse=False, tool=config_p2w['tool'],
        vocab=config_p2w['vocab'],
        use_ctc=config['model_type'] == 'ctc' or (
            config['model_type'] == 'attention' and config['ctc_loss_weight'] > 0),
        subsampling_factor=2 ** sum(config['subsample_list']))
    config_p2w['num_classes_input'] = dataset_p2w.num_classes_in
    config_p2w['num_classes'] = dataset_p2w.num_classes
    config_p2w['num_classes_sub'] = dataset_p2w.num_classes
    assert config_a2p['num_classes'] == config_p2w['num_classes_input']

    if args.corpus == 'swbd':
        dataset_p2w.glm_path = join(args.data_save_path, 'eval2000', 'glm')

    # Load the ASR model
    model_a2p = load(model_type=config_a2p['model_type'],
                     config=config_a2p,
                     backend=config_a2p['backend'])
    model_p2w = load(model_type=config_p2w['model_type'],
                     config=config_p2w,
                     backend=config_p2w['backend'])

    # Restore the saved parameters
    model_a2p.load_checkpoint(save_path=args.model_path_a2p, epoch=args.epoch)
    model_p2w.load_checkpoint(save_path=args.model_path_p2w, epoch=args.epoch)

    # For shallow fusion
    if args.rnnlm_path is not None and args.rnnlm_weight > 0:
        # Load a RNNLM config file
        config_rnnlm = load_config(
            join(args.rnnlm_path, 'config.yml'), is_eval=True)
        assert config_p2w['label_type'] == config_rnnlm['label_type']
        config_rnnlm['num_classes'] = dataset_p2w.num_classes

        # Load the pre-trianed RNNLM
        rnnlm = load(model_type=config_rnnlm['model_type'],
                     config=config_rnnlm,
                     backend=config_rnnlm['backend'])
        rnnlm.load_checkpoint(save_path=args.rnnlm_path, epoch=-1)
        rnnlm.flatten_parameters()
        if config_rnnlm['backward']:
            model_p2w.rnnlm_0_bwd = rnnlm
        else:
            model_p2w.rnnlm_0_fwd = rnnlm

    # GPU setting
    model_a2p.set_cuda(deterministic=False, benchmark=True)
    model_p2w.set_cuda(deterministic=False, benchmark=True)

    if not args.stdout:
        sys.stdout = open(join(args.model_path_p2w, 'decode.txt'), 'w')

    if 'phone' in dataset_a2p.label_type:
        map_fn_a2p = dataset_a2p.idx2phone
        max_decode_len_a2p = MAX_DECODE_LEN_PHONE
        min_decode_len_a2p = MIN_DECODE_LEN_PHONE
        min_decode_len_ratio_a2p = MIN_DECODE_LEN_RATIO_PHONE
    elif 'character' in dataset_a2p.label_type:
        map_fn_a2p = dataset_a2p.idx2char
        max_decode_len_a2p = MAX_DECODE_LEN_CHAR
        min_decode_len_a2p = MIN_DECODE_LEN_CHAR
        min_decode_len_ratio_a2p = MIN_DECODE_LEN_RATIO_CHAR
    else:
        raise ValueError(dataset_a2p.label_type)

    if dataset_p2w.label_type == 'word':
        map_fn_p2w = dataset_p2w.idx2word
        max_decode_len_p2w = MAX_DECODE_LEN_WORD
        min_decode_len_p2w = MIN_DECODE_LEN_WORD
        min_decode_len_ratio_p2w = MIN_DECODE_LEN_RATIO_WORD
    elif 'character' in dataset_p2w.label_type:
        map_fn_p2w = dataset_p2w.idx2char
        max_decode_len_p2w = MAX_DECODE_LEN_CHAR
        min_decode_len_p2w = MIN_DECODE_LEN_CHAR
        min_decode_len_ratio_p2w = MIN_DECODE_LEN_RATIO_CHAR

    for (batch_a2p, is_new_epoch), (batch_p2w, _) in zip(dataset_a2p, dataset_p2w):
        # Decode (A2P)
        best_hyps_a2p, _, perm_idx = model_a2p.decode(
            batch_a2p['xs'],
            beam_width=args.beam_width_a2p,
            max_decode_len=max_decode_len_a2p,
            min_decode_len=min_decode_len_a2p,
            length_penalty=args.length_penalty,
            min_decode_len_ratio=min_decode_len_ratio_a2p,
            coverage_penalty=args.coverage_penalty,
            rnnlm_weight=args.rnnlm_weight)

        if len(best_hyps_a2p[0]) < 1:
            continue

        # Decode (P2W)
        best_hyps_p2w, _, perm_idx = model_p2w.decode(
            best_hyps_a2p,
            beam_width=args.beam_width_p2w,
            max_decode_len=max_decode_len_p2w,
            min_decode_len=min_decode_len_p2w,
            length_penalty=args.length_penalty,
            min_decode_len_ratio=min_decode_len_ratio_p2w,
            coverage_penalty=args.coverage_penalty,
            rnnlm_weight=args.rnnlm_weight)

        ys = [batch_p2w['ys'][i] for i in perm_idx]

        for b in range(len(batch_p2w['xs'])):
            # Reference
            if dataset_p2w.is_test:
                str_ref = ys[b]
                # NOTE: transcript is seperated by space('_')
            else:
                str_ref = map_fn_p2w(ys[b])

            # Hypothesis
            str_hyp = map_fn_p2w(best_hyps_p2w[b])

            print('\n----- wav: %s -----' % batch_p2w['input_names'][b])

            if dataset_p2w.corpus == 'swbd':
                glm = GLM(dataset_p2w.glm_path)
                str_ref = normalize_swbd(str_ref, glm)
                str_hyp = normalize_swbd(str_hyp, glm)
            else:
                str_hyp = normalize(str_hyp)

            # A2P
            # if dataset_a2p.label_type == 'character':
            #     cer = wer_align(ref=list(str_ref.replace('_', '')),
            #                     hyp=list(str_hyp.replace('_', '')),
            #                     normalize=True,
            #                     japanese=True if args.corpus == 'csj' else False)[0]
            #     print('\nCER: %.3f %%' % cer)
            # elif 'phone' in dataset_a2p.label_type:
            #     per = wer_align(ref=str_ref.split('_'),
            #                     hyp=str_hyp.split('_'),
            #                     normalize=True)[0]
            #     print('\nPER: %.3f %%' % per)
            # else:
            #     raise ValueError(dataset_a2p.label_type)

            # P2W
            if dataset_p2w.label_type in ['word', 'character_wb'] or (args.corpus != 'csj' and dataset_p2w.label_type == 'character'):
                wer = wer_align(ref=str_ref.split('_'),
                                hyp=str_hyp.split('_'),
                                normalize=True,
                                japanese=True if args.corpus == 'csj' else False)[0]
                print('\nWER: %.3f %%' % wer)
            elif dataset_p2w.label_type == 'character':
                cer = wer_align(ref=list(str_ref.replace('_', '')),
                                hyp=list(str_hyp.replace('_', '')),
                                normalize=True,
                                japanese=True if args.corpus == 'csj' else False)[0]
                print('\nCER: %.3f %%' % cer)
            else:
                raise ValueError(dataset_p2w.label_type)

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
