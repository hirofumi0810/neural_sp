#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate texts by the hierarchical ASR model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import argparse
from distutils.util import strtobool

sys.path.append(abspath('../../../'))
from src.models.load_model import load
from src.dataset.loader import Dataset as Dataset_asr
from src.dataset.loader_hierarchical_p2w import Dataset as Dataset_p2w
from src.utils.io.labels.word import Word2char
from src.utils.config import load_config
from src.utils.evaluation.edit_distance import wer_align
from src.utils.evaluation.resolving_unk import resolve_unk
from src.utils.evaluation.normalization import normalize, normalize_swbd, GLM

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str,
                    help='the name of corpus')
parser.add_argument('--data_type', type=str,
                    help='the type of data (ex. train, dev etc.)')
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
parser.add_argument('--rnnlm_weight', type=float, default=0,
                    help='the weight of RNNLM score of the main task')
parser.add_argument('--rnnlm_weight_sub', type=float, default=0,
                    help='the weight of RNNLM score of the sub task')
parser.add_argument('--rnnlm_path', default=None, type=str,  nargs='?',
                    help='path to the RMMLM of the main task')
parser.add_argument('--rnnlm_path_sub', default=None, type=str, nargs='?',
                    help='path to the RMMLM of the sub task')
parser.add_argument('--resolving_unk', type=strtobool, default=False)
parser.add_argument('--a2c_oracle', type=strtobool, default=False)
parser.add_argument('--joint_decoding', type=strtobool, default=False)
parser.add_argument('--score_sub_weight', type=float, default=0)
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
    config = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Load dataset
    if config['input_type'] == 'speech':
        dataset = Dataset_asr(
            corpus=args.corpus,
            data_save_path=args.data_save_path,
            input_freq=config['input_freq'],
            use_delta=config['use_delta'],
            use_double_delta=config['use_double_delta'],
            data_size=config['data_size'] if 'data_size' in config.keys(
            ) else '',
            data_type=args.data_type,
            label_type=config['label_type'],
            label_type_sub=config['label_type_sub'],
            batch_size=args.eval_batch_size,
            sort_utt=False, reverse=False, tool=config['tool'])
    elif config['input_type'] == 'text':
        dataset = Dataset_p2w(
            corpus=args.corpus,
            data_save_path=args.data_save_path,
            data_type=args.data_type,
            data_size=config['data_size'],
            label_type_in=config['label_type_in'],
            label_type=config['label_type'],
            label_type_sub=config['label_type_sub'],
            batch_size=args.eval_batch_size,
            sort_utt=False, reverse=False, tool=config['tool'],
            vocab=config['vocab'],
            use_ctc=config['model_type'] == 'hierarchical_ctc',
            subsampling_factor=2 ** sum(config['subsample_list']),
            use_ctc_sub=config['model_type'] == 'hierarchical_ctc' or (
                config['model_type'] == 'hierarchical_attention' and config['ctc_loss_weight_sub'] > 0),
            subsampling_factor_sub=2 ** sum(config['subsample_list'][:config['encoder_num_layers_sub'] - 1]))
        config['num_classes_input'] = dataset.num_classes_in

    config['num_classes'] = dataset.num_classes
    config['num_classes_sub'] = dataset.num_classes_sub

    if args.corpus == 'swbd':
        dataset.glm_path = join(args.data_save_path, 'eval2000', 'glm')

    # For cold fusion
    if config['rnnlm_fusion_type'] and config['rnnlm_path']:
        # Load a RNNLM config file
        config['rnnlm_config'] = load_config(
            join(args.model_path, 'config_rnnlm.yml'))
        assert config['label_type'] == config['rnnlm_config']['label_type']
        assert args.rnnlm_weight > 0
        config['rnnlm_config']['num_classes'] = dataset.num_classes
    else:
        config['rnnlm_config'] = None

    if config['rnnlm_fusion_type'] and config['rnnlm_path_sub']:
        # Load a RNNLM config file
        config['rnnlm_config_sub'] = load_config(
            join(args.model_path, 'config_rnnlm_sub.yml'))
        assert config['label_type_sub'] == config['rnnlm_config_sub']['label_type']
        assert args.rnnlm_weight_sub > 0
        config['rnnlm_config_sub']['num_classes'] = dataset.num_classes_sub
    else:
        config['rnnlm_config_sub'] = None

    # Load the ASR model
    model = load(model_type=config['model_type'],
                 config=config,
                 backend=config['backend'])

    # Restore the saved parameters
    model.load_checkpoint(save_path=args.model_path, epoch=args.epoch)

    # For shallow fusion
    if not (config['rnnlm_fusion_type'] and config['rnnlm_path']) and args.rnnlm_path is not None and args.rnnlm_weight > 0:
        # Load a RNNLM config file
        config_rnnlm = load_config(
            join(args.rnnlm_path, 'config.yml'), is_eval=True)
        assert config['label_type'] == config_rnnlm['label_type']
        config_rnnlm['num_classes'] = dataset.num_classes

        # Load the pre-trianed RNNLM
        rnnlm = load(model_type=config_rnnlm['model_type'],
                     config=config_rnnlm,
                     backend=config_rnnlm['backend'])
        rnnlm.load_checkpoint(save_path=args.rnnlm_path, epoch=-1)
        rnnlm.flatten_parameters()
        model.rnnlm_0_fwd = rnnlm

    if not (config['rnnlm_fusion_type'] and config['rnnlm_path_sub']) and args.rnnlm_path_sub is not None and args.rnnlm_weight_sub > 0:
        # Load a RNNLM config file
        config_rnnlm_sub = load_config(
            join(args.rnnlm_path_sub, 'config.yml'), is_eval=True)
        assert config['label_type_sub'] == config_rnnlm_sub['label_type']
        config_rnnlm_sub['num_classes'] = dataset.num_classes_sub

        # Load the pre-trianed RNNLM
        rnnlm_sub = load(model_type=config_rnnlm_sub['model_type'],
                         config=config_rnnlm_sub,
                         backend=config_rnnlm_sub['backend'])
        rnnlm_sub.load_checkpoint(save_path=args.rnnlm_path_sub, epoch=-1)
        rnnlm_sub.flatten_parameters()
        model.rnnlm_1_fwd = rnnlm_sub

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    if not args.stdout:
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
                min_decode_len_ratio=MIN_DECODE_LEN_RATIO_WORD,
                beam_width_sub=args.beam_width_sub,
                max_decode_len_sub=MAX_DECODE_LEN_CHAR,
                min_decode_len_sub=MIN_DECODE_LEN_CHAR,
                min_decode_len_ratio_sub=MIN_DECODE_LEN_RATIO_CHAR,
                length_penalty=args.length_penalty,
                coverage_penalty=args.coverage_penalty,
                rnnlm_weight=args.rnnlm_weight,
                rnnlm_weight_sub=args.rnnlm_weight_sub,
                teacher_forcing=args.a2c_oracle,
                ys_sub=ys_sub)
        else:
            best_hyps, aw, perm_idx = model.decode(
                batch['xs'],
                beam_width=args.beam_width,
                max_decode_len=MAX_DECODE_LEN_WORD,
                min_decode_len=MIN_DECODE_LEN_WORD,
                min_decode_len_ratio=MIN_DECODE_LEN_RATIO_WORD,
                length_penalty=args.length_penalty,
                coverage_penalty=args.coverage_penalty,
                rnnlm_weight=args.rnnlm_weight)
            best_hyps_sub, aw_sub, _ = model.decode(
                batch['xs'],
                beam_width=args.beam_width_sub,
                max_decode_len=MAX_DECODE_LEN_CHAR,
                min_decode_len=MIN_DECODE_LEN_CHAR,
                min_decode_len_ratio=MIN_DECODE_LEN_RATIO_CHAR,
                length_penalty=args.length_penalty,
                coverage_penalty=args.coverage_penalty,
                rnnlm_weight=args.rnnlm_weight_sub,
                task_index=1)

        if model.model_type == 'hierarchical_attention' and args.joint_decoding:
            best_hyps_joint, aw_joint, best_hyps_sub_joint, aw_sub_joint, _ = model.decode(
                batch['xs'],
                beam_width=args.beam_width,
                max_decode_len=MAX_DECODE_LEN_WORD,
                min_decode_len=MIN_DECODE_LEN_WORD,
                min_decode_len_ratio=MIN_DECODE_LEN_RATIO_WORD,
                length_penalty=args.length_penalty,
                coverage_penalty=args.coverage_penalty,
                rnnlm_weight=args.rnnlm_weight,
                rnnlm_weight_sub=args.rnnlm_weight_sub,
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
                # NOTE: transcript is seperated by space('_')
            else:
                str_ref = dataset.idx2word(ys[b])

            # Hypothesis
            str_hyp = dataset.idx2word(best_hyps[b])
            str_hyp_sub = dataset.idx2char(best_hyps_sub[b])
            if model.model_type == 'hierarchical_attention' and args.joint_decoding:
                str_hyp_joint = dataset.idx2word(best_hyps_joint[b])
                str_hyp_joint_sub = dataset.idx2char(best_hyps_sub_joint[b])

            # Resolving UNK
            if 'OOV' in str_hyp and args.resolving_unk:
                str_hyp_no_unk = resolve_unk(
                    str_hyp, best_hyps_sub[b], aw[b], aw_sub[b], dataset.idx2char,
                    diff_time_resolution=2 ** sum(model.subsample_list) // 2 ** sum(model.subsample_list[:model.encoder_num_layers_sub - 1]))
            if model.model_type == 'hierarchical_attention' and args.joint_decoding:
                if 'OOV' in str_hyp_joint and args.resolving_unk:
                    str_hyp_no_unk_joint = resolve_unk(
                        str_hyp_joint, best_hyps_sub_joint[b], aw_joint[b], aw_sub_joint[b], dataset.idx2char,
                        diff_time_resolution=2 ** sum(model.subsample_list) // 2 ** sum(model.subsample_list[:model.encoder_num_layers_sub - 1]))

            print('\n----- wav: %s -----' % batch['input_names'][b])

            if dataset.corpus == 'swbd':
                glm = GLM(dataset.glm_path)
                str_ref = normalize_swbd(str_ref, glm)
                str_hyp = normalize_swbd(str_hyp, glm)
                str_hyp_sub = normalize_swbd(str_hyp_sub, glm)
            else:
                str_hyp = normalize(str_hyp, remove_tokens=['>'])
                str_hyp_sub = normalize(str_hyp_sub, remove_tokens=['>'])

            # Resolving UNK
            if 'OOV' in str_hyp and args.resolving_unk:
                if dataset.corpus == 'swbd':
                    str_hyp_no_unk = normalize_swbd(str_hyp_no_unk, glm)
                else:
                    str_hyp_no_unk = normalize(
                        str_hyp_no_unk, remove_tokens=['>'])

            wer = wer_align(ref=str_ref.split('_'),
                            hyp=str_hyp.split('_'),
                            normalize=True,
                            japanese=True if dataset.corpus == 'csj' else False)[0]
            print('\nWER (main)  : %.3f %%' % wer)
            if dataset.corpus != 'csj' or dataset.label_type_sub == 'character_wb':
                wer_sub = wer_align(ref=str_ref.split('_'),
                                    hyp=str_hyp_sub.split('_'),
                                    normalize=True,
                                    japanese=True if dataset.corpus == 'csj' else False)[0]
                print('\nWER (sub)   : %.3f %%' % wer_sub)
            else:
                cer = wer_align(ref=list(str_ref.replace('_', '')),
                                hyp=list(str_hyp_sub.replace('_', '')),
                                normalize=True,
                                japanese=True if dataset.corpus == 'csj' else False)[0]
                print('\nCER (sub)   : %.3f %%' % cer)
            if 'OOV' in str_hyp and args.resolving_unk:
                wer_no_unk = wer_align(
                    ref=str_ref.split('_'),
                    hyp=str_hyp_no_unk.replace('*', '').split('_'),
                    normalize=True,
                    japanese=True if dataset.corpus == 'csj' else False)[0]
                print('\nWER (no UNK): %.3f %%' % wer_no_unk)

            if model.model_type == 'hierarchical_attention' and args.joint_decoding:
                print('===== joint decoding =====')
                if dataset.corpus == 'swbd':
                    str_hyp_joint = normalize_swbd(str_hyp_joint, glm)
                    str_hyp_joint_sub = normalize_swbd(
                        str_hyp_joint_sub, glm)
                else:
                    str_hyp_joint = normalize(
                        str_hyp_joint, remove_tokens=['>'])
                    str_hyp_joint_sub = normalize(
                        str_hyp_joint_sub, remove_tokens=['>'])

                if 'OOV' in str_hyp_joint and args.resolving_unk:
                    if dataset.corpus == 'swbd':
                        str_hyp_no_unk_joint = normalize_swbd(
                            str_hyp_no_unk_joint, glm)
                    else:
                        str_hyp_no_unk_joint = normalize(
                            str_hyp_no_unk_joint, remove_tokens=['>'])

                wer_joint = wer_align(ref=str_ref.split('_'),
                                      hyp=str_hyp_joint.split('_'),
                                      normalize=True,
                                      japanese=True if dataset.corpus == 'csj' else False)[0]
                print('\nWER (main)  : %.3f %%' % wer_joint)
                if 'OOV' in str_hyp_joint and args.resolving_unk:
                    wer_no_unk_joint = wer_align(
                        ref=str_ref.split('_'),
                        hyp=str_hyp_no_unk_joint.replace(
                            '*', '').split('_'),
                        normalize=True,
                        japanese=True if dataset.corpus == 'csj' else False)[0]
                    print('\nWER (no UNK): %.3f %%' % wer_no_unk_joint)

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
