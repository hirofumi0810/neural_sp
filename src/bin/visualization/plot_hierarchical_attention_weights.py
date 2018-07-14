#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot attention weights of the hierarchical attention model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath, isdir
import sys
import argparse
import shutil
from distutils.util import strtobool

sys.path.append(abspath('../../../'))
from src.models.load_model import load
from src.dataset.loader_hierarchical import Dataset as Dataset_asr
from src.dataset.loader_hierarchical_p2w import Dataset as Dataset_p2w
from src.utils.directory import mkdir_join, mkdir
from src.bin.visualization.utils.attention import plot_hierarchical_attention_weights
from src.utils.config import load_config
from src.utils.io.labels.word import Word2char
from src.utils.evaluation.edit_distance import wer_align
from src.utils.evaluation.normalization import normalize

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

# main task
parser.add_argument('--beam_width', type=int, default=1,
                    help='the size of beam of the main task')
parser.add_argument('--length_penalty', type=float, default=0,
                    help='length penalty of the main task')
parser.add_argument('--coverage_penalty', type=float, default=0,
                    help='coverage penalty of the main task')
parser.add_argument('--rnnlm_weight', type=float, default=0,
                    help='the weight of RNNLM score of the main task')
parser.add_argument('--rnnlm_path', default=None, type=str,  nargs='?',
                    help='path to the RMMLM of the main task')

# sub task
parser.add_argument('--beam_width_sub', type=int, default=1,
                    help='the size of beam of the sub task')
parser.add_argument('--length_penalty_sub', type=float, default=0,
                    help='length penalty of the sub task')
parser.add_argument('--coverage_penalty_sub', type=float, default=0,
                    help='coverage penalty_sub of the sub task')
parser.add_argument('--rnnlm_weight_sub', type=float, default=0,
                    help='the weight of RNNLM score of the sub task')
parser.add_argument('--rnnlm_path_sub', default=None, type=str, nargs='?',
                    help='path to the RMMLM of the sub task')

parser.add_argument('--joint_decoding', type=strtobool, default=False)
parser.add_argument('--score_sub_weight', type=float, default=0)
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
    MIN_DECODE_LEN_RATIO_CHAR = 0.1

    MAX_DECODE_LEN_PHONE = 300
    MIN_DECODE_LEN_PHONE = 1
    MAX_DECODE_LEN_RATIO_PHONE = 1
    MIN_DECODE_LEN_RATIO_PHONE = 0.05
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

    MAX_DECODE_LEN_PHONE = 200
    MIN_DECODE_LEN_PHONE = 1
    MAX_DECODE_LEN_RATIO_PHONE = 1
    MIN_DECODE_LEN_RATIO_PHONE = 0
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
            model_type=config['model_type'],
            input_freq=config['input_freq'],
            use_delta=config['use_delta'],
            use_double_delta=config['use_double_delta'],
            data_size=config['data_size'] if 'data_size' in config.keys(
            ) else '',
            vocab=config['vocab'],
            data_type=args.data_type,
            label_type=config['label_type'],
            label_type_sub=config['label_type_sub'],
            batch_size=args.eval_batch_size,
            sort_utt=False, reverse=False, tool=config['tool'])
    elif config['input_type'] == 'text':
        dataset = Dataset_p2w(
            corpus=args.corpus,
            data_save_path=args.data_save_path,
            model_type=config['model_type'],
            data_type=args.data_type,
            data_size=config['data_size'] if 'data_size' in config.keys(
            ) else '',
            vocab=config['vocab'],
            label_type_in=config['label_type_in'],
            label_type=config['label_type'],
            label_type_sub=config['label_type_sub'],
            batch_size=args.eval_batch_size,
            sort_utt=False, reverse=False, tool=config['tool'],
            use_ctc=config['model_type'] == 'hierarchical_ctc',
            subsampling_factor=2 ** sum(config['subsample_list']),
            use_ctc_sub=config['model_type'] == 'hierarchical_ctc' or (
                config['model_type'] == 'hierarchical_attention' and config['ctc_loss_weight_sub'] > 0),
            subsampling_factor_sub=2 ** sum(config['subsample_list'][:config['encoder_num_layers_sub'] - 1]))
        config['num_classes_input'] = dataset.num_classes_in

    config['num_classes'] = dataset.num_classes
    config['num_classes_sub'] = dataset.num_classes_sub

    # For cold fusion
    if config['rnnlm_fusion_type'] and config['rnnlm_path']:
        # Load a RNNLM config file
        config['rnnlm_config'] = load_config(
            join(args.model_path, 'config_rnnlm.yml'))
        assert config['label_type'] == config['rnnlm_config']['label_type']
        config['rnnlm_config']['num_classes'] = dataset.num_classes
    else:
        config['rnnlm_config'] = None

    if config['rnnlm_fusion_type'] and config['rnnlm_path_sub']:
        # Load a RNNLM config file
        config['rnnlm_config_sub'] = load_config(
            join(args.model_path, 'config_rnnlm_sub.yml'))
        assert config['label_type_sub'] == config['rnnlm_config_sub']['label_type']
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
        model.rnnlm_1_fwd = rnnlm_sub

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    save_path = mkdir_join(args.model_path, 'att_weights')

    word2char = Word2char(dataset.vocab_file_path,
                          dataset.vocab_file_path_sub)

    # Clean directory
    if save_path is not None and isdir(save_path):
        shutil.rmtree(save_path)
        mkdir(save_path)

    for batch, is_new_epoch in dataset:
        # Decode
        if model.model_type == 'hierarchical_attention' and args.joint_decoding:
            best_hyps, aw, best_hyps_sub, aw_sub, perm_idx = model.decode(
                batch['xs'],
                beam_width=args.beam_width,
                max_decode_len=MAX_DECODE_LEN_WORD,
                min_decode_len=MIN_DECODE_LEN_WORD,
                min_decode_len_ratio=MIN_DECODE_LEN_RATIO_WORD,
                length_penalty=args.length_penalty,
                coverage_penalty=args.coverage_penalty,
                rnnlm_weight=args.rnnlm_weight,
                joint_decoding=args.joint_decoding,
                space_index=dataset.char2idx('_')[0],
                oov_index=dataset.word2idx('OOV')[0],
                word2char=word2char,
                idx2word=dataset.idx2word,
                idx2char=dataset.idx2char,
                score_sub_weight=args.score_sub_weight,
                exclude_eos=False)

            best_hyps_sub, aw_sub, _ = model.decode(
                batch['xs'],
                beam_width=args.beam_width_sub,
                max_decode_len=MAX_DECODE_LEN_CHAR,
                min_decode_len=MIN_DECODE_LEN_CHAR,
                min_decode_len_ratio=MIN_DECODE_LEN_RATIO_CHAR,
                length_penalty=args.length_penalty_sub,
                coverage_penalty=args.coverage_penalty_sub,
                rnnlm_weight=args.rnnlm_weight_sub,
                task_index=1,
                exclude_eos=False)
        else:
            best_hyps, aw, perm_idx = model.decode(
                batch['xs'],
                beam_width=args.beam_width,
                max_decode_len=MAX_DECODE_LEN_WORD,
                min_decode_len=MIN_DECODE_LEN_WORD,
                min_decode_len_ratio=MIN_DECODE_LEN_RATIO_WORD,
                length_penalty=args.length_penalty,
                coverage_penalty=args.coverage_penalty,
                rnnlm_weight=args.rnnlm_weight,
                exclude_eos=False)
            best_hyps_sub, aw_sub, _ = model.decode(
                batch['xs'],
                beam_width=args.beam_width_sub,
                max_decode_len=MAX_DECODE_LEN_CHAR,
                min_decode_len=MIN_DECODE_LEN_CHAR,
                min_decode_len_ratio=MIN_DECODE_LEN_RATIO_CHAR,
                length_penalty=args.length_penalty_sub,
                coverage_penalty=args.coverage_penalty_sub,
                rnnlm_weight=args.rnnlm_weight_sub,
                task_index=1,
                exclude_eos=False)

        ys = [batch['ys'][i] for i in perm_idx]

        for b in range(len(batch['xs'])):
            word_list = dataset.idx2word(best_hyps[b], return_list=True)
            char_list = dataset.idx2char(best_hyps_sub[b], return_list=True)

            if args.corpus == 'csj':
                speaker = batch['input_names'][b].split('_')[0]
            elif args.corpus == 'swbd':
                speaker = '_'.join(batch['input_names'][b].split('_')[:2])
            elif args.corpus == 'librispeech':
                speaker = '-'.join(batch['input_names'][b].split('-')[:2])
            else:
                speaker = ''

            plot_hierarchical_attention_weights(
                aw[b][:len(word_list)],
                aw_sub[b][:len(char_list)],
                label_list=word_list,
                label_list_sub=char_list,
                spectrogram=batch['xs'][b][:,
                                           :dataset.input_freq] if config['input_type'] == 'speech' else None,
                save_path=mkdir_join(save_path, speaker,
                                     batch['input_names'][b] + '.png'),
                figsize=(40, 8)
            )

            # Reference
            if dataset.is_test:
                str_ref = ys[b]
                # NOTE: transcript is seperated by space('_')
            else:
                str_ref = dataset.idx2word(ys[b])

            # Hypothesis
            str_hyp = dataset.idx2word(best_hyps[b])
            str_hyp_sub = dataset.idx2char(best_hyps_sub[b])
            str_hyp = normalize(str_hyp, remove_tokens=['>'])
            str_hyp_sub = normalize(str_hyp_sub, remove_tokens=['>'])

            sys.stdout = open(
                join(save_path, speaker, batch['input_names'][b] + '.txt'), 'w')
            wer = wer_align(ref=str_ref.split('_'),
                            hyp=str_hyp.split('_'),
                            normalize=True,
                            japanese=True if dataset.corpus == 'csj' else False)[0]
            print('\nWER (main)  : %.3f %%' % wer)
            if 'character' in dataset.label_type_sub and 'nowb' not in dataset.label_type_sub:
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

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
