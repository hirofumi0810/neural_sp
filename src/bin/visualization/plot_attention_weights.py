#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot attention weights of the attention model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath, isdir
import sys
import argparse
import shutil

sys.path.append(abspath('../../../'))
from src.models.load_model import load
from src.dataset.loader import Dataset as Dataset_asr
from src.dataset.loader_p2w import Dataset as Dataset_p2w
from src.utils.directory import mkdir_join, mkdir
from src.bin.visualization.utils.attention import plot_attention_weights
from src.utils.config import load_config

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
                    help='the size of beam')
parser.add_argument('--length_penalty', type=float, default=0,
                    help='length penalty')
parser.add_argument('--coverage_penalty', type=float, default=0,
                    help='coverage penalty')
parser.add_argument('--rnnlm_weight', type=float, default=0,
                    help='the weight of RNNLM score')
parser.add_argument('--rnnlm_path', default=None, type=str, nargs='?',
                    help='path to the RMMLM')
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
            data_type=args.data_type,
            label_type=config['label_type'],
            batch_size=args.eval_batch_size,
            sort_utt=False, reverse=False, tool=config['tool'])
    elif config['input_type'] == 'text':
        dataset = Dataset_p2w(
            corpus=args.corpus,
            data_save_path=args.data_save_path,
            model_type=config['model_type'],
            data_type=args.data_type,
            data_size=config['data_size'],
            label_type_in=config['label_type_in'],
            label_type=config['label_type'],
            batch_size=args.eval_batch_size,
            sort_utt=False, reverse=False, tool=config['tool'],
            vocab=config['vocab'],
            use_ctc=config['model_type'] == 'ctc' or (
                config['model_type'] == 'attention' and config['ctc_loss_weight'] > 0),
            subsampling_factor=2 ** sum(config['subsample_list']))
        config['num_classes_input'] = dataset.num_classes_in

    config['num_classes'] = dataset.num_classes
    config['num_classes_sub'] = dataset.num_classes

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
        if config_rnnlm['backward']:
            model.rnnlm_0_bwd = rnnlm
        else:
            model.rnnlm_0_fwd = rnnlm

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    save_path = mkdir_join(args.model_path, 'att_weights')

    # Clean directory
    if save_path is not None and isdir(save_path):
        shutil.rmtree(save_path)
        mkdir(save_path)

    if dataset.label_type == 'word':
        map_fn = dataset.idx2word
        max_decode_len = MAX_DECODE_LEN_WORD
        min_decode_len = MIN_DECODE_LEN_WORD
        min_decode_len_ratio = MIN_DECODE_LEN_RATIO_WORD
    elif 'character' in dataset.label_type:
        map_fn = dataset.idx2char
        max_decode_len = MAX_DECODE_LEN_CHAR
        min_decode_len = MIN_DECODE_LEN_CHAR
        min_decode_len_ratio = MIN_DECODE_LEN_RATIO_CHAR
    elif 'phone' in dataset.label_type:
        map_fn = dataset.idx2phone
        max_decode_len = MAX_DECODE_LEN_PHONE
        min_decode_len = MIN_DECODE_LEN_PHONE
        min_decode_len_ratio = MIN_DECODE_LEN_RATIO_PHONE
    else:
        raise ValueError(dataset.label_type)

    for batch, is_new_epoch in dataset:
        # Decode
        best_hyps, aw, perm_idx = model.decode(
            batch['xs'],
            beam_width=args.beam_width,
            max_decode_len=max_decode_len,
            min_decode_len=min_decode_len,
            min_decode_len_ratio=min_decode_len_ratio,
            length_penalty=args.length_penalty,
            coverage_penalty=args.coverage_penalty,
            rnnlm_weight=args.rnnlm_weight,
            exclude_eos=False)

        ys = [batch['ys'][i] for i in perm_idx]

        for b in range(len(batch['xs'])):
            token_list = map_fn(best_hyps[b], return_list=True)
            if args.corpus == 'csj':
                speaker = batch['input_names'][b].split('_')[0]
            elif args.corpus == 'swbd':
                speaker = '_'.join(batch['input_names'][b].split('_')[:2])
            elif args.corpus == 'librispeech':
                speaker = '-'.join(batch['input_names'][b].split('-')[:2])
            else:
                speaker = ''

            plot_attention_weights(
                aw[b][:len(token_list)],
                label_list=token_list,
                spectrogram=batch['xs'][b][:,
                                           :dataset.input_freq] if config['input_type'] == 'speech' else None,
                save_path=mkdir_join(save_path, speaker,
                                     batch['input_names'][b] + '.png'),
                figsize=(20, 8))

            # Reference
            if dataset.is_test:
                str_ref = ys[b]
                # NOTE: transcript is seperated by space('_')
            else:
                str_ref = map_fn(ys[b])

            try:
                with open(join(save_path, speaker, batch['input_names'][b] + '.txt'), 'w') as f:
                    f.write(str_ref)
            except:
                pass

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
