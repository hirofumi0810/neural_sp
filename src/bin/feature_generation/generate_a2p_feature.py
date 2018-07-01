#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate phoneme labels by the A2P model for the P2W model on the modular training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import argparse
from tqdm import tqdm

sys.path.append(abspath('../../../'))
from src.models.load_model import load
from src.dataset.loader import Dataset
from src.utils.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str,
                    help='the name of corpus')
parser.add_argument('--data_type', type=str,
                    help='the type of data (ex. train, dev etc.)')
parser.add_argument('--data_save_path', type=str,
                    help='path to saved data')
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate (A2P)')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')
parser.add_argument('--beam_width', type=int, default=1,
                    help='the size of beam (A2P)')
parser.add_argument('--length_penalty', type=float, default=0,
                    help='length penalty')
parser.add_argument('--coverage_penalty', type=float, default=0,
                    help='coverage penalty')
args = parser.parse_args()

# corpus depending
if args.corpus == 'csj':
    MAX_DECODE_LEN_PHONE = 200
    MIN_DECODE_LEN_PHONE = 1
    MAX_DECODE_LEN_RATIO_PHONE = 1
    MIN_DECODE_LEN_RATIO_PHONE = 0
elif args.corpus == 'swbd':
    MAX_DECODE_LEN_PHONE = 300
    MIN_DECODE_LEN_PHONE = 1
    MAX_DECODE_LEN_RATIO_PHONE = 1
    MIN_DECODE_LEN_RATIO_PHONE = 0
elif args.corpus == 'librispeech':
    raise NotImplementedError
elif args.corpus == 'wsj':
    raise NotImplementedError
elif args.corpus == 'timit':
    raise NotImplementedError
else:
    raise ValueError(args.corpus)


def main():

    # Load a A2P config file
    config = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Load dataset
    dataset = Dataset(
        corpus=args.corpus,
        data_save_path=args.data_save_path,
        model_type=config['model_type'],
        input_freq=config['input_freq'],
        use_delta=config['use_delta'],
        use_double_delta=config['use_double_delta'],
        data_size=config['data_size'] if 'data_size' in config.keys(
        ) else '',
        data_type=args.data_type,  # train
        label_type=config['label_type'],
        batch_size=args.eval_batch_size,
        max_epoch=1,
        max_frame_num=config['max_frame_num'] if 'max_frame_num' in config.keys(
        ) else 10000,
        min_frame_num=config['min_frame_num'] if 'min_frame_num' in config.keys(
        ) else 0,
        sort_utt=True, tool=config['tool'],
        use_ctc=config['model_type'] == 'ctc' or (
            config['model_type'] == 'attention' and config['ctc_loss_weight'] > 0),
        subsampling_factor=2 ** sum(config['subsample_list']))
    config['num_classes'] = dataset.num_classes

    # Load the A2P model
    model = load(model_type=config['model_type'],
                 config=config,
                 backend=config['backend'])

    # Restore the saved parameters
    model.load_checkpoint(save_path=args.model_path, epoch=args.epoch)

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    # Copy original csv file
    df = dataset.df

    # Decoder per utterace
    pbar = tqdm(total=len(df))
    for batch, is_new_epoch in dataset:
        # Decode
        best_hyps, _, perm_idx = model.decode(
            batch['xs'],
            beam_width=args.beam_width,
            max_decode_len=MAX_DECODE_LEN_PHONE,
            min_decode_len=MIN_DECODE_LEN_PHONE,
            length_penalty=args.length_penalty,
            min_decode_len_ratio=MIN_DECODE_LEN_RATIO_PHONE,
            coverage_penalty=args.coverage_penalty)

        for b in range(len(batch['xs'])):
            if len(best_hyps[b]) < 1:
                print('skip (x_len: %d)' % len(batch['xs'][perm_idx[b]]))
                continue

            phone_list = dataset.idx2phone(best_hyps[b]).split('_')
            df.loc[batch['index'][perm_idx[b]],
                   'transcript'] = ' '.join(phone_list)
            pbar.update(1)

        if is_new_epoch:
            pbar.close()
            break

    # Save as a new file
    dataset_path = dataset.dataset_save_path.replace(
        config['label_type'], config['label_type'] + '_a2p_' + model.model_type + '_beam' + args.beam_width)
    df.to_csv(dataset_path, encoding='utf-8')


if __name__ == '__main__':
    main()
