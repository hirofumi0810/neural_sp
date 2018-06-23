#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot the CTC posteriors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath, isdir
import sys
import argparse
import shutil

sys.path.append(abspath('../../../'))
from src.models.load_model import load
from src.dataset.loader import Dataset
from src.utils.directory import mkdir_join, mkdir
from src.utils.visualization.ctc import plot_ctc_probs
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
args = parser.parse_args()


def main():

    # Load a config file
    config = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Load dataset
    dataset = Dataset(
        corpus=args.corpus,
        data_save_path=args.data_save_path,
        input_freq=config['input_freq'],
        use_delta=config['use_delta'],
        use_double_delta=config['use_double_delta'],
        data_size=config['data_size'] if 'data_size' in config.keys() else '',
        data_type=args.data_type,
        label_type=config['label_type'],
        batch_size=args.eval_batch_size,
        sort_utt=False, reverse=False, tool=config['tool'])
    config['num_classes'] = dataset.num_classes

    # Load model
    model = load(model_type=config['model_type'],
                 config=config,
                 backend=config['backend'])

    # Restore the saved parameters
    model.load_checkpoint(save_path=args.model_path, epoch=args.epoch)

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    save_path = mkdir_join(args.model_path, 'ctc_probs')

    # Clean directory
    if save_path is not None and isdir(save_path):
        shutil.rmtree(save_path)
        mkdir(save_path)

    for batch, is_new_epoch in dataset:
        # Get CTC probs
        probs, x_lens, _ = model.posteriors(batch['xs'], temperature=1)
        # NOTE: probs: '[B, T, num_classes]'

        # Visualize
        for b in range(len(batch['xs'])):
            if args.corpus == 'csj':
                speaker = batch['input_names'][b].split('_')[0]
            elif args.corpus == 'swbd':
                speaker = '_'.join(batch['input_names'][b].split('_')[:2])
            elif args.corpus == 'librispeech':
                speaker = '-'.join(batch['input_names'][b].split('-')[:2])
            else:
                speaker = ''

            plot_ctc_probs(
                probs[b, :x_lens[b]],
                frame_num=x_lens[b],
                num_stack=model.num_stack,
                spectrogram=batch['xs'][b][:, :dataset.input_freq],
                save_path=mkdir_join(save_path, speaker,
                                     batch['input_names'][b] + '.png'),
                figsize=(20, 8))

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
