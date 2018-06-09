#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot the hierarchical CTC posteriors (Switchboard corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath, isdir
import sys
import argparse
import shutil

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.swbd.s5c.exp.dataset.load_dataset_hierarchical import Dataset
from utils.directory import mkdir_join, mkdir
from utils.visualization.ctc import plot_hierarchical_ctc_probs
from utils.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument('--data_save_path', type=str,
                    help='path to saved data')
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')


def main():

    args = parser.parse_args()

    # Load a config file (.yml)
    params = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Load dataset
    dataset = Dataset(
        data_save_path=args.data_save_path,
        input_freq=params['input_freq'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        data_type='eval2000_swbd',
        # data_type='eval2000_ch',
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

    save_path = mkdir_join(args.model_path, 'ctc_probs')

    ######################################################################

    # Clean directory
    if save_path is not None and isdir(save_path):
        shutil.rmtree(save_path)
        mkdir(save_path)

    for batch, is_new_epoch in dataset:
        # Get CTC probs
        probs = model.posteriors(batch['xs'], batch['x_lens'],
                                 temperature=1)
        probs_sub = model.posteriors(batch['xs'], batch['x_lens'], task_index=1,
                                     temperature=1)
        # NOTE: probs: '[B, T, num_classes]'
        # NOTE: probs_sub: '[B, T, num_classes_sub]'

        # Visualize
        for b in range(len(batch['xs'])):
            speaker = batch['input_names'][b].split('_')[0]
            plot_hierarchical_ctc_probs(
                probs[b, :batch['x_lens'][b], :],
                probs_sub[b, :batch['x_lens'][b], :],
                frame_num=batch['x_lens'][b],
                num_stack=dataset.num_stack,
                spectrogram=batch['xs'][b, :, :dataset.input_freq],
                save_path=mkdir_join(save_path, speaker,
                                     batch['input_names'][b] + '.png'),
                figsize=(40, 8))

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
