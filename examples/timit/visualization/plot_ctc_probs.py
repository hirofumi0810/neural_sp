#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot the CTC posteriors (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath, isdir
import sys
import argparse
import shutil

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.timit.data.load_dataset import Dataset
from utils.io.labels.phone import Idx2phone
from utils.directory import mkdir_join, mkdir
from utils.visualization.ctc import plot_ctc_probs
from utils.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')


def main():

    args = parser.parse_args()

    # Load a config file (.yml)
    params = load_config(join(args.model_path, 'config.yml'))

    # Load model
    model = load(model_type=params['model_type'],
                 params=params,
                 backend=params['backend'])

    # Restore the saved parameters
    model.load_checkpoint(save_path=args.model_path, epoch=args.epoch)

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    # Load dataset
    vocab_file_path = '../metrics/vocab_files/' + params['label_type'] + '.txt'
    test_data = Dataset(
        backend=params['backend'],
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        model_type=params['model_type'],
        data_type='test', label_type=params['label_type'],
        vocab_file_path=vocab_file_path,
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=True, reverse=True, save_format=params['save_format'])

    # Visualize
    plot_probs(model=model,
               dataset=test_data,
               label_type=params['label_type'],
               eval_batch_size=args.eval_batch_size,
               save_path=mkdir_join(args.model_path, 'ctc_probs'))


def plot_probs(model, dataset, label_type, eval_batch_size=None,
               save_path=None):
    """
    Args:
        dataset ():
        eval_batch_size (int, optional): the batch size when evaluating the model
        save_path (string): path to save figures of CTC posteriors
    """
    # Set batch size in the evaluation
    if eval_batch_size is not None:
        dataset.batch_size = eval_batch_size

    # Clean directory
    if isdir(save_path):
        shutil.rmtree(save_path)
        mkdir(save_path)

    for batch, is_new_epoch in dataset:

        # Get CTC probs
        probs = model.posteriors(batch['xs'], batch['x_lens'], temperature=1)
        # NOTE: probs: '[B, T, num_classes]'

        # Visualize
        for i_batch in range(batch['xs'].shape[0]):

            plot_ctc_probs(
                probs[i_batch, : batch['x_lens'][i_batch], :],
                frame_num=batch['x_lens'][i_batch],
                num_stack=dataset.num_stack,
                save_path=join(save_path, batch['input_names'][i_batch] + '.png'))

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
