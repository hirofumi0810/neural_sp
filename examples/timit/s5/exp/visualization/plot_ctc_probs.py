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
from examples.timit.s5.exp.dataset.load_dataset import Dataset
from utils.io.labels.phone import Idx2phone
from utils.directory import mkdir_join, mkdir
from utils.visualization.ctc import plot_ctc_probs
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
    test_data = Dataset(
        data_save_path=args.data_save_path,
        backend=params['backend'],
        input_freq=params['input_freq'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        data_type='test', label_type=params['label_type'],
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=True, reverse=True, tool=params['tool'])

    params['num_classes'] = test_data.num_classes

    # Load model
    model = load(model_type=params['model_type'],
                 params=params,
                 backend=params['backend'])

    # Restore the saved parameters
    model.load_checkpoint(save_path=args.model_path, epoch=args.epoch)

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    # Visualize
    plot_probs(model=model,
               dataset=test_data,
               eval_batch_size=args.eval_batch_size,
               save_path=mkdir_join(args.model_path, 'ctc_probs'))


def plot_probs(model, dataset, eval_batch_size, save_path=None):
    """Plot CTC posteriors.
    Args:
        model: model to evaluate
        dataset: An instance of a `Dataset` class
        eval_batch_size (int): the batch size when evaluating the model
        save_path (string): path to save figures of CTC posteriors
    """
    # Clean directory
    if save_path is not None and isdir(save_path):
        shutil.rmtree(save_path)
        mkdir(save_path)

    for batch, is_new_epoch in dataset:

        # Get CTC probs
        probs, x_lens, _ = model.posteriors(
            batch['xs'], batch['x_lens'], temperature=1)
        # NOTE: probs: '[B, T, num_classes]'

        # Visualize
        for b in range(len(batch['xs'])):
            plot_ctc_probs(
                probs[b, : x_lens[b], :],
                frame_num=x_lens[b],
                num_stack=dataset.num_stack,
                spectrogram=batch['xs'][b, :, :40],
                save_path=join(save_path, batch['input_names'][b] + '.png'),
                figsize=(14, 7))

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
