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
from utils.io.labels.character import Idx2char
from utils.io.labels.word import Idx2word
from utils.directory import mkdir_join, mkdir
from utils.visualization.ctc import plot_hierarchical_ctc_probs
from utils.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')
parser.add_argument('--data_save_path', type=str, help='path to saved data')


def main():

    args = parser.parse_args()

    # Load a config file (.yml)
    params = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Load dataset
    test_data = Dataset(
        data_save_path=args.data_save_path,
        backend=params['backend'],
        model_type=params['model_type'],
        data_type='eval2000_swbd',
        # data_type='eval2000_ch',
        data_size=params['data_size'],
        label_type=params['label_type'], label_type_sub=params['label_type_sub'],
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=True, reverse=True, tool=params['tool'])

    params['num_classes'] = test_data.num_classes
    params['num_classes_sub'] = test_data.num_classes_sub

    # Load model
    model = load(model_type=params['model_type'],
                 params=params,
                 backend=params['backend'])

    # Restore the saved parameters
    model.load_checkpoint(save_path=args.model_path, epoch=args.epoch)

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    space_index = 37 if params['label_type_sub'] == 'character' else None
    # NOTE: index 0 is reserved for blank in warpctc_pytorch

    # Visualize
    plot(model=model,
         dataset=test_data,
         eval_batch_size=args.eval_batch_size,
         save_path=mkdir_join(args.model_path, 'ctc_probs'),
         space_index=space_index)


def plot(model, dataset, eval_batch_size, save_path=None, space_index=None):
    """
    Args:
        model: the model to evaluate
        dataset: An instance of a `Dataset` class
        eval_batch_size (int): the batch size when evaluating the model
        save_path (string): path to save figures of CTC posteriors
        space_index (int, optional):
    """
    # Set batch size in the evaluation
    if eval_batch_size is not None:
        dataset.batch_size = eval_batch_size

    # Clean directory
    if isdir(save_path):
        shutil.rmtree(save_path)
        mkdir(save_path)

    idx2word = Idx2word(dataset.vocab_file_path)
    idx2char = Idx2char(dataset.vocab_file_path,
                        capital_divide=dataset.label_type_sub == 'character_capital_divide')

    for batch, is_new_epoch in dataset:

        # Get CTC probs
        probs = model.posteriors(batch['xs'], batch['x_lens'],
                                 temperature=1)
        probs_sub = model.posteriors(batch['xs'], batch['x_lens'], is_sub_task=True,
                                     temperature=1)
        # NOTE: probs: '[B, T, num_classes]'
        # NOTE: probs_sub: '[B, T, num_classes_sub]'

        # Decode
        best_hyps = model.decode(batch['xs'], batch['x_lens'],
                                 beam_width=1)
        best_hyps_sub = model.decode(batch['xs'], batch['x_lens'],
                                     beam_width=1,
                                     is_sub_task=True)

        # Visualize
        for b in range(len(batch['xs'])):

            # Convert from list of index to string
            str_hyp = idx2word(best_hyps[b])
            str_hyp_sub = idx2char(best_hyps_sub[b])

            speaker = batch['input_names'][b].split('_')[0]
            plot_hierarchical_ctc_probs(
                probs[b, :batch['x_lens'][b], :],
                probs_sub[b, :batch['x_lens'][b], :],
                frame_num=batch['x_lens'][b],
                num_stack=dataset.num_stack,
                space_index=space_index,
                str_hyp=str_hyp,
                str_hyp_sub=str_hyp_sub,
                save_path=mkdir_join(save_path, speaker, batch['input_names'][b] + '.png'))

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
