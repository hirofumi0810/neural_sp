#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot the CTC posteriors (Switchboard corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath, isdir
import sys
import yaml
import argparse
import shutil

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.swbd.data.load_dataset import Dataset
from utils.io.labels.character import Idx2char
from utils.io.labels.word import Idx2word
from utils.directory import mkdir_join, mkdir
from utils.visualization.ctc import plot_ctc_probs

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')


def main():

    args = parser.parse_args()

    # Load config file
    with open(join(args.model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Get voabulary number (excluding a blank class)
    with open('../metrics/vocab_num.yml', "r") as f:
        vocab_num = yaml.load(f)
        params['num_classes'] = vocab_num[params['data_size']
                                          ][params['label_type']]

    # Load model
    model = load(model_type=params['model_type'],
                 params=params,
                 backend=params['backend'])

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    # Restore the saved model
    checkpoint = model.load_checkpoint(
        save_path=args.model_path, epoch=args.epoch)
    model.load_state_dict(checkpoint['state_dict'])

    # Load dataset
    vocab_file_path = '../metrics/vocab_files/' + \
        params['label_type'] + '_' + params['data_size'] + '.txt'
    test_data = Dataset(
        backend=params['backend'],
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        model_type=params['model_type'],
        data_type='eval2000_swbd',
        # data_type='eval2000_ch',
        data_size=params['data_size'],
        label_type=params['label_type'], vocab_file_path=vocab_file_path,
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=True, reverse=True, save_format=params['save_format'])

    space_index = 37 if params['label_type'] == 'character' else None
    # NOTE: index 0 is reserved for blank in warpctc_pytorch

    # Visualize
    plot(model=model,
         dataset=test_data,
         eval_batch_size=args.eval_batch_size,
         save_path=mkdir_join(args.model_path, 'ctc_probs'),
         space_index=space_index)


def plot(model, dataset, eval_batch_size=None, save_path=None,
         space_index=None):
    """
    Args:
        model: the model to evaluate
        dataset: An instance of a `Dataset` class
        eval_batch_size (int, optional): the batch size when evaluating the model
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

    vocab_file_path = '../metrics/vocab_files/' + \
        dataset.label_type + '_' + dataset.data_size + '.txt'
    if dataset.label_type == 'character':
        map_fn = Idx2char(vocab_file_path)
    elif dataset.label_type == 'character_capital_divide':
        map_fn = Idx2char(vocab_file_path, capital_divide=True)
    else:
        map_fn = Idx2word(vocab_file_path)

    for batch, is_new_epoch in dataset:

        # Get CTC probs
        probs = model.posteriors(batch['xs'], batch['x_lens'], temperature=1)
        # NOTE: probs: '[B, T, num_classes]'

        # Decode
        labels_pred = model.decode(batch['xs'], batch['x_lens'], beam_width=1)

        # Visualize
        for i_batch in range(len(batch['xs'])):

            # Convert from list of index to string
            str_pred = map_fn(labels_pred[i_batch])

            speaker = batch['input_names'][i_batch].split('_')[0]
            plot_ctc_probs(
                probs[i_batch, :batch['x_lens'][i_batch], :],
                frame_num=batch['x_lens'][i_batch],
                num_stack=dataset.num_stack,
                space_index=space_index,
                str_pred=str_pred,
                save_path=mkdir_join(save_path, speaker, batch['input_names'][i_batch] + '.png'))

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
