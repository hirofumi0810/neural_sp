#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot attention weights (TIMIT corpus)."""

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
from utils.visualization.attention import plot_attention_weights
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
parser.add_argument('--beam_width', type=int, default=1,
                    help='the size of beam')
parser.add_argument('--length_penalty', type=float, default=0,
                    help='length penalty in beam search decoding')
parser.add_argument('--coverage_penalty', type=float, default=0,
                    help='coverage penalty in beam search decoding')

MAX_DECODE_LEN_PHONE = 71
MIN_DECODE_LEN_PHONE = 13


def main():

    args = parser.parse_args()

    # Load a config file (.yml)
    params = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Load dataset
    eval_data = Dataset(
        data_save_path=args.data_save_path,
        backend=params['backend'],
        input_freq=params['input_freq'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        data_type='test', label_type=params['label_type'],
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False, tool=params['tool'])

    params['num_classes'] = eval_data.num_classes

    # Load model
    model = load(model_type=params['model_type'],
                 params=params,
                 backend=params['backend'])

    # Restore the saved parameters
    model.load_checkpoint(save_path=args.model_path, epoch=args.epoch)

    # GPU setting
    model.set_cuda(deterministic=False, benchmark=True)

    # Visualize
    plot_attention(model=model,
                   dataset=eval_data,
                   eval_batch_size=args.eval_batch_size,
                   beam_width=args.beam_width,
                   length_penalty=args.length_penalty,
                   coverage_penalty=args.coverage_penalty,
                   save_path=mkdir_join(args.model_path, 'att_weights'))


def plot_attention(model, dataset, eval_batch_size, beam_width,
                   length_penalty, coverage_penalty, save_path=None):
    """Visualize attention weights of the attetnion-based model.
    Args:
        model: model to evaluate
        dataset: An instance of a `Dataset` class
        eval_batch_size (int): the batch size when evaluating the model
        beam_width: (int): the size of beam
        length_penalty (float): length penalty in beam search decoding
        coverage_penalty (float): coverage penalty in beam search decoding
        save_path (string, optional): path to save attention weights plotting
    """
    # Clean directory
    if save_path is not None and isdir(save_path):
        shutil.rmtree(save_path)
        mkdir(save_path)

    idx2phone = Idx2phone(dataset.vocab_file_path)

    for batch, is_new_epoch in dataset:

        # Decode
        best_hyps, aw, perm_idx = model.decode(
            batch['xs'], batch['x_lens'],
            beam_width=beam_width,
            max_decode_len=MAX_DECODE_LEN_PHONE,
            min_decode_len=MIN_DECODE_LEN_PHONE,
            length_penalty=length_penalty,
            coverage_penalty=coverage_penalty)

        ys = batch['ys'][perm_idx]
        y_lens = batch['y_lens'][perm_idx]

        for b in range(len(batch['xs'])):
            ##############################
            # Reference
            ##############################
            if dataset.is_test:
                str_ref = ys[b][0]
                # NOTE: transcript is seperated by space(' ')
            else:
                # Convert from list of index to string
                str_ref = idx2phone(ys[b][:y_lens[b]])

            token_list = idx2phone(best_hyps[b])

            plot_attention_weights(
                aw[b][:len(token_list), :batch['x_lens'][b]],
                # label_list=token_list,
                label_list=[],
                spectrogram=batch['xs'][b, :, :dataset.input_freq],
                str_ref=str_ref,
                save_path=join(save_path, batch['input_names'][b] + '.png'),
                figsize=(20, 8))

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
