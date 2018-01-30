#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Decode the hierarchical model's outputs (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import argparse

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.csj.data.load_dataset_hierarchical import Dataset
from utils.io.labels.character import Idx2char
from utils.io.labels.word import Idx2word
from utils.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')
parser.add_argument('--beam_width', type=int, default=1,
                    help='beam_width (int, optional): beam width for beam search.' +
                    ' 1 disables beam search, which mean greedy decoding.')
parser.add_argument('--max_decode_len', type=int, default=60,
                    help='the length of output sequences to stop prediction when EOS token have not been emitted')
parser.add_argument('--max_decode_len_sub', type=int, default=100,
                    help='the length of output sequences to stop prediction when EOS token have not been emitted')


def main():

    args = parser.parse_args()

    # Load a config file (.yml)
    params = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Load dataset
    vocab_file_path = '../metrics/vocab_files/' + \
        params['label_type'] + '_' + params['data_size'] + '.txt'
    vocab_file_path_sub = '../metrics/vocab_files/' + \
        params['label_type_sub'] + '_' + params['data_size'] + '.txt'
    test_data = Dataset(
        backend=params['backend'],
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        model_type=params['model_type'],
        data_type='eval1',
        # data_type='eval2',
        # data_type='eval3',
        data_size=params['data_size'],
        label_type=params['label_type'], label_type_sub=params['label_type_sub'],
        vocab_file_path=vocab_file_path,
        vocab_file_path_sub=vocab_file_path_sub,
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=True, reverse=True, save_format=params['save_format'])
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

    # Visualize
    decode(model=model,
           model_type=params['model_type'],
           dataset=test_data,
           beam_width=args.beam_width,
           max_decode_len=args.max_decode_len,
           max_decode_len_sub=args.max_decode_len_sub,
           eval_batch_size=args.eval_batch_size,
           save_path=None)
    # save_path=args.model_path)


def decode(model, model_type, dataset, beam_width,
           max_decode_len, max_decode_len_sub,
           eval_batch_size=None, save_path=None):
    """Visualize label outputs.
    Args:
        model: the model to evaluate
        model_type (string): hierarchical_ctc or hierarchical_attention
        dataset: An instance of a `Dataset` class
        beam_width: (int): the size of beam
        max_decode_len (int): the length of output sequences
            to stop prediction when EOS token have not been emitted.
            This is used for seq2seq models.
        max_decode_len_sub (int):
        eval_batch_size (int, optional): the batch size when evaluating the model
        save_path (string): path to save decoding results
    """
    # Set batch size in the evaluation
    if eval_batch_size is not None:
        dataset.batch_size = eval_batch_size

    if dataset.label_type == 'pos':
        map_fn_main = Idx2word(
            vocab_file_path='../metrics/vocab_files/' +
            dataset.label_type + '_' + dataset.data_size + '.txt')
        map_fn_sub = Idx2word(
            vocab_file_path='../metrics/vocab_files/' +
            dataset.label_type_sub + '_' + dataset.data_size + '.txt')
    else:
        map_fn_main = Idx2word(
            vocab_file_path='../metrics/vocab_files/' +
            dataset.label_type + '_' + dataset.data_size + '.txt')
        map_fn_sub = Idx2char(
            vocab_file_path='../metrics/vocab_files/' +
            dataset.label_type_sub + '_' + dataset.data_size + '.txt')

    if save_path is not None:
        sys.stdout = open(join(model.model_dir, 'decode.txt'), 'w')

    for batch, is_new_epoch in dataset:

        # Decode
        best_hyps = model.decode(batch['xs'], batch['x_lens'],
                                 beam_width=beam_width,
                                 max_decode_len=max_decode_len)
        best_hyps_sub = model.decode(batch['xs'], batch['x_lens'],
                                     beam_width=beam_width,
                                     max_decode_len=max_decode_len_sub,
                                     is_sub_task=True)

        for i_batch in range(len(batch['xs'])):
            print('----- wav: %s -----' % batch['input_names'][i_batch])

            ##############################
            # Reference
            ##############################
            if dataset.is_test:
                str_true = batch['ys'][i_batch][0]
                str_true_sub = batch['ys_sub'][i_batch][0]
                # NOTE: transcript is seperated by space('_')
            else:
                # Convert from list of index to string
                if model_type == 'hierarchical_ctc':
                    str_true = map_fn_main(
                        batch['ys'][i_batch][:batch['y_lens'][i_batch]])
                    str_true_sub = map_fn_main(
                        batch['ys_sub'][i_batch][:batch['y_lens_sub'][i_batch]])
                elif model_type == 'hierarchical_attention':
                    str_true = map_fn_main(
                        batch['ys'][i_batch][1:batch['y_lens'][i_batch] - 1])
                    str_true_sub = map_fn_main(
                        batch['ys_sub'][i_batch][1:batch['y_lens_sub'][i_batch] - 1])
                    # NOTE: Exclude <SOS> and <EOS>

            ##############################
            # Hypothesis
            ##############################
            # Convert from list of index to string
            str_pred = map_fn_main(best_hyps[i_batch])
            str_pred_sub = map_fn_sub(best_hyps_sub[i_batch])

            if model_type == 'hierarchical_attention':
                str_pred = str_pred.split('>')[0]
                str_pred_sub = str_pred_sub.split('>')[0]
                # NOTE: Trancate by the first <EOS>

                # Remove the last space
                if len(str_pred) > 0 and str_pred[-1] == '_':
                    str_pred = str_pred[:-1]
                if len(str_pred_sub) > 0 and str_pred_sub[-1] == '_':
                    str_pred_sub = str_pred_sub[:-1]

            print('Ref (main): %s' % str_true.replace('_', ' '))
            print('Hyp (main): %s' % str_pred.replace('_', ' '))
            print('Ref (sub): %s' % str_true_sub.replace('_', ' '))
            print('Hyp (sub): %s' % str_pred_sub.replace('_', ' '))

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
