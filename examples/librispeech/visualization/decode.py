#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Decode the model's outputs (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import yaml
import argparse
import re

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.librispeech.data.load_dataset import Dataset
from utils.io.labels.character import Idx2char
from utils.io.labels.word import Idx2word

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--beam_width', type=int, default=1,
                    help='beam_width (int, optional): beam width for beam search.' +
                    ' 1 disables beam search, which mean greedy decoding.')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')
parser.add_argument('--max_decode_len', type=int, default=600,  # or 100
                    help='the length of output sequences to stop prediction when EOS token have not been emitted')


def main():

    args = parser.parse_args()

    # Load config file
    with open(join(args.model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Get voabulary number (excluding blank, <SOS>, <EOS> classes)
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
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        model_type=params['model_type'],
        data_type='test_clean',
        # data_type='test_other',
        data_size=params['data_size'],
        label_type=params['label_type'], vocab_file_path=vocab_file_path,
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=True, reverse=True, save_format=params['save_format'])

    # Visualize
    decode(model=model,
           model_type=params['model_type'],
           dataset=test_data,
           beam_width=args.beam_width,
           max_decode_len=args.max_decode_len,
           eval_batch_size=args.eval_batch_size,
           save_path=None)
    # save_path=args.model_path)


def decode(model, model_type, dataset, beam_width,
           max_decode_len, eval_batch_size=None, save_path=None):
    """Visualize label outputs.
    Args:
        model: the model to evaluate
        model_type (string): ctc or attention
        dataset: An instance of a `Dataset` class
        beam_width: (int): the size of beam
        max_decode_len (int): the length of output sequences
            to stop prediction when EOS token have not been emitted.
            This is used for seq2seq models.
        eval_batch_size (int, optional): the batch size when evaluating the model
        save_path (string): path to save decoding results
    """
    # Set batch size in the evaluation
    if eval_batch_size is not None:
        dataset.batch_size = eval_batch_size

    vocab_file_path = '../metrics/vocab_files/' + \
        dataset.label_type + '_' + dataset.data_size + '.txt'
    if dataset.label_type == 'character':
        map_fn = Idx2char(vocab_file_path)
    elif dataset.label_type == 'character_capital_divide':
        map_fn = Idx2char(vocab_file_path, capital_divide=True)
    else:
        map_fn = Idx2word(vocab_file_path)

    if save_path is not None:
        sys.stdout = open(join(model.model_dir, 'decode.txt'), 'w')

    for batch, is_new_epoch in dataset:

        inputs, labels, inputs_seq_len, labels_seq_len, input_names = batch

        # Decode
        labels_pred = model.decode(inputs, inputs_seq_len,
                                   beam_width=beam_width,
                                   max_decode_len=max_decode_len)

        for i_batch in range(inputs.shape[0]):
            print('----- wav: %s -----' % input_names[i_batch])

            ##############################
            # Reference
            ##############################
            if dataset.is_test:
                str_true = labels[i_batch][0]
                # NOTE: transcript is seperated by space('_')
            else:
                # Convert from list of index to string
                if model_type == 'ctc':
                    str_true = map_fn(
                        labels[i_batch][:labels_seq_len[i_batch]])
                elif model_type == 'attention':
                    str_true = map_fn(
                        labels[i_batch][1:labels_seq_len[i_batch] - 1])
                    # NOTE: Exclude <SOS> and <EOS>

            ##############################
            # Hypothesis
            ##############################
            # Convert from list of index to string
            str_pred = map_fn(labels_pred[i_batch])

            if model_type == 'attention':
                str_pred = str_pred.split('>')[0]
                # NOTE: Trancate by the first <EOS>

                # Remove the last space
                if len(str_pred) > 0 and str_pred[-1] == '_':
                    str_pred = str_pred[:-1]

            # Remove consecutive spaces
            str_pred = re.sub(r'[_]+', '_', str_pred)

            ##############################
            # Post-proccessing
            ##############################
            # Remove garbage labels
            str_true = re.sub(r'[\'<>]+', '', str_true)
            str_pred = re.sub(r'[\'<>]+', '', str_pred)

            print('Ref: %s' % str_true.replace('_', ' '))
            print('Hyp: %s' % str_pred.replace('_', ' '))

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
