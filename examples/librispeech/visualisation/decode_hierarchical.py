#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Decode the trained hierarchical model's outputs (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import yaml
import argparse

sys.path.append(abspath('../../../'))
from models.pytorch.load_model import load
from examples.librispeech.data.load_dataset_hierarchical_ctc import Dataset as Dataset_ctc
from examples.librispeech.data.load_dataset_hierarchical_attention import Dataset as Dataset_attention
from utils.io.labels.character import Idx2char
from utils.io.labels.word import Idx2word
from utils.io.variable import np2var

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--beam_width', type=int, default=1,
                    help='beam_width (int, optional): beam width for beam search.' +
                    ' 1 disables beam search, which mean greedy decoding.')
parser.add_argument('--eval_batch_size', type=str, default=1,
                    help='the size of mini-batch in evaluation')
parser.add_argument('--max_decode_length', type=int, default=100,
                    help='the length of output sequences to stop prediction when EOS token have not been emitted')
parser.add_argument('--max_decode_length_sub', type=int, default=600,
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
        params['num_classes_sub'] = vocab_num[params['data_size']
                                              ][params['label_type_sub']]

    # Model setting
    model = load(model_type=params['model_type'], params=params)

    # Load dataset
    if params['model_type'] == 'hierarchical_ctc':
        Dataset = Dataset_ctc
    elif params['model_type'] == 'hierarchical_attention':
        Dataset = Dataset_attention
    test_data = Dataset(
        data_type='test_clean',
        # data_type='test_other',
        data_size=params['data_size'],
        label_type=params['label_type'], label_type_sub=params['label_type_sub'],
        num_classes=params['num_classes'], num_classes_sub=params['num_classes_sub'],
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=True, reverse=True)

    # GPU setting
    model.set_cuda(deterministic=False)

    # Load the saved model
    checkpoint = model.load_checkpoint(
        save_path=args.model_path, epoch=args.epoch)
    model.load_state_dict(checkpoint['state_dict'])

    # Change to evaluation mode
    model.eval()

    # Visualize
    decode(model=model,
           model_type=params['model_type'],
           dataset=test_data,
           label_type=params['label_type'],
           label_type_sub=params['label_type_sub'],
           data_size=params['data_size'],
           beam_width=args.beam_width,
           max_decode_length=args.max_decode_length,
           save_path=None)
    # save_path=model.save_path)


def decode(model, model_type, dataset, label_type, label_type_sub, data_size,
           beam_width, max_decode_length=100, save_path=None):
    """Visualize label outputs.
    Args:
        model: the model to evaluate
        model_type (string): hierarchical_ctc or hierarchical_attention
        dataset: An instance of a `Dataset` class
        label_type (string): word_freq1 or word_freq5 or word_freq10 or word_freq15
        label_type_sub (string): kanji or kanji or kanji_divide or kana_divide
        data_size (string): train_fullset or train_subset
        beam_width: (int): the size of beam
        max_decode_length (int, optional): the length of output sequences
            to stop prediction when EOS token have not been emitted.
            This is used for seq2seq models.
        save_path (string): path to save decoding results
    """
    vocab_file_path = '../metrics/vocab_files/' + \
        label_type + '_' + data_size + '.txt'
    if label_type_sub == 'character':
        vocab_file_path_sub = '../metrics/vocab_files/character.txt'
    else:
        vocab_file_path_sub = '../metrics/vocab_files/' + \
            label_type_sub + '_' + data_size + '.txt'

    map_fn = Idx2word(vocab_file_path)
    map_fn_sub = Idx2char(vocab_file_path_sub)

    if save_path is not None:
        sys.stdout = open(join(model.model_dir, 'decode.txt'), 'w')

    for data, is_new_epoch in dataset:

        # Create feed dictionary for next mini batch
        inputs, labels, labels_sub, inputs_seq_len, labels_seq_len, labels_seq_len_sub, input_names = data

        # TODO: labels must be sorted when eval_batch_size >= 1

        # Wrap by variable
        inputs = np2var(inputs, use_cuda=model.use_cuda, volatile=True)
        inputs_seq_len = np2var(
            inputs_seq_len, use_cuda=model.use_cuda, volatile=True, dtype='int')

        batch_size = inputs[0].size(0)

        # Decode
        if model_type == 'hierarchical_attention':
            labels_pred, _ = model.decode_infer(
                inputs[0], inputs_seq_len[0], beam_width=beam_width, max_decode_length=max_decode_length)
            labels_pred_sub, _ = model.decode_infer_sub(
                inputs[0], inputs_seq_len[0], beam_width=beam_width, max_decode_length=max_decode_length)
        elif model_type == 'hierarchical_ctc':
            logits, logits_sub, perm_indices = model(
                inputs[0], inputs_seq_len[0])
            labels_pred = model.decode(
                logits, inputs_seq_len[0][perm_indices], beam_width=beam_width)
            labels_pred_sub = model.decode(
                logits_sub, inputs_seq_len[0][perm_indices], beam_width=beam_width)
            labels_pred -= 1
            labels_pred_sub -= 1
            # NOTE: index 0 is reserved for blank

        for i_batch in range(batch_size):
            print('----- wav: %s -----' % input_names[0][i_batch])

            ##############################
            # Reference
            ##############################
            if dataset.is_test:
                str_true = labels[0][i_batch][0].replace('_', ' ')
                # NOTE: transcript is seperated by space('_')
            else:
                # Convert from list of index to string
                if model_type == 'hierarchical_ctc':
                    word_list_true = map_fn(
                        labels[0][i_batch][:labels_seq_len[0][i_batch]])
                elif model_type == 'hierarchical_attention':
                    word_list_true = map_fn(
                        labels[0][i_batch][1:labels_seq_len[0][i_batch] - 1])
                    # NOTE: Exclude <SOS> and <EOS>
                str_true = ' '.join(word_list_true)

                # if model_type == 'hierarchical_ctc':
                #     str_true_sub = map_fn_sub(
                #         labels_sub[0][i_batch][:labels_seq_len_sub[0][i_batch]])
                # elif model_type == 'hierarchical_attention':
                #     str_true_sub = map_fn_sub(
                #         labels_sub[0][i_batch][1:labels_seq_len_sub[0][i_batch] - 1])
                #     # NOTE: Exclude <SOS> and <EOS>

            ##############################
            # Hypothesis
            ##############################
            # Convert from list of index to string
            word_list_pred = map_fn(labels_pred[i_batch])
            str_pred = ' '.join(word_list_pred)

            str_pred_sub = map_fn_sub(
                labels_pred_sub[i_batch]).replace('_', ' ')

            if model_type == 'hierarchical_attention':
                str_pred = str_pred.split('>')[0]
                str_pred_sub = str_pred_sub.split('>')[0]
                # NOTE: Trancate by the first <EOS>

                # Remove the last space
                if len(str_pred) > 0 and str_pred[-1] == ' ':
                    str_pred = str_pred[:-1]
                if len(str_pred_sub) > 0 and str_pred_sub[-1] == ' ':
                    str_pred_sub = str_pred_sub[:-1]

            print('Ref: %s' % str_true)
            print('Hyp (word): %s' % str_pred)
            print('Hyp (char): %s' % str_pred_sub)

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
