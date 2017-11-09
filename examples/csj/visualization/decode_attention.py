#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Decode the trained Attention outputs (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import yaml
import argparse

sys.path.append(abspath('../../../'))
from examples.csj.data.load_dataset_attention import Dataset
from utils.io.labels.character import Idx2char
from utils.io.variable import np2var
from models.pytorch.attention.attention_seq2seq import AttentionSeq2seq

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--beam_width', type=int, default=1,
                    help='beam_width (int, optional): beam width for beam search.' +
                    ' 1 disables beam search, which mean greedy decoding.')
parser.add_argument('--eval_batch_size', type=str, default=1,
                    help='the size of mini-batch in evaluation')


def do_decode(model, params, epoch, beam_width, eval_batch_size):
    """Decode the Attention outputs.
    Args:
        model: the model to restore
        params (dict): A dictionary of parameters
        epoch (int): the epoch to restore
        beam_width (int): beam width for beam search.
            1 disables beam search, which mean greedy decoding.
        eval_batch_size (int): the size of mini-batch when evaluation
    """
    if 'kanji' in params['label_type']:
        map_file_path = '../metrics/mapping_files/' + \
            params['label_type'] + '_' + params['train_data_size'] + '.txt'
    elif 'kana' in params['label_type']:
        map_file_path = '../metrics/mapping_files/' + \
            params['label_type'] + '.txt'

    # Load dataset
    test_data = Dataset(
        data_type='dev', label_type=params['label_type'],
        train_data_size=params['train_data_size'],
        batch_size=eval_batch_size,
        map_file_path=map_file_path,
        splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=True, reverse=True, progressbar=True)

    # GPU setting
    model.set_cuda(deterministic=False)

    # Load the saved model
    checkpoint = model.load_checkpoint(save_path=model.save_path, epoch=epoch)
    model.load_state_dict(checkpoint['state_dict'])

    # Change to evaluation mode
    model.eval()

    # Visualize
    decode(
        model=model,
        dataset=test_data,
        label_type=params['label_type'],
        train_data_size=params['train_data_size'],
        beam_width=beam_width,
        is_test=test_data.is_test,
        save_path=None)
    # save_path=model.save_path)


def decode(model, dataset, label_type, train_data_size, beam_width,
           is_test=False, save_path=None):
    """Visualize label outputs of Attention-based model.
    Args:
        model: the model to evaluate
        dataset: An instance of a `Dataset` class
        label_type (string): kanji or kanji or kanji_divide or kana_divide
        train_data_size (string): train_fullset or train_subset
        beam_width: (int): the size of beam
        is_test (bool, optional):
        save_path (string): path to save decoding results
    """
    if 'kanji' in label_type:
        map_file_path = '../metrics/mapping_files/' + \
            label_type + '_' + train_data_size + '.txt'
    elif 'kana' in label_type:
        map_file_path = '../metrics/mapping_files/' + label_type + '.txt'

    idx2char = Idx2char(map_file_path=map_file_path)

    if save_path is not None:
        sys.stdout = open(join(model.model_dir, 'decode.txt'), 'w')

    for data, is_new_epoch in dataset:

        # Create feed dictionary for next mini batch
        inputs, labels_true, _, labels_seq_len, input_names = data
        inputs = np2var(
            inputs, use_cuda=model.use_cuda, volatile=True)

        batch_size = inputs[0].size(0)

        # Decode
        labels_pred, _ = model.decode_infer(
            inputs[0], beam_width=beam_width)

        for i_batch in range(batch_size):

            print('----- wav: %s -----' % input_names[0][i_batch])

            # Convert from list of index to string
            if is_test:
                str_true = labels_true[0][i_batch][0]
                # NOTE: transcript is seperated by space('_')
            else:
                str_true = idx2char(
                    labels_true[0][i_batch][1:labels_seq_len[0][i_batch] - 1])
            str_pred = idx2char(labels_pred[i_batch]).split('>')[0]
            # NOTE: Trancate by <EOS>

            print('Ref: %s' % str_true)
            print('Hyp: %s' % str_pred)

        if is_new_epoch:
            break


def main():

    args = parser.parse_args()

    # Load config file
    with open(join(args.model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Except for a <SOS> and <EOS> class
    if params['label_type'] == 'kana':
        params['num_classes'] = 146
    elif params['label_type'] == 'kana_divide':
        params['num_classes'] = 147
    elif params['label_type'] == 'kanji':
        if params['train_data_size'] == 'train_subset':
            params['num_classes'] = 2981
        elif params['train_data_size'] == 'train_fullset':
            params['num_classes'] = 3385
    elif params['label_type'] == 'kanji_divide':
        if params['train_data_size'] == 'train_subset':
            params['num_classes'] = 2982
        elif params['train_data_size'] == 'train_fullset':
            params['num_classes'] = 3386
    else:
        raise TypeError

    downsample_list = [False] * params['encoder_num_layers']
    downsample_list[1] = True
    downsample_list[2] = True

    # Model setting
    model = AttentionSeq2seq(
        input_size=params['input_size'],
        num_stack=params['num_stack'],
        splice=params['splice'],
        encoder_type=params['encoder_type'],
        encoder_bidirectional=params['encoder_bidirectional'],
        encoder_num_units=params['encoder_num_units'],
        encoder_num_proj=params['encoder_num_proj'],
        encoder_num_layers=params['encoder_num_layers'],
        encoder_dropout=params['dropout_encoder'],
        attention_type=params['attention_type'],
        attention_dim=params['attention_dim'],
        decoder_type=params['decoder_type'],
        decoder_num_units=params['decoder_num_units'],
        decoder_num_proj=params['decoder_num_proj'],
        decdoder_num_layers=params['decoder_num_layers'],
        decoder_dropout=params['dropout_decoder'],
        embedding_dim=params['embedding_dim'],
        embedding_dropout=params['dropout_embedding'],
        num_classes=params['num_classes'],
        sos_index=params['num_classes'],
        eos_index=params['num_classes'] + 1,
        max_decode_length=params['max_decode_length'],
        parameter_init=params['parameter_init'],
        downsample_list=downsample_list,
        init_dec_state_with_enc_state=True,
        sharpening_factor=params['sharpening_factor'],
        logits_temperature=params['logits_temperature'],
        sigmoid_smoothing=params['sigmoid_smoothing'],
        input_feeding_approach=params['input_feeding_approach'])

    model.save_path = args.model_path
    do_decode(model=model, params=params,
              epoch=args.epoch, eval_batch_size=args.eval_batch_size,
              beam_width=args.beam_width)


if __name__ == '__main__':
    main()
