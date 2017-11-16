#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Decode the trained model's outputs (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import yaml
import argparse

sys.path.append(abspath('../../../'))
from examples.timit.data.load_dataset_ctc import Dataset as Dataset_ctc
from examples.timit.data.load_dataset_attention import Dataset as Dataset_attention
from utils.io.labels.phone import Idx2phone
from utils.io.variable import np2var
from models.pytorch.ctc.ctc import CTC
from models.pytorch.attention.attention_seq2seq import AttentionSeq2seq

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


def do_decode(model, params, epoch, beam_width, eval_batch_size):
    """Conduct decoding.
    Args:
        model: the model to restore
        params (dict): A dictionary of parameters
        epoch (int): the epoch to restore
        beam_width (int): beam width for beam search.
            1 disables beam search, which mean greedy decoding.
        eval_batch_size (int): the size of mini-batch when evaluation
    """
    # Load dataset
    if params['model_type'] == 'ctc':
        Dataset = Dataset_ctc
    elif params['model_type'] == 'attention':
        Dataset = Dataset_attention
    test_data = Dataset(
        data_type='test', label_type='phone61',
        vocab_file_path='../metrics/vocab_files/phone61.txt',
        batch_size=eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=True, reverse=True)

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
        model_type=params['model_type'],
        dataset=test_data,
        label_type=params['label_type'],
        beam_width=beam_width,
        is_test=test_data.is_test,
        save_path=None)
    # save_path=model.save_path)


def decode(model, model_type, dataset, label_type, beam_width,
           is_test=False, save_path=None):
    """Visualize label outputs.
    Args:
        model: the model to evaluate
        model_type (string): ctc or attention
        dataset: An instance of a `Dataset` class
        label_type (string): phone39 or phone48 or phone61
        beam_width: (int): the size of beam
        is_test (bool, optional):
        save_path (string): path to save decoding results
    """
    idx2phone = Idx2phone('../metrics/vocab_files/' + label_type + '.txt')

    if save_path is not None:
        sys.stdout = open(join(model.model_dir, 'decode.txt'), 'w')

    for data, is_new_epoch in dataset:

        # Create feed dictionary for next mini batch
        if model_type in ['ctc', 'attention']:
            inputs, labels, inputs_seq_len, labels_seq_len, input_names = data
        elif model_type == 'joint_ctc_attention':
            raise NotImplementedError
        else:
            raise TypeError

        # Wrap by variable
        inputs = np2var(inputs, use_cuda=model.use_cuda, volatile=True)
        inputs_seq_len = np2var(
            inputs_seq_len, use_cuda=model.use_cuda, volatile=True, dtype='int')

        batch_size = inputs[0].size(0)

        # Decode
        if model_type == 'attention':
            labels_pred, _ = model.decode_infer(
                inputs[0], inputs_seq_len[0], beam_width=beam_width)
        elif model_type == 'ctc':
            labels_pred = model.decode(
                inputs[0], inputs_seq_len[0], beam_width=beam_width)
            labels_pred -= 1
            # NOTE: index 0 is reserved for blank
        elif model_type == 'joint_ctc_attention':
            raise NotImplementedError

        for i_batch in range(batch_size):
            print('----- wav: %s -----' % input_names[0][i_batch])

            ##############################
            # Reference
            ##############################
            if is_test:
                str_true = labels[0][i_batch][0]
            else:
                # Convert from list of index to string
                if model_type in ['ctc']:
                    str_true = idx2phone(
                        labels[0][i_batch][:labels_seq_len[0][i_batch]])
                elif model_type in ['attention', 'joint_ctc_attention']:
                    str_true = idx2phone(
                        labels[0][i_batch][1:labels_seq_len[0][i_batch] - 1])
                    # NOTE: Exclude <SOS> and <EOS>

            ##############################
            # Hypothesis
            ##############################
            # Convert from list of index to string
            str_pred = idx2phone(labels_pred[i_batch])

            if model_type in ['attention', 'joint_ctc_attention']:
                str_pred = str_pred.split('>')[0]
                # NOTE: Trancate by the first <EOS>

                # Remove the last space
                if len(str_pred) > 0 and str_pred[-1] == ' ':
                    str_pred = str_pred[:-1]

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
    if params['label_type'] == 'phone61':
        params['num_classes'] = 61
    elif params['label_type'] == 'phone48':
        params['num_classes'] = 48
    elif params['label_type'] == 'phone39':
        params['num_classes'] = 39
    else:
        TypeError

    # Model setting
    if params['model_type'] == 'ctc':
        model = CTC(
            input_size=params['input_size'],
            num_stack=params['num_stack'],
            splice=params['splice'],
            encoder_type=params['encoder_type'],
            bidirectional=params['bidirectional'],
            num_units=params['num_units'],
            num_proj=params['num_proj'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            num_classes=params['num_classes'],
            parameter_init=params['parameter_init'],
            logits_temperature=params['logits_temperature'])

    elif params['model_type'] == 'attention':
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
            max_decode_length=params['max_decode_length'],
            parameter_init=params['parameter_init'],
            downsample_list=[],
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
