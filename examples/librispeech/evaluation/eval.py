#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the trained model (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import yaml
import argparse

sys.path.append(abspath('../../../'))
from models.pytorch.load_model import load
from examples.librispeech.data.load_dataset_ctc import Dataset as Dataset_ctc
from examples.librispeech.data.load_dataset_attention import Dataset as Dataset_attention
from examples.librispeech.metrics.cer import do_eval_cer
from examples.librispeech.metrics.wer import do_eval_wer

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


def do_eval(model, params, epoch, beam_width, eval_batch_size):
    """Evaluate the model.
    Args:
        model: the model to restore
        params (dict): A dictionary of parameters
        epoch (int): the epoch to restore
        beam_width (int): beam_width (int, optional): beam width for beam search.
            1 disables beam search, which mean greedy decoding.
        eval_batch_size (int): the size of mini-batch when evaluation
    """
    if params['label_type'] == 'character':
        vocab_file_path = '../metrics/vocab_files/character.txt'
    else:
        vocab_file_path = '../metrics/vocab_files/' + \
            params['label_type'] + '_' + params['data_size'] + '.txt'
    if params['model_type'] == 'ctc':
        Dataset = Dataset_ctc
    elif params['model_type'] == 'attention':
        Dataset = Dataset_attention
    test_clean_data = Dataset(
        data_type='test_clean', data_size=params['data_size'],
        label_type=params['label_type'], vocab_file_path=vocab_file_path,
        batch_size=params['batch_size'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False)
    test_other_data = Dataset(
        data_type='test_other', data_size=params['data_size'],
        label_type=params['label_type'], vocab_file_path=vocab_file_path,
        batch_size=params['batch_size'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False)

    # GPU setting
    model.set_cuda(deterministic=False)

    # Load the saved model
    checkpoint = model.load_checkpoint(save_path=model.save_path, epoch=epoch)
    model.load_state_dict(checkpoint['state_dict'])

    # Change to evaluation mode
    model.eval()

    print('Test Data Evaluation:')
    if 'char' in params['label_type']:
        cer_test_clean, wer_test_clean = do_eval_cer(
            model=model,
            model_type=params['model_type'],
            dataset=test_clean_data,
            label_type=params['label_type'],
            data_size=params['data_size'],
            beam_width=beam_width,
            is_test=test_clean_data.is_test,
            eval_batch_size=eval_batch_size,
            progressbar=True)
        print('  CER (test-clean): %f %%' % (cer_test_clean * 100))
        print('  WER (test-clean): %f %%' % (wer_test_clean * 100))

        cer_test_other, wer_test_other = do_eval_cer(
            model=model,
            model_type=params['model_type'],
            dataset=test_other_data,
            label_type=params['label_type'],
            data_size=params['data_size'],
            beam_width=beam_width,
            is_test=test_other_data.is_test,
            eval_batch_size=eval_batch_size,
            progressbar=True)
        print('  CER (test-other): %f %%' % (cer_test_other * 100))
        print('  WER (test-other): %f %%' % (wer_test_other * 100))

    else:
        wer_test_clean = do_eval_wer(
            model=model,
            model_type=params['model_type'],
            dataset=test_clean_data,
            label_type=params['label_type'],
            data_size=params['data_size'],
            beam_width=beam_width,
            is_test=test_clean_data.is_test,
            eval_batch_size=eval_batch_size)
        print('  WER (clean): %f %%' % (wer_test_clean * 100))

        # test-other
        wer_test_other = do_eval_wer(
            model=model,
            model_type=params['model_type'],
            dataset=test_other_data,
            label_type=params['label_type'],
            data_size=params['data_size'],
            beam_width=beam_width,
            is_test=test_other_data.is_test,
            eval_batch_size=eval_batch_size)
        print('  WER (other): %f %%' % (wer_test_other * 100))


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

    # Model setting
    model = load(model_type=params['model_type'], params=params)

    model.save_path = args.model_path
    do_eval(model=model, params=params,
            epoch=args.epoch, eval_batch_size=args.eval_batch_size,
            beam_width=args.beam_width)


if __name__ == '__main__':
    main()
