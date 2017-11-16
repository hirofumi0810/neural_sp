#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the trained model (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import yaml
import argparse

sys.path.append(abspath('../../../'))
from models.pytorch.load_model import load
from examples.csj.data.load_dataset_ctc import Dataset as Dataset_ctc
from examples.csj.data.load_dataset_attention import Dataset as Dataset_attention
from examples.csj.metrics.cer import do_eval_cer
from examples.csj.metrics.wer import do_eval_wer

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
    # Load dataset
    if params['model_type'] == 'ctc':
        Dataset = Dataset_ctc
    elif params['model_type'] == 'attention':
        Dataset = Dataset_attention
    eval1_data = Dataset(
        data_type='eval1', label_type=params['label_type'],
        data_size=params['data_size'],
        batch_size=eval_batch_size, num_classes=params['num_classes'],
        splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False, progressbar=True)
    eval2_data = Dataset(
        data_type='eval2', label_type=params['label_type'],
        data_size=params['data_size'],
        batch_size=eval_batch_size, num_classes=params['num_classes'],
        splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False, progressbar=True)
    eval3_data = Dataset(
        data_type='eval3', label_type=params['label_type'],
        data_size=params['data_size'],
        batch_size=eval_batch_size, num_classes=params['num_classes'],
        splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False, progressbar=True)

    # GPU setting
    model.set_cuda(deterministic=False)

    # Load the saved model
    checkpoint = model.load_checkpoint(save_path=model.save_path, epoch=epoch)
    model.load_state_dict(checkpoint['state_dict'])

    # Change to evaluation mode
    model.eval()

    print('Test Data Evaluation:')
    cer_eval1 = do_eval_cer(
        model=model,
        model_type='attention',
        dataset=eval1_data,
        label_type=params['label_type'],
        data_size=params['data_size'],
        beam_width=beam_width,
        is_test=True,
        eval_batch_size=eval_batch_size,
        progressbar=True)
    print('  CER (eval1): %f %%' % (cer_eval1 * 100))

    cer_eval2 = do_eval_cer(
        model=model,
        model_type='attention',
        dataset=eval2_data,
        label_type=params['label_type'],
        data_size=params['data_size'],
        beam_width=beam_width,
        is_test=True,
        eval_batch_size=eval_batch_size,
        progressbar=True)
    print('  CER (eval2): %f %%' % (cer_eval2 * 100))

    cer_eval3 = do_eval_cer(
        model=model,
        model_type='attention',
        dataset=eval3_data,
        label_type=params['label_type'],
        data_size=params['data_size'],
        beam_width=beam_width,
        is_test=True,
        eval_batch_size=eval_batch_size,
        progressbar=True)
    print('  CER (eval3): %f %%' % (cer_eval3 * 100))

    print('  CER (mean): %f %%' %
          ((cer_eval1 + cer_eval2 + cer_eval3) * 100 / 3))


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
