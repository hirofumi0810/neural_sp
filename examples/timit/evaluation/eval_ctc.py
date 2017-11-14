#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the trained CTC model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import yaml
import argparse

sys.path.append(abspath('../../../'))
from examples.timit.data.load_dataset_ctc import Dataset
from examples.timit.metrics.per import do_eval_per
from models.pytorch.ctc.ctc import CTC

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--beam_width', type=int, default=10,
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
    test_data = Dataset(
        data_type='test', label_type='phone39',
        batch_size=eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False)

    # GPU setting
    model.set_cuda(deterministic=False)

    # Load the saved model
    checkpoint = model.load_checkpoint(save_path=model.save_path, epoch=epoch)
    model.load_state_dict(checkpoint['state_dict'])

    # Change to evaluation mode
    model.eval()

    print('Test Data Evaluation:')
    per_test = do_eval_per(
        model=model,
        model_type='ctc',
        dataset=test_data,
        label_type=params['label_type'],
        beam_width=beam_width,
        is_test=test_data.is_test,
        eval_batch_size=eval_batch_size,
        progressbar=True)
    print('  PER: %f %%' % (per_test * 100))


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
        raise TypeError

    # Model setting
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

    model.save_path = args.model_path
    do_eval(model=model, params=params,
            epoch=args.epoch, eval_batch_size=args.eval_batch_size,
            beam_width=args.beam_width)


if __name__ == '__main__':
    main()
