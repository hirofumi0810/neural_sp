#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the trained Attention-based model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import yaml
import argparse

sys.path.append(abspath('../../../'))
from examples.timit.data.load_dataset_attention import Dataset
from examples.timit.metrics.attention import do_eval_per
from models.pytorch.attention.attention_seq2seq import AttentionSeq2seq

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--beam_width', type=int, default=10,
                    help='beam_width (int, optional): beam width for beam search.' +
                    ' 1 disables beam search, which mean greedy decoding.')
parser.add_argument('--eval_batch_size', type=str, default=1,
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
        batch_size=eval_batch_size,
        map_file_path='../metrics/mapping_files/phone39.txt',
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
    per_test = do_eval_per(
        model=model,
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
        max_decode_length=40,
        parameter_init=params['parameter_init'],
        downsample_list=[],
        init_dec_state_with_enc_state=True,
        sharpening_factor=params['sharpening_factor'],
        logits_temperature=params['logits_temperature'],
        sigmoid_smoothing=params['sigmoid_smoothing'],
        input_feeding_approach=params['input_feeding_approach'])

    model.save_path = args.model_path
    do_eval(model=model, params=params,
            epoch=args.epoch, eval_batch_size=args.eval_batch_size,
            beam_width=args.beam_width)


if __name__ == '__main__':
    main()
