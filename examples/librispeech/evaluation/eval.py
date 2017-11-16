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
from models.pytorch.ctc.ctc import CTC
from models.pytorch.attention.attention_seq2seq import AttentionSeq2seq

from examples.librispeech.data.load_dataset_ctc import Dataset as Dataset_ctc
from examples.librispeech.data.load_dataset_attention import Dataset as Dataset_attention

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
            model_type='attention',
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
            model_type='attention',
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

    # Except for blank, <SOS>, <EOS> classes
    if params['label_type'] == 'kana':
        params['num_classes'] = 146
    elif params['label_type'] == 'kana_divide':
        params['num_classes'] = 147
    elif params['label_type'] == 'kanji':
        if params['data_size'] == 'subset':
            params['num_classes'] = 2978
        elif params['data_size'] == 'fullset':
            params['num_classes'] = 3383
    elif params['label_type'] == 'kanji_divide':
        if params['data_size'] == 'subset':
            params['num_classes'] = 2979
        elif params['data_size'] == 'fullset':
            params['num_classes'] = 3384
    elif params['label_type'] == 'word_freq1':
        if params['data_size'] == 'subset':
            params['num_classes'] = 39169
        elif params['data_size'] == 'fullset':
            params['num_classes'] = 66277
    elif params['label_type'] == 'word_freq5':
        if params['data_size'] == 'subset':
            params['num_classes'] = 12877
        elif params['data_size'] == 'fullset':
            params['num_classes'] = 23528
    elif params['label_type'] == 'word_freq10':
        if params['data_size'] == 'subset':
            params['num_classes'] = 8542
        elif params['data_size'] == 'fullset':
            params['num_classes'] = 15536
    elif params['label_type'] == 'word_freq15':
        if params['data_size'] == 'subset':
            params['num_classes'] = 6726
        elif params['data_size'] == 'fullset':
            params['num_classes'] = 12111
    else:
        raise TypeError

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
            max_decode_length=100,
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
