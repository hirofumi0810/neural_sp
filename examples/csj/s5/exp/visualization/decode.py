#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Decode the model's outputs (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import argparse

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.csj.s5.exp.dataset.load_dataset import Dataset
from utils.io.labels.character import Idx2char
from utils.io.labels.word import Idx2word
from utils.config import load_config
from utils.evaluation.edit_distance import compute_wer

parser = argparse.ArgumentParser()
parser.add_argument('--data_save_path', type=str,
                    help='path to saved data')
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--beam_width', type=int, default=1,
                    help='the size of beam')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')


MAX_DECODE_LEN_WORD = 100
MAX_DECODE_LEN_CHAR = 200


def main():

    args = parser.parse_args()

    # Load a config file (.yml)
    params = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Load dataset
    test_data = Dataset(
        data_save_path=args.data_save_path,
        backend=params['backend'],
        input_freq=params['input_freq'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        data_type='eval1',
        # data_type='eval2',
        # data_type='eval3',
        data_size=params['data_size'],
        label_type=params['label_type'],
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False, reverse=False, tool=params['tool'])

    params['num_classes'] = test_data.num_classes
    params['num_classes_sub'] = test_data.num_classes

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
           dataset=test_data,
           beam_width=args.beam_width,
           eval_batch_size=args.eval_batch_size,
           save_path=None)
    # save_path=args.model_path)


def decode(model, dataset, beam_width,
           eval_batch_size=None, save_path=None):
    """Visualize label outputs.
    Args:
        model: the model to evaluate
        dataset: An instance of a `Dataset` class
        beam_width: (int): the size of beam
        eval_batch_size (int, optional): the batch size when evaluating the model
        save_path (string): path to save decoding results
    """
    # Set batch size in the evaluation
    if eval_batch_size is not None:
        dataset.batch_size = eval_batch_size

    if 'word' in dataset.label_type:
        map_fn = Idx2word(dataset.vocab_file_path)
        max_decode_len = MAX_DECODE_LEN_WORD
    else:
        map_fn = Idx2char(dataset.vocab_file_path)
        max_decode_len = MAX_DECODE_LEN_CHAR

    if save_path is not None:
        sys.stdout = open(join(model.model_dir, 'decode.txt'), 'w')

    for batch, is_new_epoch in dataset:

        # Decode
        if model.model_type == 'nested_attention':
            best_hyps, _, best_hyps_sub, _, perm_idx = model.decode(
                batch['xs'], batch['x_lens'],
                beam_width=beam_width,
                max_decode_len=max_decode_len,
                max_decode_len_sub=max_decode_len)
        else:
            best_hyps, _, perm_idx = model.decode(
                batch['xs'], batch['x_lens'],
                beam_width=beam_width,
                max_decode_len=max_decode_len)

        if model.model_type == 'attention' and model.ctc_loss_weight > 0:
            best_hyps_ctc, perm_idx = model.decode_ctc(
                batch['xs'], batch['x_lens'], beam_width=beam_width)

        ys = batch['ys'][perm_idx]
        y_lens = batch['y_lens'][perm_idx]

        for b in range(len(batch['xs'])):
            ##############################
            # Reference
            ##############################
            if dataset.is_test:
                str_ref = ys[b][0]
                # NOTE: transcript is seperated by space('_')
            else:
                # Convert from list of index to string
                str_ref = map_fn(ys[b][:y_lens[b]])

            ##############################
            # Hypothesis
            ##############################
            # Convert from list of index to string
            str_hyp = map_fn(best_hyps[b])

            print('----- wav: %s -----' % batch['input_names'][b])
            print('Ref: %s' % str_ref.replace('_', ' '))
            print('Hyp: %s' % str_hyp.replace('_', ' '))
            if model.model_type == 'attention' and model.ctc_loss_weight > 0:
                str_hyp_ctc = map_fn(best_hyps_ctc[b])
                print('Hyp (CTC): %s' % str_hyp_ctc)

            try:
                if 'word' in dataset.label_type:
                    wer, _, _, _ = compute_wer(
                        ref=str_ref.split('_'),
                        hyp=str_hyp.replace('_>', '').split('_'),
                        normalize=True)
                    print('WER: %.3f %%' % (wer * 100))
                    if model.ctc_loss_weight > 0:
                        wer_ctc, _, _, _ = compute_wer(
                            ref=str_ref.split('_'),
                            hyp=str_hyp_ctc.split('_'),
                            normalize=True)
                        print('WER (CTC): %.3f %%' % (wer_ctc * 100))
                else:
                    cer, _, _, _ = compute_wer(
                        ref=list(str_ref.replace('_', '')),
                        hyp=list(str_hyp.replace('>', '').replace('_', '')),
                        normalize=True)
                    print('CER: %.3f %%' % (cer * 100))
                    if model.model_type == 'attention' and model.ctc_loss_weight > 0:
                        cer_ctc, _, _, _ = compute_wer(
                            ref=list(str_ref.replace('_', '')),
                            hyp=list(str_hyp_ctc.replace('_', '')),
                            normalize=True)
                        print('CER (CTC): %.3f %%' % (cer_ctc * 100))
            except:
                print('--- skipped ---')

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
