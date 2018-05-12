#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Decode the hierarchical model's outputs (Switchboard corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import argparse

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.swbd.s5c.exp.dataset.load_dataset_hierarchical import Dataset
from examples.swbd.s5c.exp.metrics.glm import GLM
from examples.swbd.s5c.exp.metrics.post_processing import fix_trans
from utils.io.labels.character import Idx2char
from utils.io.labels.word import Idx2word
from utils.config import load_config
from utils.evaluation.edit_distance import compute_wer
from utils.evaluation.resolving_unk import resolve_unk

parser = argparse.ArgumentParser()
parser.add_argument('--data_save_path', type=str,
                    help='path to saved data')
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--beam_width', type=int, default=1,
                    help='the size of beam in the main task')
parser.add_argument('--beam_width_sub', type=int, default=1,
                    help='the size of beam in the sub task')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')

MAX_DECODE_LEN_WORD = 100
MAX_DECODE_LEN_CHAR = 300


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
        # data_type='eval2000_swbd',
        data_type='eval2000_ch',
        data_size=params['data_size'],
        label_type=params['label_type'], label_type_sub=params['label_type_sub'],
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        # sort_utt=True, reverse=True,
        sort_utt=False, reverse=False, tool=params['tool'])

    params['num_classes'] = test_data.num_classes
    params['num_classes_sub'] = test_data.num_classes_sub

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
           beam_width_sub=args.beam_width_sub,
           eval_batch_size=args.eval_batch_size,
           save_path=None
           # save_path=args.model_path
           resolving_unk=False)


def decode(model, dataset, beam_width, beam_width_sub,
           eval_batch_size=None, save_path=None, resolving_unk=False):
    """Visualize label outputs.
    Args:
        model: the model to evaluate
        dataset: An instance of a `Dataset` class
        beam_width: (int): the size of beam in the main task
        beam_width: (int): the size of beam in the sub task
        eval_batch_size (int, optional): the batch size when evaluating the model
        save_path (string): path to save decoding results
        resolving_unk (bool, optional):
    """
    # Set batch size in the evaluation
    if eval_batch_size is not None:
        dataset.batch_size = eval_batch_size

    idx2word = Idx2word(vocab_file_path=dataset.vocab_file_path)
    idx2char = Idx2char(vocab_file_path=dataset.vocab_file_path_sub,
                        capital_divide=dataset.label_type_sub == 'character_capital_divide')

    # Read GLM file
    glm = GLM(
        glm_path='/n/sd8/inaguma/corpus/swbd/data/eval2000/LDC2002T43/reference/en20000405_hub5.glm')

    if save_path is not None:
        sys.stdout = open(join(model.model_dir, 'decode.txt'), 'w')

    for batch, is_new_epoch in dataset:

        # Decode
        if model.model_type == 'nested_attention':
            best_hyps, aw, best_hyps_sub, aw_sub, perm_idx = model.decode(
                batch['xs'], batch['x_lens'],
                beam_width=beam_width,
                beam_width_sub=beam_width_sub,
                max_decode_len=MAX_DECODE_LEN_WORD,
                max_decode_len_sub=MAX_DECODE_LEN_CHAR)
        else:
            best_hyps, aw, perm_idx = model.decode(
                batch['xs'], batch['x_lens'],
                beam_width=beam_width,
                max_decode_len=MAX_DECODE_LEN_WORD)
            best_hyps_sub, aw_sub, _ = model.decode(
                batch['xs'], batch['x_lens'],
                beam_width=beam_width_sub,
                max_decode_len=MAX_DECODE_LEN_CHAR,
                task_index=1)

        ys = batch['ys'][perm_idx]
        y_lens = batch['y_lens'][perm_idx]
        ys_sub = batch['ys_sub'][perm_idx]
        y_lens_sub = batch['y_lens_sub'][perm_idx]

        for b in range(len(batch['xs'])):

            ##############################
            # Reference
            ##############################
            if dataset.is_test:
                str_ref_original = ys[b][0]
                str_ref_sub = ys_sub[b][0]
                # NOTE: transcript is seperated by space('_')
            else:
                # Convert from list of index to string
                str_ref_original = idx2word(ys[b][: y_lens[b]])
                str_ref_sub = idx2word(ys_sub[b][:y_lens_sub[b]])

            ##############################
            # Hypothesis
            ##############################
            # Convert from list of index to string
            str_hyp = idx2word(best_hyps[b])
            str_hyp_sub = idx2char(best_hyps_sub[b])

            ##############################
            # Resolving UNK
            ##############################
            if 'OOV' in str_hyp and resolving_unk:
                str_hyp_no_unk = resolve_unk(
                    str_hyp, best_hyps_sub[b], aw[b], aw_sub[b], idx2char)

            # if 'OOV' not in str_hyp:
            #     continue

            ##############################
            # Post-proccessing
            ##############################
            str_ref = fix_trans(str_ref_original, glm)
            str_ref_sub = fix_trans(str_ref_sub, glm)
            str_hyp = fix_trans(str_hyp, glm)
            str_hyp_sub = fix_trans(str_hyp_sub, glm)
            str_hyp_no_unk = fix_trans(str_hyp_no_unk, glm)

            if len(str_ref) == 0:
                continue

            print('----- wav: %s -----' % batch['input_names'][b])
            print('Ref         : %s' % str_ref.replace('_', ' '))
            print('Hyp (main)  : %s' % str_hyp.replace('_', ' '))
            print('Hyp (sub)   : %s' % str_hyp_sub.replace('_', ' '))
            if 'OOV' in str_hyp and resolving_unk:
                print('Hyp (no UNK): %s' % str_hyp_no_unk.replace('_', ' '))

            try:
                # Compute WER
                wer, _, _, _ = compute_wer(
                    ref=str_ref.split('_'),
                    hyp=str_hyp.replace(r'_>.*', '').split('_'),
                    normalize=True)
                print('WER (main)  : %.3f %%' % (wer * 100))
                wer_sub, _, _, _ = compute_wer(
                    ref=str_ref_sub.split('_'),
                    hyp=str_hyp_sub.replace(r'>.*', '').split('_'),
                    normalize=True)
                print('WER (sub)   : %.3f %%' % (wer_sub * 100))
                if 'OOV' in str_hyp and resolving_unk:
                    wer_no_unk, _, _, _ = compute_wer(
                        ref=str_ref.split('_'),
                        hyp=str_hyp_no_unk.replace(
                            '*', '').replace(r'_>.*', '').split('_'),
                        normalize=True)
                    print('WER (no UNK): %.3f %%' % (wer_no_unk * 100))
            except:
                print('--- skipped ---')

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
