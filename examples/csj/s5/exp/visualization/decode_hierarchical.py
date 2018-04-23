#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Decode the hierarchical model's outputs (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import argparse

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.csj.s5.exp.dataset.load_dataset_hierarchical import Dataset
from utils.io.labels.character import Idx2char
from utils.io.labels.word import Idx2word
from utils.config import load_config
from utils.evaluation.edit_distance import compute_wer
from utils.evaluation.resolving_unk import resolve_unk

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')
parser.add_argument('--beam_width', type=int, default=1,
                    help='beam_width (int, optional): beam width for beam search.' +
                    ' 1 disables beam search, which mean greedy decoding.')
parser.add_argument('--max_decode_len', type=int, default=80,
                    help='the length of output sequences to stop prediction when EOS token have not been emitted')
parser.add_argument('--max_decode_len_sub', type=int, default=150,
                    help='the length of output sequences to stop prediction when EOS token have not been emitted')
parser.add_argument('--data_save_path', type=str, help='path to saved data')


def main():

    args = parser.parse_args()

    # Load a config file (.yml)
    params = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Load dataset
    test_data = Dataset(
        data_save_path=args.data_save_path,
        backend=params['backend'],
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        data_type='eval1',
        # data_type='eval2',
        # data_type='eval3',
        data_size=params['data_size'],
        label_type=params['label_type'], label_type_sub=params['label_type_sub'],
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
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
           max_decode_len=args.max_decode_len,
           max_decode_len_sub=args.max_decode_len_sub,
           eval_batch_size=args.eval_batch_size,
           save_path=None,
           # save_path=args.model_path,
           resolving_unk=False)


def decode(model, dataset, beam_width, max_decode_len, max_decode_len_sub,
           eval_batch_size=None, save_path=None, resolving_unk=False):
    """Visualize label outputs.
    Args:
        model: the model to evaluate
        dataset: An instance of a `Dataset` class
        beam_width: (int): the size of beam
        max_decode_len (int): the length of output sequences
            to stop prediction when EOS token have not been emitted.
            This is used for seq2seq models.
        max_decode_len_sub (int):
        eval_batch_size (int, optional): the batch size when evaluating the model
        save_path (string): path to save decoding results
        resolving_unk (bool, optional):
    """
    # Set batch size in the evaluation
    if eval_batch_size is not None:
        dataset.batch_size = eval_batch_size

    idx2word = Idx2word(dataset.vocab_file_path)
    idx2char = Idx2char(dataset.vocab_file_path_sub)
    # idx2char = Idx2word(dataset.vocab_file_path_sub)

    if save_path is not None:
        sys.stdout = open(join(model.model_dir, 'decode.txt'), 'w')

    for batch, is_new_epoch in dataset:

        # Decode
        if model.model_type == 'nested_attention':
            if resolving_unk:
                best_hyps, aw, best_hyps_sub, aw_sub, perm_idx = model.decode(
                    batch['xs'], batch['x_lens'],
                    beam_width=beam_width,
                    max_decode_len=max_decode_len,
                    max_decode_len_sub=100,
                    resolving_unk=True)
            else:
                best_hyps, best_hyps_sub, perm_idx = model.decode(
                    batch['xs'], batch['x_lens'],
                    beam_width=beam_width,
                    max_decode_len=max_decode_len,
                    max_decode_len_sub=100,
                    resolving_unk=False)
        else:
            if resolving_unk:
                best_hyps, aw, perm_idx = model.decode(
                    batch['xs'], batch['x_lens'],
                    beam_width=beam_width,
                    max_decode_len=max_decode_len,
                    task_index=0, resolving_unk=True)
                best_hyps_sub, aw_sub, perm_idx = model.decode(
                    batch['xs'], batch['x_lens'],
                    beam_width=beam_width,
                    max_decode_len=max_decode_len_sub,
                    task_index=1, resolving_unk=True)
            else:
                best_hyps, perm_idx = model.decode(
                    batch['xs'], batch['x_lens'],
                    beam_width=beam_width,
                    max_decode_len=max_decode_len,
                    task_index=0, resolving_unk=False)
                best_hyps_sub, perm_idx = model.decode(
                    batch['xs'], batch['x_lens'],
                    beam_width=beam_width,
                    max_decode_len=max_decode_len_sub,
                    task_index=1, resolving_unk=False)

        ys = batch['ys'][perm_idx]
        y_lens = batch['y_lens'][perm_idx]
        ys_sub = batch['ys_sub'][perm_idx]
        y_lens_sub = batch['y_lens_sub'][perm_idx]

        for b in range(len(batch['xs'])):

            ##############################
            # Reference
            ##############################
            if dataset.is_test:
                str_ref = ys[b][0]
                str_ref_sub = ys_sub[b][0]
                # NOTE: transcript is seperated by space('_')
            else:
                # Convert from list of index to string
                str_ref = idx2word(ys[b][: y_lens[b]])
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
            if resolving_unk:
                if 'OOV' in str_hyp:
                    str_hyp_no_unk = resolve_unk(
                        str_hyp, best_hyps_sub[b], aw[b], aw_sub[b], idx2char)
                else:
                    str_hyp_no_unk = str_hyp
            else:
                str_hyp_no_unk = str_hyp

            if model.model_type != 'hierarchical_ctc':
                str_hyp = str_hyp.split('>')[0]
                str_hyp_sub = str_hyp_sub.split('>')[0]
                str_hyp_no_unk = str_hyp_no_unk.split('>')[0]
                # NOTE: Trancate by the first <EOS>

                # Remove the last space
                if len(str_hyp) > 0 and str_hyp[-1] == '_':
                    str_hyp = str_hyp[:-1]
                if len(str_hyp_sub) > 0 and str_hyp_sub[-1] == '_':
                    str_hyp_sub = str_hyp_sub[:-1]
                if len(str_hyp_no_unk) > 0 and str_hyp_no_unk[-1] == '_':
                    str_hyp_no_unk = str_hyp_no_unk[:-1]

            if 'OOV' not in str_hyp:
                continue

            print('----- wav: %s -----' % batch['input_names'][b])
            print('Ref         : %s' % str_ref.replace('_', ' '))
            print('Hyp (main)  : %s' % str_hyp.replace('_', ' '))
            # print('Ref (sub) : %s' % str_ref_sub.replace('_', ' '))
            print('Hyp (sub)   : %s' % str_hyp_sub.replace('_', ' '))
            print('Hyp (no UNK): %s' % str_hyp_no_unk.replace('_', ' '))

            try:
                wer, _, _, _ = compute_wer(ref=str_ref.split('_'),
                                           hyp=str_hyp.split('_'),
                                           normalize=True)
                print('WER (main)  : %.3f %%' % (wer * 100))
                cer, _, _, _ = compute_wer(ref=list(str_ref_sub.replace('_', '')),
                                           hyp=list(
                                               str_hyp_sub.replace('_', '')),
                                           normalize=True)
                print('CER (sub)   : %.3f %%' % (cer * 100))
                wer_no_unk, _, _, _ = compute_wer(ref=str_ref.split('_'),
                                                  hyp=str_hyp_no_unk.replace(
                                                      '*', '').split('_'),
                                                  normalize=True)
                print('WER (no UNK): %.3f %%' % (wer_no_unk * 100))
            except:
                pass

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
