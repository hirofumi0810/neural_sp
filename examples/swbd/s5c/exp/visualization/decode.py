#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Decode the model's outputs (Switchboard corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import argparse

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.swbd.data.load_dataset import Dataset
from utils.io.labels.character import Idx2char
from utils.io.labels.word import Idx2word
from examples.swbd.metrics.glm import GLM
from examples.swbd.metrics.post_processing import fix_trans
from utils.config import load_config
from utils.evaluation.edit_distance import compute_wer

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
parser.add_argument('--max_decode_len', type=int, default=300,  # or 100
                    help='the length of output sequences to stop prediction when EOS token have not been emitted')

LAUGHTER = 'LA'
NOISE = 'NZ'
VOCALIZED_NOISE = 'VN'
HESITATION = '%hesitation'


def main():

    args = parser.parse_args()

    # Load a config file (.yml)
    params = load_config(join(args.model_path, 'config.yml'), is_eval=True)

    # Load dataset
    vocab_file_path = '../metrics/vocab_files/' + \
        params['label_type'] + '_' + params['data_size'] + '.txt'
    test_data = Dataset(
        backend=params['backend'],
        input_channel=params['input_channel'],
        use_delta=params['use_delta'],
        use_double_delta=params['use_double_delta'],
        # data_type='eval2000_swbd',
        data_type='eval2000_ch',
        data_size=params['data_size'],
        label_type=params['label_type'], vocab_file_path=vocab_file_path,
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=True, reverse=True, save_format=params['save_format'])
    params['num_classes'] = test_data.num_classes

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
           eval_batch_size=args.eval_batch_size,
           save_path=None)
    # save_path=args.model_path)


def decode(model, dataset, beam_width,
           max_decode_len, eval_batch_size=None, save_path=None):
    """Visualize label outputs.
    Args:
        model: the model to evaluate
        dataset: An instance of a `Dataset` class
        beam_width: (int): the size of beam
        max_decode_len (int): the length of output sequences
            to stop prediction when EOS token have not been emitted.
            This is used for seq2seq models.
        eval_batch_size (int, optional): the batch size when evaluating the model
        save_path (string): path to save decoding results
    """
    # Set batch size in the evaluation
    if eval_batch_size is not None:
        dataset.batch_size = eval_batch_size

    vocab_file_path = '../metrics/vocab_files/' + \
        dataset.label_type + '_' + dataset.data_size + '.txt'
    if dataset.label_type == 'character':
        map_fn = Idx2char(vocab_file_path)
    elif dataset.label_type == 'character_capital_divide':
        map_fn = Idx2char(vocab_file_path, capital_divide=True)
    else:
        map_fn = Idx2word(vocab_file_path)

    # Read GLM file
    glm = GLM(
        glm_path='/n/sd8/inaguma/corpus/swbd/data/eval2000/LDC2002T43/reference/en20000405_hub5.glm')

    if save_path is not None:
        sys.stdout = open(join(model.model_dir, 'decode.txt'), 'w')

    for batch, is_new_epoch in dataset:

        # Decode
        best_hyps, perm_idx = model.decode(batch['xs'], batch['x_lens'],
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
                str_ref_original = ys[b][0]
                # NOTE: transcript is seperated by space('_')
            else:
                # Convert from list of index to string
                str_ref_original = map_fn(ys[b][:y_lens[b]])

            ##############################
            # Hypothesis
            ##############################
            # Convert from list of index to string
            str_hyp = map_fn(best_hyps[b])

            if model.model_type == 'attention':
                str_hyp = str_hyp.split('>')[0]
                # NOTE: Trancate by the first <EOS>

                # Remove the last space
                if len(str_hyp) > 0 and str_hyp[-1] == '_':
                    str_hyp = str_hyp[:-1]

            ##############################
            # Post-proccessing
            ##############################
            str_ref = fix_trans(str_ref_original, glm)
            str_hyp = fix_trans(str_hyp, glm)

            if len(str_ref) == 0:
                continue

            print('----- wav: %s -----' % batch['input_names'][b])
            print('Ref: %s' % str_ref.replace('_', ' '))
            print('Hyp: %s' % str_hyp.replace('_', ' '))
            if model.model_type == 'attention' and model.ctc_loss_weight > 0:
                str_hyp_ctc = map_fn(best_hyps_ctc[b])
                print('Hyp (CTC): %s' % str_hyp_ctc)

            # Compute CER
            if 'word' in dataset.label_type:
                wer, _, _, _ = compute_wer(ref=str_ref.split('_'),
                                           hyp=str_hyp.split('_'),
                                           normalize=True)
                print('WER: %.3f %%' % (wer * 100))
                if model.ctc_loss_weight > 0:
                    wer_ctc, _, _, _ = compute_wer(ref=str_ref.split('_'),
                                                   hyp=str_hyp_ctc.split('_'),
                                                   normalize=True)
                    print('WER (CTC): %.3f %%' % (wer_ctc * 100))
            else:
                cer, _, _, _ = compute_wer(ref=str_ref.split('_'),
                                           hyp=str_hyp.split('_'),
                                           normalize=True)
                print('CER: %.3f %%' % (cer * 100))
                if model.model_type == 'attention' and model.ctc_loss_weight > 0:
                    cer_ctc, _, _, _ = compute_wer(
                        ref=list(str_ref.replace('_', '')),
                        hyp=list(str_hyp.replace('_', '')),
                        normalize=True)
                    print('CER (CTC): %.3f %%' % (cer_ctc * 100))

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()
