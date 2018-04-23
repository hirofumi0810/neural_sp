#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the trained model (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import argparse

sys.path.append(abspath('../../../'))
from models.load_model import load
from examples.csj.s5.exp.dataset.load_dataset import Dataset
from examples.csj.s5.exp.metrics.cer import do_eval_cer
from examples.csj.s5.exp.metrics.wer import do_eval_wer
from utils.config import load_config

parser = argparse.ArgumentParser()
# parser.add_argument('--model_path', type=str,
#                     help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')
parser.add_argument('--beam_width', type=int, default=1,
                    help='beam_width (int, optional): beam width for beam search.' +
                    ' 1 disables beam search, which mean greedy decoding.')
parser.add_argument('--max_decode_len', type=int, default=150,  # or 80
                    help='the length of output sequences to stop prediction when EOS token have not been emitted')
parser.add_argument('--data_save_path', type=str, help='path to saved data')


def main():

    args = parser.parse_args()

    model_paths = [
        "/n/sd8/inaguma/result/csj/pytorch/ctc/word5/subset/blstm320H5L_drop8_fc_256_adam_lr1e-3_dropen0.2_input80_charinit",
        "/n/sd8/inaguma/result/csj/pytorch/ctc/word5/subset/blstm320H5L_drop8_fc_256_adam_lr1e-3_dropen0.2_input80_charinit_1",
        "/n/sd8/inaguma/result/csj/pytorch/ctc/word5/subset/blstm320H5L_drop8_fc_256_adam_lr1e-3_dropen0.2_input80_charinit_2",
        "/n/sd8/inaguma/result/csj/pytorch/ctc/word5/subset/blstm320H5L_drop8_fc_256_adam_lr1e-3_dropen0.2_input80_charinit_3",
    ]

    # Load a config file (.yml)
    params_list = []
    for model_path in model_paths:
        params = load_config(join(model_path, 'config.yml'), is_eval=True)
        params_list.append(params)

    # Load dataset
    eval1_data = Dataset(
        data_save_path=args.data_save_path,
        backend=params_list[0]['backend'],
        input_channel=params_list[0]['input_channel'],
        use_delta=params_list[0]['use_delta'],
        use_double_delta=params_list[0]['use_double_delta'],
        data_type='eval1', data_size=params_list[0]['data_size'],
        label_type=params_list[0]['label_type'],
        batch_size=args.eval_batch_size, splice=params_list[0]['splice'],
        num_stack=params_list[0]['num_stack'], num_skip=params_list[0]['num_skip'],
        shuffle=False, tool=params_list[0]['tool'])
    eval2_data = Dataset(
        data_save_path=args.data_save_path,
        backend=params_list[0]['backend'],
        input_channel=params_list[0]['input_channel'],
        use_delta=params_list[0]['use_delta'],
        use_double_delta=params_list[0]['use_double_delta'],
        data_type='eval2', data_size=params_list[0]['data_size'],
        label_type=params_list[0]['label_type'],
        batch_size=args.eval_batch_size, splice=params_list[0]['splice'],
        num_stack=params_list[0]['num_stack'], num_skip=params_list[0]['num_skip'],
        shuffle=False, tool=params_list[0]['tool'])
    eval3_data = Dataset(
        data_save_path=args.data_save_path,
        backend=params_list[0]['backend'],
        input_channel=params_list[0]['input_channel'],
        use_delta=params_list[0]['use_delta'],
        use_double_delta=params_list[0]['use_double_delta'],
        data_type='eval3', data_size=params_list[0]['data_size'],
        label_type=params_list[0]['label_type'],
        batch_size=args.eval_batch_size, splice=params_list[0]['splice'],
        num_stack=params_list[0]['num_stack'], num_skip=params_list[0]['num_skip'],
        shuffle=False, tool=params_list[0]['tool'])
    params_list[0]['num_classes'] = eval1_data.num_classes

    # Load model
    models = []
    for i in range(len(model_paths)):
        model = load(model_type=params_list[i]['model_type'],
                     params=params_list[0],
                     backend=params_list[0]['backend'])

        # Restore the saved parameters
        model.load_checkpoint(save_path=model_paths[i], epoch=args.epoch)

        # GPU setting
        model.set_cuda(deterministic=False, benchmark=True)
        models.append(model)

    if 'word' in params['label_type']:
        wer_eval1, df_wer_eval1 = do_eval_wer(
            models=models,
            dataset=eval1_data,
            beam_width=args.beam_width,
            max_decode_len=args.max_decode_len,
            eval_batch_size=args.eval_batch_size,
            progressbar=True)
        print('  WER (eval1): %.3f %%' % (wer_eval1 * 100))
        print(df_wer_eval1)

        wer_eval2, df_wer_eval2 = do_eval_wer(
            models=models,
            dataset=eval2_data,
            beam_width=args.beam_width,
            max_decode_len=args.max_decode_len,
            eval_batch_size=args.eval_batch_size,
            progressbar=True)
        print('  WER (eval2): %.3f %%' % (wer_eval2 * 100))
        print(df_wer_eval2)

        wer_eval3, df_wer_eval3 = do_eval_wer(
            models=models,
            dataset=eval3_data,
            beam_width=args.beam_width,
            max_decode_len=args.max_decode_len,
            eval_batch_size=args.eval_batch_size,
            progressbar=True)
        print('  WER (eval3): %.3f %%' % (wer_eval3 * 100))
        print(df_wer_eval3)

        print('  WER (mean): %.3f %%' %
              ((wer_eval1 + wer_eval2 + wer_eval3) * 100 / 3))
    else:
        cer_eval1, wer_eval1, df_cer_eval1 = do_eval_cer(
            models=models,
            dataset=eval1_data,
            beam_width=args.beam_width,
            max_decode_len=args.max_decode_len,
            eval_batch_size=args.eval_batch_size,
            progressbar=True)
        print('  CER (eval1): %.3f %%' % (cer_eval1 * 100))
        if params['label_type'] == 'kanji_wb':
            print('  WER (eval1): %.3f %%' % (wer_eval1 * 100))
        print(df_cer_eval1)

        cer_eval2, wer_eval2, df_cer_eval2 = do_eval_cer(
            models=models,
            dataset=eval2_data,
            beam_width=args.beam_width,
            max_decode_len=args.max_decode_len,
            eval_batch_size=args.eval_batch_size,
            progressbar=True)
        print('  CER (eval2): %.3f %%' % (cer_eval2 * 100))
        if params['label_type'] == 'kanji_wb':
            print('  WER (eval2): %.3f %%' % (wer_eval2 * 100))
        print(df_cer_eval2)

        cer_eval3, wer_eval3, df_cer_eval3 = do_eval_cer(
            models=models,
            dataset=eval3_data,
            beam_width=args.beam_width,
            max_decode_len=args.max_decode_len,
            eval_batch_size=args.eval_batch_size,
            progressbar=True)
        print('  CER (eval3): %.3f %%' % (cer_eval3 * 100))
        if params['label_type'] == 'kanji_wb':
            print('  WER (eval3): %.3f %%' % (wer_eval3 * 100))
        print(df_cer_eval3)

        print('  CER (mean): %.3f %%' %
              ((cer_eval1 + cer_eval2 + cer_eval3) * 100 / 3))
        if params['label_type'] == 'kanji_wb':
            print('  WER (mean): %.3f %%' %
                  ((wer_eval1 + wer_eval2 + wer_eval3) * 100 / 3))


if __name__ == '__main__':
    main()
