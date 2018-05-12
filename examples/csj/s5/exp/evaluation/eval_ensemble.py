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
from examples.csj.s5.exp.metrics.character import eval_char
from examples.csj.s5.exp.metrics.word import eval_word
from utils.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument('--data_save_path', type=str,
                    help='path to saved data')
# parser.add_argument('--model_path', type=str,
#                     help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--beam_width', type=int, default=1,
                    help='the size of beam in the main task')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')


MAX_DECODE_LEN_WORD = 100
MAX_DECODE_LEN_CHAR = 200


def main():

    args = parser.parse_args()

    # model_paths = [
    #     "/n/sd8/inaguma/result/csj/pytorch/ctc/word5/subset/blstm320H5L_drop8_fc_256_adam_lr1e-3_dropen0.2_input80_charinit",
    #     "/n/sd8/inaguma/result/csj/pytorch/ctc/word5/subset/blstm320H5L_drop8_fc_256_adam_lr1e-3_dropen0.2_input80_charinit_1",
    #     "/n/sd8/inaguma/result/csj/pytorch/ctc/word5/subset/blstm320H5L_drop8_fc_256_adam_lr1e-3_dropen0.2_input80_charinit_2",
    #     "/n/sd8/inaguma/result/csj/pytorch/ctc/word5/subset/blstm320H5L_drop8_fc_256_adam_lr1e-3_dropen0.2_input80_charinit_3",
    # ]

    # model_paths = [
    #     "/n/sd8/inaguma/result/csj/pytorch/ctc/kanji_wb/subset/blstm320H5L_drop4_fc_256_adam_lr1e-3_dropen0.2_input240",
    #     "/n/sd8/inaguma/result/csj/pytorch/ctc/kanji_wb/subset/blstm320H5L_drop4_fc_256_adam_lr1e-3_dropen0.2_input240_1",
    #     "/n/sd8/inaguma/result/csj/pytorch/ctc/kanji_wb/subset/blstm320H5L_drop4_fc_256_adam_lr1e-3_dropen0.2_input240_2",
    #     "/n/sd8/inaguma/result/csj/pytorch/ctc/kanji_wb/subset/blstm320H5L_drop4_fc_256_adam_lr1e-3_dropen0.2_input240_3",
    # ]
    # model_paths = [
    #     "/n/sd8/inaguma/result/csj/pytorch/ctc/kanji_wb/subset/blstm320H5L_drop4_fc_256_adam_lr1e-3_dropen0.2_temp2_input240",
    #     "/n/sd8/inaguma/result/csj/pytorch/ctc/kanji_wb/subset/blstm320H5L_drop4_fc_256_adam_lr1e-3_dropen0.2_temp2_input240_1",
    #     "/n/sd8/inaguma/result/csj/pytorch/ctc/kanji_wb/subset/blstm320H5L_drop4_fc_256_adam_lr1e-3_dropen0.2_temp2_input240_2",
    #     "/n/sd8/inaguma/result/csj/pytorch/ctc/kanji_wb/subset/blstm320H5L_drop4_fc_256_adam_lr1e-3_dropen0.2_temp2_input240_3",
    # ]

    model_paths = [
        "/n/sd8/inaguma/result/csj/pytorch/ctc/kanji_wb/subset/conv_64_64_128_128_bn_blstm320H4L_fc_256_adam_lr1e-3_dropen0.2_input240",
        "/n/sd8/inaguma/result/csj/pytorch/ctc/kanji_wb/subset/blstm320H5L_drop4_fc_256_adam_lr1e-3_dropen0.2_input240",
    ]

    temp_infer = 2

    # Load a config file (.yml)
    params_list = []
    for model_path in model_paths:
        params = load_config(join(model_path, 'config.yml'), is_eval=True)
        params_list.append(params)

    # Load dataset
    eval1_data = Dataset(
        data_save_path=args.data_save_path,
        backend=params_list[0]['backend'],
        input_freq=params_list[0]['input_freq'],
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
        input_freq=params_list[0]['input_freq'],
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
        input_freq=params_list[0]['input_freq'],
        use_delta=params_list[0]['use_delta'],
        use_double_delta=params_list[0]['use_double_delta'],
        data_type='eval3', data_size=params_list[0]['data_size'],
        label_type=params_list[0]['label_type'],
        batch_size=args.eval_batch_size, splice=params_list[0]['splice'],
        num_stack=params_list[0]['num_stack'], num_skip=params_list[0]['num_skip'],
        shuffle=False, tool=params_list[0]['tool'])

    # Load model
    models = []
    for i in range(len(model_paths)):
        params_list[i]['num_classes'] = eval1_data.num_classes

        model = load(model_type=params_list[i]['model_type'],
                     params=params_list[i],
                     backend=params_list[i]['backend'])

        # Restore the saved parameters
        model.load_checkpoint(save_path=model_paths[i], epoch=args.epoch)

        # GPU setting
        model.set_cuda(deterministic=False, benchmark=True)
        models.append(model)

    if 'word' in params['label_type']:
        wer_eval1, df_eval1 = eval_word(
            models=models,
            dataset=eval1_data,
            beam_width=args.beam_width,
            max_decode_len=MAX_DECODE_LEN_WORD,
            eval_batch_size=args.eval_batch_size,
            length_penalty=args.length_penalty,
            progressbar=True)
        print('  WER (eval1): %.3f %%' % (wer_eval1 * 100))
        print(df_eval1)

        wer_eval2, df_eval2 = eval_word(
            models=models,
            dataset=eval2_data,
            beam_width=args.beam_width,
            max_decode_len=MAX_DECODE_LEN_WORD,
            eval_batch_size=args.eval_batch_size,
            length_penalty=args.length_penalty,
            progressbar=True)
        print('  WER (eval2): %.3f %%' % (wer_eval2 * 100))
        print(df_eval2)

        wer_eval3, df_eval3 = eval_word(
            models=models,
            dataset=eval3_data,
            beam_width=args.beam_width,
            max_decode_len=MAX_DECODE_LEN_WORD,
            eval_batch_size=args.eval_batch_size,
            length_penalty=args.length_penalty,
            progressbar=True)
        print('  WER (eval3): %.3f %%' % (wer_eval3 * 100))
        print(df_eval3)

        print('  WER (mean): %.3f %%' %
              ((wer_eval1 + wer_eval2 + wer_eval3) * 100 / 3))
    else:
        wer_eval1, cer_eval1, df_eval1 = eval_char(
            models=models,
            dataset=eval1_data,
            beam_width=args.beam_width,
            max_decode_len=MAX_DECODE_LEN_CHAR,
            eval_batch_size=args.eval_batch_size,
            length_penalty=args.length_penalty,
            progressbar=True,
            temperature=temp_infer)
        print(' WER / CER (eval1): %.3f / %.3f %%' %
              ((wer_eval1 * 100), (cer_eval1 * 100)))
        print(df_eval1)

        wer_eval2, cer_eval2, df_eval2 = eval_char(
            models=models,
            dataset=eval2_data,
            beam_width=args.beam_width,
            max_decode_len=MAX_DECODE_LEN_CHAR,
            eval_batch_size=args.eval_batch_size,
            length_penalty=args.length_penalty,
            progressbar=True,
            temperature=temp_infer)
        print(' WER / CER (eval2): %.3f / %.3f %%' %
              ((wer_eval2 * 100), (cer_eval2 * 100)))
        print(df_eval2)

        wer_eval3, cer_eval3, df_eval3 = eval_char(
            models=models,
            dataset=eval3_data,
            beam_width=args.beam_width,
            max_decode_len=MAX_DECODE_LEN_CHAR,
            eval_batch_size=args.eval_batch_size,
            length_penalty=args.length_penalty,
            progressbar=True,
            temperature=temp_infer)
        print(' WER / CER (eval3): %.3f / %.3f %%' %
              ((wer_eval3 * 100), (cer_eval3 * 100)))
        print(df_eval3)

        print('  WER / CER (mean): %.3f / %.3f %%' %
              (((wer_eval1 + wer_eval2 + wer_eval3) * 100 / 3),
               ((cer_eval1 + cer_eval2 + cer_eval3) * 100 / 3)))


if __name__ == '__main__':
    main()
