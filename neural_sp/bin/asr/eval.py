#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


"""Evaluate the ASR model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import os
import time

from neural_sp.bin.args_asr import parse
from neural_sp.bin.eval_utils import average_checkpoints
from neural_sp.bin.train_utils import load_checkpoint
from neural_sp.bin.train_utils import load_config
from neural_sp.bin.train_utils import set_logger
from neural_sp.datasets.asr import Dataset
from neural_sp.evaluators.character import eval_char
from neural_sp.evaluators.phone import eval_phone
from neural_sp.evaluators.ppl import eval_ppl
from neural_sp.evaluators.word import eval_word
from neural_sp.evaluators.wordpiece import eval_wordpiece
from neural_sp.models.lm.build import build_lm
from neural_sp.models.seq2seq.speech2text import Speech2Text


def main():

    args = parse()

    # Load a conf file
    dir_name = os.path.dirname(args.recog_model[0])
    conf = load_config(os.path.join(dir_name, 'conf.yml'))

    # Overwrite conf
    for k, v in conf.items():
        if 'recog' not in k:
            setattr(args, k, v)
    recog_params = vars(args)

    # Setting for logging
    if os.path.isfile(os.path.join(args.recog_dir, 'decode.log')):
        os.remove(os.path.join(args.recog_dir, 'decode.log'))
    logger = set_logger(os.path.join(args.recog_dir, 'decode.log'),
                        key='decoding', stdout=args.recog_stdout)

    wer_avg, cer_avg, per_avg = 0, 0, 0
    ppl_avg, loss_avg = 0, 0
    for i, s in enumerate(args.recog_sets):
        # Load dataset
        dataset = Dataset(corpus=args.corpus,
                          tsv_path=s,
                          dict_path=os.path.join(dir_name, 'dict.txt'),
                          dict_path_sub1=os.path.join(dir_name, 'dict_sub1.txt') if os.path.isfile(
                              os.path.join(dir_name, 'dict_sub1.txt')) else False,
                          dict_path_sub2=os.path.join(dir_name, 'dict_sub2.txt') if os.path.isfile(
                              os.path.join(dir_name, 'dict_sub2.txt')) else False,
                          nlsyms=os.path.join(dir_name, 'nlsyms.txt'),
                          wp_model=os.path.join(dir_name, 'wp.model'),
                          wp_model_sub1=os.path.join(dir_name, 'wp_sub1.model'),
                          wp_model_sub2=os.path.join(dir_name, 'wp_sub2.model'),
                          unit=args.unit,
                          unit_sub1=args.unit_sub1,
                          unit_sub2=args.unit_sub2,
                          batch_size=args.recog_batch_size,
                          is_test=True)

        if i == 0:
            # Load the ASR model
            model = Speech2Text(args, dir_name)
            model = load_checkpoint(model, args.recog_model[0])[0]
            epoch = int(args.recog_model[0].split('-')[-1])

            # Model averaging for Transformer
            if 'transformer' in conf['enc_type'] and conf['dec_type'] == 'transformer':
                model = average_checkpoints(model, args.recog_model[0], epoch,
                                            n_average=args.recog_n_average)

            # Ensemble (different models)
            ensemble_models = [model]
            if len(args.recog_model) > 1:
                for recog_model_e in args.recog_model[1:]:
                    conf_e = load_config(os.path.join(os.path.dirname(recog_model_e), 'conf.yml'))
                    args_e = copy.deepcopy(args)
                    for k, v in conf_e.items():
                        if 'recog' not in k:
                            setattr(args_e, k, v)
                    model_e = Speech2Text(args_e)
                    model_e = load_checkpoint(model_e, recog_model_e)[0]
                    model_e.cuda()
                    ensemble_models += [model_e]

            # Load the LM for shallow fusion
            if not args.lm_fusion:
                if args.recog_lm is not None and args.recog_lm_weight > 0:
                    conf_lm = load_config(os.path.join(os.path.dirname(args.recog_lm), 'conf.yml'))
                    args_lm = argparse.Namespace()
                    for k, v in conf_lm.items():
                        setattr(args_lm, k, v)
                    lm = build_lm(args_lm, wordlm=args.recog_wordlm,
                                  lm_dict_path=os.path.join(os.path.dirname(args.recog_lm), 'dict.txt'),
                                  asr_dict_path=os.path.join(dir_name, 'dict.txt'))
                    lm = load_checkpoint(lm, args.recog_lm)[0]
                    if args_lm.backward:
                        model.lm_bwd = lm
                    else:
                        model.lm_fwd = lm

                if args.recog_lm_bwd is not None and args.recog_lm_weight > 0 \
                        and (args.recog_fwd_bwd_attention or args.recog_reverse_lm_rescoring):
                    conf_lm = load_config(os.path.join(os.path.dirname(args.recog_lm_bwd), 'conf.yml'))
                    args_lm_bwd = argparse.Namespace()
                    for k, v in conf_lm.items():
                        setattr(args_lm_bwd, k, v)
                    lm_bwd = build_lm(args_lm_bwd)
                    lm_bwd = load_checkpoint(lm_bwd, args.recog_lm_bwd)[0]
                    model.lm_bwd = lm_bwd

            if not args.recog_unit:
                args.recog_unit = args.unit

            logger.info('recog unit: %s' % args.recog_unit)
            logger.info('recog metric: %s' % args.recog_metric)
            logger.info('recog oracle: %s' % args.recog_oracle)
            logger.info('epoch: %d' % epoch)
            logger.info('batch size: %d' % args.recog_batch_size)
            logger.info('beam width: %d' % args.recog_beam_width)
            logger.info('min length ratio: %.3f' % args.recog_min_len_ratio)
            logger.info('max length ratio: %.3f' % args.recog_max_len_ratio)
            logger.info('length penalty: %.3f' % args.recog_length_penalty)
            logger.info('coverage penalty: %.3f' % args.recog_coverage_penalty)
            logger.info('coverage threshold: %.3f' % args.recog_coverage_threshold)
            logger.info('CTC weight: %.3f' % args.recog_ctc_weight)
            logger.info('LM path: %s' % args.recog_lm)
            logger.info('LM path (bwd): %s' % args.recog_lm_bwd)
            logger.info('LM weight: %.3f' % args.recog_lm_weight)
            logger.info('GNMT: %s' % args.recog_gnmt_decoding)
            logger.info('forward-backward attention: %s' % args.recog_fwd_bwd_attention)
            logger.info('reverse LM rescoring: %s' % args.recog_reverse_lm_rescoring)
            logger.info('resolving UNK: %s' % args.recog_resolving_unk)
            logger.info('ensemble: %d' % (len(ensemble_models)))
            logger.info('ASR decoder state carry over: %s' % (args.recog_asr_state_carry_over))
            logger.info('LM state carry over: %s' % (args.recog_lm_state_carry_over))
            logger.info('model average (Transformer): %d' % (args.recog_n_average))

            # GPU setting
            model.cuda()

        start_time = time.time()

        if args.recog_metric == 'edit_distance':
            if args.recog_unit in ['word', 'word_char']:
                wer, cer, _ = eval_word(ensemble_models, dataset, recog_params,
                                        epoch=epoch - 1,
                                        recog_dir=args.recog_dir,
                                        progressbar=True)
                wer_avg += wer
                cer_avg += cer
            elif args.recog_unit == 'wp':
                wer, cer = eval_wordpiece(ensemble_models, dataset, recog_params,
                                          epoch=epoch - 1,
                                          recog_dir=args.recog_dir,
                                          progressbar=True)
                wer_avg += wer
                cer_avg += cer
            elif 'char' in args.recog_unit:
                wer, cer = eval_char(ensemble_models, dataset, recog_params,
                                     epoch=epoch - 1,
                                     recog_dir=args.recog_dir,
                                     progressbar=True,
                                     task_idx=0)
                #  task_idx=1 if args.recog_unit and 'char' in args.recog_unit else 0)
                wer_avg += wer
                cer_avg += cer
            elif 'phone' in args.recog_unit:
                per = eval_phone(ensemble_models, dataset, recog_params,
                                 epoch=epoch - 1,
                                 recog_dir=args.recog_dir,
                                 progressbar=True)
                per_avg += per
            else:
                raise ValueError(args.recog_unit)
        elif args.recog_metric == 'acc':
            raise NotImplementedError
        elif args.recog_metric in ['ppl', 'loss']:
            ppl, loss = eval_ppl(ensemble_models, dataset,
                                 progressbar=True)
            ppl_avg += ppl
            loss_avg += loss
        elif args.recog_metric == 'bleu':
            raise NotImplementedError
        else:
            raise NotImplementedError
        logger.info('Elasped time: %.2f [sec]:' % (time.time() - start_time))

    if args.recog_metric == 'edit_distance':
        if 'phone' in args.recog_unit:
            logger.info('PER (avg.): %.2f %%\n' % (per_avg / len(args.recog_sets)))
        else:
            logger.info('WER / CER (avg.): %.2f / %.2f %%\n' %
                        (wer_avg / len(args.recog_sets), cer_avg / len(args.recog_sets)))
    elif args.recog_metric in ['ppl', 'loss']:
        logger.info('PPL (avg.): %.2f\n' % (ppl_avg / len(args.recog_sets)))
        print('PPL (avg.): %.2f' % (ppl_avg / len(args.recog_sets)))
        logger.info('Loss (avg.): %.2f\n' % (loss_avg / len(args.recog_sets)))
        print('Loss (avg.): %.2f' % (loss_avg / len(args.recog_sets)))


if __name__ == '__main__':
    main()
