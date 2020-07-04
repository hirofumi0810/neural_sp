#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Set model name."""

import os

from neural_sp.bin.train_utils import load_config


def _define_encoder_name(dir_name, args):
    if args.enc_type == 'tds':
        from neural_sp.models.seq2seq.encoders.tds import TDSEncoder as module
    elif args.enc_type == 'gated_conv':
        from neural_sp.models.seq2seq.encoders.gated_conv import GatedConvEncoder as module
    elif 'transformer' in args.enc_type:
        from neural_sp.models.seq2seq.encoders.transformer import TransformerEncoder as module
    elif 'conformer' in args.enc_type:
        from neural_sp.models.seq2seq.encoders.conformer import ConformerEncoder as module
    else:
        from neural_sp.models.seq2seq.encoders.rnn import RNNEncoder as module
    if hasattr(module, 'define_name'):
        dir_name = module.define_name(dir_name, args)
    else:
        raise NotImplementedError(module)
    return dir_name


def _define_decoder_name(dir_name, args):
    if args.dec_type in ['transformer', 'transformer_xl']:
        from neural_sp.models.seq2seq.decoders.transformer import TransformerDecoder as module
    elif args.dec_type in ['transformer_transducer', 'transformer_transducer_xl']:
        from neural_sp.models.seq2seq.decoders.transformer_transducer import TransformerTransducer as module
    elif args.dec_type in ['lstm_transducer', 'gru_transducer']:
        from neural_sp.models.seq2seq.decoders.rnn_transducer import RNNTransducer as module
    elif args.dec_type == 'asg':
        from neural_sp.models.seq2seq.decoders.asg import ASGDecoder as module
    else:
        from neural_sp.models.seq2seq.decoders.las import RNNDecoder as module
    if hasattr(module, 'define_name'):
        dir_name = module.define_name(dir_name, args)
    else:
        raise NotImplementedError(module)
    return dir_name


def _define_lm_name(dir_name, args):
    if 'gated_conv' in args.lm_type:
        from neural_sp.models.lm.gated_convlm import GatedConvLM as module
    elif args.lm_type == 'transformer':
        from neural_sp.models.lm.transformerlm import TransformerLM as module
    elif args.lm_type == 'transformer_xl':
        from neural_sp.models.lm.transformer_xl import TransformerXL as module
    else:
        from neural_sp.models.lm.rnnlm import RNNLM as module
    if hasattr(module, 'define_name'):
        dir_name = module.define_name(dir_name, args)
    else:
        raise NotImplementedError(module)
    return dir_name


def set_asr_model_name(args):
    # encoder
    dir_name = args.enc_type.replace('conv_', '')
    dir_name = _define_encoder_name(dir_name, args)

    if args.n_stacks > 1:
        dir_name += '_stack' + str(args.n_stacks)
    else:
        dir_name += '_' + args.subsample_type + str(args.subsample_factor)
    if args.sequence_summary_network:
        dir_name += '_ssn'

    # decoder
    if args.ctc_weight < 1:
        dir_name = _define_decoder_name(dir_name, args)

    # optimization
    dir_name += '_' + args.optimizer
    if args.optimizer == 'noam':
        dir_name += '_lr' + str(args.lr_factor)
    else:
        dir_name += '_lr' + str(args.lr)
    dir_name += '_bs' + str(args.batch_size)
    if args.train_dtype in ["O0", "O1", "O2", "O3"]:
        dir_name += '_' + args.train_dtype
    # if args.shuffle_bucket:
    #     dir_name += '_bucket'
    # if 'transformer' in args.enc_type or 'transformer' in args.dec_type:
    #     dir_name += '_' + args.transformer_param_init

    # regularization
    if args.ctc_weight < 1 and args.ss_prob > 0:
        dir_name += '_ss' + str(args.ss_prob)
    if args.lsm_prob > 0:
        dir_name += '_ls' + str(args.lsm_prob)
    if args.warmup_n_steps > 0:
        dir_name += '_warmup' + str(args.warmup_n_steps)
    if args.accum_grad_n_steps > 1:
        dir_name += '_accum' + str(args.accum_grad_n_steps)

    # LM integration
    if args.lm_fusion:
        dir_name += '_' + args.lm_fusion

    # MTL
    if args.mtl_per_batch:
        if args.ctc_weight > 0:
            dir_name += '_' + args.unit + 'ctc'
        if args.bwd_weight > 0:
            dir_name += '_' + args.unit + 'bwd'
        for sub in ['sub1', 'sub2']:
            if getattr(args, 'train_set_' + sub):
                dir_name += '_' + getattr(args, 'unit_' + sub) + str(getattr(args, 'vocab_' + sub))
                if getattr(args, 'ctc_weight_' + sub) > 0:
                    dir_name += 'ctc'
                if getattr(args, sub + '_weight') - getattr(args, 'ctc_weight_' + sub) > 0:
                    dir_name += 'fwd'
    else:
        if args.ctc_weight > 0:
            dir_name += '_ctc' + str(args.ctc_weight)
        if args.bwd_weight > 0:
            dir_name += '_bwd' + str(args.bwd_weight)
        for sub in ['sub1', 'sub2']:
            if getattr(args, sub + '_weight') > 0:
                dir_name += '_' + getattr(args, 'unit_' + sub) + str(getattr(args, 'vocab_' + sub))
                if getattr(args, 'ctc_weight_' + sub) > 0:
                    dir_name += 'ctc%.1f' % getattr(args, 'ctc_weight_' + sub)
                if getattr(args, sub + '_weight') - getattr(args, 'ctc_weight_' + sub) > 0:
                    dir_name += 'fwd%.2f' % (1.0 - getattr(args, sub + '_weight') - getattr(args, 'ctc_weight_' + sub))
    if args.task_specific_layer:
        dir_name += '_tsl'

    # SpecAugment
    if args.n_freq_masks > 0:
        dir_name += '_' + str(args.freq_width) + 'FM' + str(args.n_freq_masks)
    if args.n_time_masks > 0:
        if args.adaptive_number_ratio > 0:
            dir_name += '_pnum' + str(args.adaptive_number_ratio)
        else:
            dir_name += '_' + str(args.time_width) + 'TM' + str(args.n_time_masks)
        if args.adaptive_size_ratio > 0:
            dir_name += '_psize' + str(args.adaptive_size_ratio)
    if args.input_noise_std > 0:
        dir_name += '_inputnoisestd'
    if args.weight_noise_std > 0:
        dir_name += '_weightnoisestd'

    # contextualization
    if args.discourse_aware:
        dir_name += '_discourse'
    if args.mem_len > 0:
        dir_name += '_mem' + str(args.mem_len)
    if args.bptt > 0:
        dir_name += '_bptt' + str(args.bptt)

    # Pre-training
    if args.asr_init and os.path.isfile(args.asr_init):
        conf_init = load_config(os.path.join(os.path.dirname(args.asr_init), 'conf.yml'))
        dir_name += '_' + conf_init['unit'] + 'pt'
    if args.freeze_encoder:
        dir_name += '_encfreeze'
    if args.lm_init:
        dir_name += '_lminit'

    # knowledge distillation
    if args.teacher:
        dir_name += '_KD' + str(args.soft_label_weight)
    if args.teacher_lm:
        dir_name += '_lmKD' + str(args.soft_label_weight)

    # MBR training
    if args.mbr_training:
        dir_name += '_MBR' + str(args.recog_beam_width) + 'best'
        dir_name += '_ce' + str(args.mbr_ce_weight) + '_smooth' + str(args.recog_softmax_smoothing)

    if args.n_gpus > 1:
        dir_name += '_' + str(args.n_gpus) + 'GPU'
    return dir_name


def set_lm_name(args):
    dir_name = ''
    dir_name = _define_lm_name(dir_name, args)

    # optimization
    dir_name += '_' + args.optimizer
    if args.optimizer == 'noam':
        dir_name += '_lr' + str(args.lr_factor)
    else:
        dir_name += '_lr' + str(args.lr)
    dir_name += '_bs' + str(args.batch_size)
    if args.train_dtype in ["O0", "O1", "O2", "O3"]:
        dir_name += '_' + args.train_dtype

    dir_name += '_bptt' + str(args.bptt)

    # regularization
    dir_name += '_dropI' + str(args.dropout_in) + 'H' + str(args.dropout_hidden)
    if getattr(args, 'dropout_layer', 0) > 0:
        dir_name += 'Layer' + str(args.dropout_layer)
    if args.lsm_prob > 0:
        dir_name += '_ls' + str(args.lsm_prob)
    if args.warmup_n_steps > 0:
        dir_name += '_warmup' + str(args.warmup_n_steps)
    if args.accum_grad_n_steps > 1:
        dir_name += '_accum' + str(args.accum_grad_n_steps)

    if args.backward:
        dir_name += '_bwd'
    if args.shuffle:
        dir_name += '_shuffle'
    if args.serialize:
        dir_name += '_serialize'
    return dir_name
