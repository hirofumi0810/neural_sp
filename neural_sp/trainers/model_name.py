#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Set model name."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from neural_sp.bin.train_utils import load_config


def set_asr_model_name(args, subsample_factor):
    # encoder
    dir_name = args.enc_type.replace('conv_', '')
    if args.conv_channels and len(args.conv_channels.split('_')) > 0 and 'conv' in args.enc_type:
        tmp = dir_name
        dir_name = 'conv' + str(len(args.conv_channels.split('_'))) + 'L'
        if args.conv_batch_norm:
            dir_name += 'bn'
        dir_name += tmp
    if 'transformer' in args.enc_type:
        dir_name += str(args.d_model) + 'dmodel'
        dir_name += str(args.d_ff) + 'dff'
        dir_name += str(args.enc_n_layers) + 'L'
        dir_name += str(args.transformer_attn_n_heads) + 'head'
    else:
        dir_name += str(args.enc_n_units) + 'H'
        if args.enc_n_projs > 0:
            dir_name += str(args.enc_n_projs) + 'P'
        dir_name += str(args.enc_n_layers) + 'L'
        if args.enc_nin:
            dir_name += 'NiN'
        if args.bidirectional_sum_fwd_bwd:
            dir_name += '_sumfwdbwd'
    if args.n_stacks > 1:
        dir_name += '_stack' + str(args.n_stacks)
    else:
        dir_name += '_' + args.subsample_type + str(subsample_factor)
    if args.sequence_summary_network:
        dir_name += '_ssn'

    # decoder
    if args.ctc_weight < 1:
        dir_name += '_' + args.dec_type
        if args.dec_type in ['transformer', 'transformer_transducer']:
            dir_name += str(args.d_model) + 'dmodel'
            dir_name += str(args.d_ff) + 'dff'
            dir_name += str(args.dec_n_layers) + 'L'
            dir_name += str(args.transformer_attn_n_heads) + 'head'
        else:
            dir_name += str(args.dec_n_units) + 'H'
            if args.dec_n_projs > 0:
                dir_name += str(args.dec_n_projs) + 'P'
            dir_name += str(args.dec_n_layers) + 'L'
            dir_name += '_' + args.attn_type
            if args.attn_sigmoid:
                dir_name += '_sig'
            if args.attn_type == 'mocha':
                dir_name += '_chunk' + str(args.mocha_chunk_size)
                if args.mocha_adaptive:
                    dir_name += '_adaptive'
                if args.mocha_1dconv:
                    dir_name += '_1dconv'
                if args.attn_sharpening_factor:
                    dir_name += '_temp' + str(args.attn_sharpening_factor)
        if args.attn_n_heads > 1:
            dir_name += '_head' + str(args.attn_n_heads)
        if args.tie_embedding:
            dir_name += '_tie'

    # optimization
    dir_name += '_' + args.optimizer
    dir_name += '_lr' + str(args.lr)
    dir_name += '_bs' + str(args.batch_size)

    # regularization
    if args.ctc_weight < 1:
        dir_name += '_ss' + str(args.ss_prob)
    dir_name += '_ls' + str(args.lsm_prob)
    if args.warmup_n_steps > 0:
        dir_name += '_warmpup' + str(args.warmup_n_steps)
    if args.accum_grad_n_steps > 0:
        dir_name += '_accum' + str(args.accum_grad_n_steps)

    # LM integration
    if args.lm_fusion:
        dir_name += '_' + args.lm_fusion_type

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
                    dir_name += 'ctc' + str(getattr(args, 'ctc_weight_' + sub))
                if getattr(args, sub + '_weight') - getattr(args, 'ctc_weight_' + sub) > 0:
                    dir_name += 'fwd' + str(1 - getattr(args, sub + '_weight') - getattr(args, 'ctc_weight_' + sub))
    if args.task_specific_layer:
        dir_name += '_tsl'

    # SpecAugment
    if args.gaussian_noise:
        dir_name += '_noise'
    if args.n_freq_masks > 0:
        dir_name += '_' + str(args.freq_width) + 'FM' + str(args.n_freq_masks)
    if args.n_time_masks > 0:
        dir_name += '_' + str(args.time_width) + 'TM' + str(args.n_time_masks)

    # contextualization
    if args.discourse_aware:
        dir_name += '_' + str(args.discourse_aware)

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

    if args.n_gpus > 1:
        dir_name += '_' + str(args.n_gpus) + 'GPU'
    return dir_name


def set_lm_name(args):
    dir_name = args.lm_type
    if args.lm_type == 'transformer':
        dir_name += str(args.d_model) + 'dmodel'
        dir_name += str(args.d_ff) + 'dff'
        dir_name += str(args.n_layers) + 'L'
        dir_name += str(args.attn_n_heads) + 'head'
    elif 'gated_conv' not in args.lm_type or args.lm_type == 'gated_conv_custom':
        dir_name += str(args.n_units) + 'H'
        dir_name += str(args.n_projs) + 'P'
        dir_name += str(args.n_layers) + 'L'
    if args.lm_type != 'transformer':
        dir_name += '_emb' + str(args.emb_dim)
    dir_name += '_' + args.optimizer
    dir_name += '_lr' + str(args.lr)
    dir_name += '_bs' + str(args.batch_size)
    dir_name += '_bptt' + str(args.bptt)
    if args.tie_embedding:
        dir_name += '_tie'
    if 'gated_conv' not in args.lm_type and args.lm_type != 'transformer':
        if args.residual:
            dir_name += '_residual'
        if args.use_glu:
            dir_name += '_glu'
        if args.n_units_null_context > 0:
            dir_name += '_nullcv' + str(args.n_units_null_context)

    # regularization
    dir_name += '_dropI' + str(args.dropout_in) + 'H' + str(args.dropout_hidden)
    if args.lsm_prob > 0:
        dir_name += '_ls' + str(args.lsm_prob)
    if args.warmup_n_steps > 0:
        dir_name += '_warmpup' + str(args.warmup_n_steps)
    if args.accum_grad_n_tokens > 0:
        dir_name += '_accum' + str(args.accum_grad_n_tokens)

    if args.backward:
        dir_name += '_bwd'
    if args.serialize:
        dir_name += '_serialize'
    if args.min_n_tokens > 1:
        dir_name += '_' + str(args.min_n_tokens) + 'tokens'
    if args.adaptive_softmax:
        dir_name += '_adaptiveSM'
    return dir_name
