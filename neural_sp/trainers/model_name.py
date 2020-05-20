#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Set model name."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from neural_sp.bin.train_utils import load_config


def set_asr_model_name(args):
    # encoder
    dir_name = args.enc_type.replace('conv_', '')
    if args.conv_channels and len(args.conv_channels.split('_')) > 0 and 'conv' in args.enc_type:
        tmp = dir_name
        dir_name = 'conv' + str(len(args.conv_channels.split('_'))) + 'L'
        if args.conv_batch_norm:
            dir_name += 'bn'
        if args.conv_layer_norm:
            dir_name += 'ln'
        dir_name += tmp
    if 'transformer' in args.enc_type:
        dir_name += str(args.transformer_d_model) + 'dmodel'
        dir_name += str(args.transformer_d_ff) + 'dff'
        dir_name += str(args.enc_n_layers) + 'L'
        dir_name += str(args.transformer_n_heads) + 'head'
        dir_name += 'pe' + str(args.transformer_enc_pe_type)
        if args.lc_chunk_size_left > 0 or args.lc_chunk_size_current > 0 or args.lc_chunk_size_right > 0:
            dir_name += '_chunkL' + str(args.lc_chunk_size_left) + 'C' + \
                str(args.lc_chunk_size_current) + 'R' + str(args.lc_chunk_size_right)
    else:
        dir_name += str(args.enc_n_units) + 'H'
        if args.enc_n_projs > 0:
            dir_name += str(args.enc_n_projs) + 'P'
        dir_name += str(args.enc_n_layers) + 'L'
        if args.bidirectional_sum_fwd_bwd:
            dir_name += '_sumfwdbwd'
        if args.lc_chunk_size_left > 0 or args.lc_chunk_size_right > 0:
            dir_name += '_chunkL' + str(args.lc_chunk_size_left) + 'R' + str(args.lc_chunk_size_right)
    if args.n_stacks > 1:
        dir_name += '_stack' + str(args.n_stacks)
    else:
        dir_name += '_' + args.subsample_type + str(args.subsample_factor)
    if args.sequence_summary_network:
        dir_name += '_ssn'

    # decoder
    if args.ctc_weight < 1:
        dir_name += '_' + args.dec_type
        if 'transformer' in args.dec_type:
            dir_name += str(args.transformer_d_model) + 'dmodel'
            dir_name += str(args.transformer_d_ff) + 'dff'
            dir_name += str(args.dec_n_layers) + 'L'
            dir_name += str(args.transformer_n_heads) + 'head'
            dir_name += 'pe' + str(args.transformer_dec_pe_type)
            dir_name += args.transformer_attn_type
            if 'mocha' in args.transformer_attn_type:
                dir_name += '_mono' + str(args.mocha_n_heads_mono) + 'H'
                dir_name += '_chunk' + str(args.mocha_n_heads_chunk) + 'H'
                dir_name += '_chunk' + str(args.mocha_chunk_size)
                dir_name += '_bias' + str(args.mocha_init_r)
                if args.mocha_no_denominator:
                    dir_name += '_denom1'
                if args.mocha_1dconv:
                    dir_name += '_1dconv'
                if args.mocha_quantity_loss_weight > 0:
                    dir_name += '_quantity' + str(args.mocha_quantity_loss_weight)
                if args.mocha_head_divergence_loss_weight > 0:
                    dir_name += '_headdiv' + str(args.mocha_head_divergence_loss_weight)
                if args.mocha_latency_metric:
                    dir_name += '_' + args.mocha_latency_metric
                    dir_name += str(args.mocha_latency_loss_weight)
            if args.mocha_first_layer > 1:
                dir_name += '_from' + str(args.mocha_first_layer) + 'L'
            if args.dropout_head > 0:
                dir_name += 'drophead' + str(args.dropout_head)
        elif 'asg' not in args.dec_type:
            dir_name += str(args.dec_n_units) + 'H'
            if args.dec_n_projs > 0:
                dir_name += str(args.dec_n_projs) + 'P'
            dir_name += str(args.dec_n_layers) + 'L'
            if 'transducer' not in args.dec_type:
                dir_name += '_' + args.attn_type
                if args.attn_sigmoid:
                    dir_name += '_sig'
                if 'mocha' in args.attn_type:
                    dir_name += '_chunk' + str(args.mocha_chunk_size)
                    if args.mocha_n_heads_mono > 1:
                        dir_name += '_mono' + str(args.mocha_n_heads_mono) + 'H'
                    if args.mocha_no_denominator:
                        dir_name += '_denom1'
                    if args.mocha_1dconv:
                        dir_name += '_1dconv'
                    if args.attn_sharpening_factor:
                        dir_name += '_temp' + str(args.attn_sharpening_factor)
                    if args.mocha_quantity_loss_weight > 0:
                        dir_name += '_quantity' + str(args.mocha_quantity_loss_weight)
                elif args.attn_type == 'gmm':
                    dir_name += '_mix' + str(args.gmm_attn_n_mixtures)
                if args.mocha_latency_metric:
                    dir_name += '_' + args.mocha_latency_metric
                    dir_name += str(args.mocha_latency_loss_weight)
                if args.attn_n_heads > 1:
                    dir_name += '_head' + str(args.attn_n_heads)
        if args.tie_embedding:
            dir_name += '_tie'

    # optimization
    dir_name += '_' + args.optimizer
    if args.optimizer == 'noam':
        dir_name += '_lr' + str(args.lr_factor)
    else:
        dir_name += '_lr' + str(args.lr)
    dir_name += '_bs' + str(args.batch_size)
    if args.shuffle_bucket:
        dir_name += '_bucket'

    if 'transformer' in args.enc_type or 'transformer' in args.dec_type:
        dir_name += '_' + args.transformer_param_init

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
                    dir_name += 'ctc' + str(getattr(args, 'ctc_weight_' + sub))
                if getattr(args, sub + '_weight') - getattr(args, 'ctc_weight_' + sub) > 0:
                    dir_name += 'fwd' + str(1 - getattr(args, sub + '_weight') - getattr(args, 'ctc_weight_' + sub))
    if args.task_specific_layer:
        dir_name += '_tsl'

    # SpecAugment
    if args.n_freq_masks > 0:
        dir_name += '_' + str(args.freq_width) + 'FM' + str(args.n_freq_masks)
    if args.n_time_masks > 0:
        dir_name += '_' + str(args.time_width) + 'TM' + str(args.n_time_masks)
    if args.flip_time_prob > 0:
        dir_name += '_flipT' + str(args.flip_time_prob)
    if args.flip_freq_prob > 0:
        dir_name += '_flipF' + str(args.flip_freq_prob)
    if args.weight_noise:
        dir_name += '_weightnoise'

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
    dir_name = args.lm_type
    if 'transformer' in args.lm_type:
        dir_name += str(args.transformer_d_model) + 'dmodel'
        dir_name += str(args.transformer_d_ff) + 'dff'
        dir_name += str(args.n_layers) + 'L'
        dir_name += str(args.transformer_n_heads) + 'head'
        dir_name += 'pe' + str(args.transformer_pe_type)
    elif 'gated_conv' not in args.lm_type or args.lm_type == 'gated_conv_custom':
        dir_name += str(args.n_units) + 'H'
        dir_name += str(args.n_projs) + 'P'
        dir_name += str(args.n_layers) + 'L'
    if 'transformer' not in args.lm_type:
        dir_name += '_emb' + str(args.emb_dim)
    dir_name += '_' + args.optimizer
    if args.optimizer == 'noam':
        dir_name += '_lr' + str(args.lr_factor)
    else:
        dir_name += '_lr' + str(args.lr)
    dir_name += '_bs' + str(args.batch_size)
    dir_name += '_bptt' + str(args.bptt)
    if args.adaptive_bptt:
        dir_name += '_' + args.adaptive_bptt
    if args.mem_len > 0:
        dir_name += '_mem' + str(args.mem_len)
    if args.lm_type == 'transformer_xl' and args.zero_center_offset:
        dir_name += '_zero_center'
    if args.tie_embedding:
        dir_name += '_tie'
    if 'gated_conv' not in args.lm_type and 'transformer' not in args.lm_type:
        if args.residual:
            dir_name += '_residual'
        if args.use_glu:
            dir_name += '_glu'
        if args.n_units_null_context > 0:
            dir_name += '_nullcv' + str(args.n_units_null_context)

    # regularization
    dir_name += '_dropI' + str(args.dropout_in) + 'H' + str(args.dropout_hidden)
    if args.dropout_residual > 0:
        dir_name += 'res' + str(args.dropout_residual)
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
    if args.min_n_tokens > 1:
        dir_name += '_' + str(args.min_n_tokens) + 'tokens'
    if args.adaptive_softmax:
        dir_name += '_adaptiveSM'
    return dir_name
