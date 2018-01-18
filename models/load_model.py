#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def load(model_type, params, backend='pytorch'):
    """Load an encoder.
    Args:
        model_type (string): ctc or student_ctc or attention or
            hierarchical_ctc or hierarchical_attention or
        params (dict):
        backend (string, optional): pytorch or chainer
    Returns:
        model (nn.Module): An encoder class
    """
    # TODO: remove these
    if 'residual' not in params.keys():
        params['residual'] = False
    if 'dense_residual' not in params.keys():
        params['dense_residual'] = False
    if 'encoder_residual' not in params.keys():
        params['encoder_residual'] = False
    if 'encoder_dense_residual' not in params.keys():
        params['encoder_dense_residual'] = False
    if 'decoder_residual' not in params.keys():
        params['decoder_residual'] = False
    if 'decoder_dense_residual' not in params.keys():
        params['decoder_dense_residual'] = False

    model_name = params['encoder_type']
    if params['encoder_type'] in ['cnn', 'resnet']:
        for c in params['conv_channels']:
            model_name += '_' + str(c)
    else:
        if params['encoder_bidirectional']:
            model_name = 'b' + model_name
        if len(params['conv_channels']) != 0:
            name_tmp = model_name
            model_name = 'conv_'
            for c in params['conv_channels']:
                model_name += str(c) + '_'
            model_name = model_name + name_tmp
    if bool(params['batch_norm']):
        model_name += '_bn'

    if model_type == 'ctc':
        if 'activation' not in params.keys():
            params['activation'] = 'relu'

        if backend == 'pytorch':
            from models.pytorch.ctc.ctc import CTC
        elif backend == 'chainer':
            from models.chainer.ctc.ctc import CTC
        else:
            raise TypeError

        model = CTC(
            input_size=params['input_channel'] *
            (1 + int(params['use_delta'] + int(params['use_double_delta']))),
            encoder_type=params['encoder_type'],
            encoder_bidirectional=params['encoder_bidirectional'],
            encoder_num_units=params['encoder_num_units'],
            encoder_num_proj=params['encoder_num_proj'],
            encoder_num_layers=params['encoder_num_layers'],
            fc_list=params['fc_list'],
            dropout=params['dropout'],
            num_classes=params['num_classes'],
            parameter_init=params['parameter_init'],
            subsample_list=params['subsample_list'],
            subsample_type=params['subsample_type'],
            logits_temperature=params['logits_temperature'],
            num_stack=params['num_stack'],
            splice=params['splice'],
            conv_channels=params['conv_channels'],
            conv_kernel_sizes=params['conv_kernel_sizes'],
            conv_strides=params['conv_strides'],
            poolings=params['poolings'],
            activation=params['activation'],
            batch_norm=params['batch_norm'],
            label_smoothing_prob=params['label_smoothing_prob'],
            weight_noise_std=params['weight_noise_std'],
            residual=params['residual'],
            dense_residual=params['dense_residual'])

        model.name = model_name
        if params['encoder_type'] not in ['cnn', 'resnet']:
            model.name += str(params['encoder_num_units']) + 'H'
            model.name += str(params['encoder_num_layers']) + 'L'
            if params['encoder_num_proj'] != 0:
                model.name += '_proj' + str(params['encoder_num_proj'])
            if sum(params['subsample_list']) > 0:
                model.name += '_' + params['subsample_type'] + \
                    str(2 ** sum(params['subsample_list']))
            if params['num_stack'] != 1:
                model.name += '_stack' + str(params['num_stack'])
        if len(params['fc_list']) != 0:
            model.name += '_fc'
            for l in params['fc_list']:
                model.name += '_' + str(l)
        model.name += '_' + params['optimizer']
        model.name += '_lr' + str(params['learning_rate'])
        if params['dropout'] != 0:
            model.name += '_drop' + str(params['dropout'])
        if params['logits_temperature'] != 1:
            model.name += '_temp' + str(params['logits_temperature'])
        if params['label_smoothing_prob'] > 0:
            model.name += '_labelsmooth' + str(params['label_smoothing_prob'])
        if params['weight_noise_std'] != 0:
            model.name += '_noise' + str(params['weight_noise_std'])
        if params['encoder_type'] == 'cnn':
            model.name += '_' + params['activation']
        if bool(params['residual']):
            model.name += '_res'
        if bool(params['dense_residual']):
            model.name += '_dense_res'
        model.name += '_input' + str(model.input_size)

    elif model_type == 'student_ctc':
        if 'activation' not in params.keys():
            params['activation'] = 'relu'

        if backend == 'pytorch':
            from models.pytorch.ctc.student_ctc import StudentCTC
        elif backend == 'chainer':
            raise NotImplementedError
        else:
            raise TypeError

        model = StudentCTC(
            input_size=params['input_channel'] *
            (1 + int(params['use_delta'] + int(params['use_double_delta']))),
            encoder_type=params['encoder_type'],
            encoder_bidirectional=params['encoder_bidirectional'],
            encoder_num_units=params['encoder_num_units'],
            encoder_num_proj=params['encoder_num_proj'],
            encoder_num_layers=params['encoder_num_layers'],
            fc_list=params['fc_list'],
            dropout=params['dropout'],
            num_classes=params['num_classes'],
            parameter_init=params['parameter_init'],
            subsample_list=params['subsample_list'],
            subsample_type=params['subsample_type'],
            logits_temperature=params['logits_temperature'],
            num_stack=params['num_stack'],
            splice=params['splice'],
            conv_channels=params['conv_channels'],
            conv_kernel_sizes=params['conv_kernel_sizes'],
            conv_strides=params['conv_strides'],
            poolings=params['poolings'],
            activation=params['activation'],
            batch_norm=params['batch_norm'],
            weight_noise_std=params['weight_noise_std'],
            residual=params['residual'],
            dense_residual=params['dense_residual'])

        model.name = model_name
        if params['encoder_type'] not in ['cnn', 'resnet']:
            model.name += str(params['encoder_num_units']) + 'H'
            model.name += str(params['encoder_num_layers']) + 'L'
            if params['encoder_num_proj'] != 0:
                model.name += '_proj' + str(params['encoder_num_proj'])
            if sum(params['subsample_list']) > 0:
                model.name += '_' + params['subsample_type'] + \
                    str(2 ** sum(params['subsample_list']))
            if params['num_stack'] != 1:
                model.name += '_stack' + str(params['num_stack'])
        if len(params['fc_list']) != 0:
            model.name += '_fc'
            for l in params['fc_list']:
                model.name += '_' + str(l)
        model.name += '_' + params['optimizer']
        model.name += '_lr' + str(params['learning_rate'])
        if params['dropout'] != 0:
            model.name += '_drop' + str(params['dropout'])
        if params['logits_temperature'] != 1:
            model.name += '_temp' + str(params['logits_temperature'])
        if params['weight_noise_std'] != 0:
            model.name += '_noise' + str(params['weight_noise_std'])
        if bool(params['residual']):
            model.name += '_res'
        if bool(params['dense_residual']):
            model.name += '_dense_res'

    elif model_type == 'attention':

        if backend == 'pytorch':
            from models.pytorch.attention.attention_seq2seq import AttentionSeq2seq
        elif backend == 'chainer':
            from models.chainer.attention.attention_seq2seq import AttentionSeq2seq
        else:
            raise TypeError

        model = AttentionSeq2seq(
            input_size=params['input_channel'] *
            (1 + int(params['use_delta'] + int(params['use_double_delta']))),
            encoder_type=params['encoder_type'],
            encoder_bidirectional=params['encoder_bidirectional'],
            encoder_num_units=params['encoder_num_units'],
            encoder_num_proj=params['encoder_num_proj'],
            encoder_num_layers=params['encoder_num_layers'],
            encoder_dropout=params['dropout_encoder'],
            attention_type=params['attention_type'],
            attention_dim=params['attention_dim'],
            decoder_type=params['decoder_type'],
            decoder_num_units=params['decoder_num_units'],
            decoder_num_layers=params['decoder_num_layers'],
            decoder_dropout=params['dropout_decoder'],
            embedding_dim=params['embedding_dim'],
            num_classes=params['num_classes'],
            parameter_init=params['parameter_init'],
            subsample_list=params['subsample_list'],
            subsample_type=params['subsample_type'],
            init_dec_state=params['init_dec_state'],
            sharpening_factor=params['sharpening_factor'],
            logits_temperature=params['logits_temperature'],
            sigmoid_smoothing=params['sigmoid_smoothing'],
            coverage_weight=params['coverage_weight'],
            ctc_loss_weight=params['ctc_loss_weight'],
            attention_conv_num_channels=params['attention_conv_num_channels'],
            attention_conv_width=params['attention_conv_width'],
            num_stack=params['num_stack'],
            splice=params['splice'],
            conv_channels=params['conv_channels'],
            conv_kernel_sizes=params['conv_kernel_sizes'],
            conv_strides=params['conv_strides'],
            poolings=params['poolings'],
            batch_norm=params['batch_norm'],
            scheduled_sampling_prob=params['scheduled_sampling_prob'],
            scheduled_sampling_ramp_max_step=params['scheduled_sampling_ramp_max_step'],
            label_smoothing_prob=params['label_smoothing_prob'],
            weight_noise_std=params['weight_noise_std'],
            encoder_residual=params['encoder_residual'],
            encoder_dense_residual=params['encoder_dense_residual'],
            decoder_residual=params['decoder_residual'],
            decoder_dense_residual=params['decoder_dense_residual'])

        model.name = model_name
        if params['encoder_type'] not in ['cnn', 'resnet']:
            model.name += str(params['encoder_num_units']) + 'H'
            model.name += str(params['encoder_num_layers']) + 'L'
            if params['encoder_num_proj'] != 0:
                model.name += '_proj' + str(params['encoder_num_proj'])
            if sum(params['subsample_list']) > 0:
                model.name += '_' + params['subsample_type'] + \
                    str(2 ** sum(params['subsample_list']))
            if params['num_stack'] != 1:
                model.name += '_stack' + str(params['num_stack'])
        model.name += '_' + params['decoder_type']
        model.name += str(params['decoder_num_units']) + 'H'
        model.name += str(params['decoder_num_layers']) + 'L'
        model.name += '_' + params['optimizer']
        model.name += '_lr' + str(params['learning_rate'])
        model.name += '_' + params['attention_type']
        if params['dropout_encoder'] != 0 or params['dropout_decoder'] != 0:
            model.name += '_drop'
            if params['dropout_encoder'] != 0:
                model.name += 'en' + str(params['dropout_encoder'])
            if params['dropout_decoder'] != 0:
                model.name += 'de' + str(params['dropout_decoder'])
        if params['sharpening_factor'] != 1:
            model.name += '_sharp' + str(params['sharpening_factor'])
        if params['logits_temperature'] != 1:
            model.name += '_temp' + str(params['logits_temperature'])
        if bool(params['sigmoid_smoothing']):
            model.name += '_smooth'
        if params['coverage_weight'] > 0:
            model.name += '_coverage' + str(params['coverage_weight'])
        if params['ctc_loss_weight'] > 0:
            model.name += '_ctc' + str(params['ctc_loss_weight'])
        if params['scheduled_sampling_prob'] > 0:
            model.name += '_scheduled' + str(params['scheduled_sampling_prob'])
        if params['label_smoothing_prob'] > 0:
            model.name += '_labelsmooth' + str(params['label_smoothing_prob'])
        if params['weight_noise_std'] != 0:
            model.name += '_noise' + str(params['weight_noise_std'])
        if bool(params['encoder_residual']):
            model.name += '_encres'
        elif bool(params['encoder_dense_residual']):
            model.name += '_encdenseres'
        if bool(params['decoder_residual']):
            model.name += '_decres'
        elif bool(params['decoder_dense_residual']):
            model.name += '_decdenseres'
        model.name += '_input' + str(model.input_size)

    elif params['model_type'] == 'hierarchical_ctc':
        if 'activation' not in params.keys():
            params['activation'] = 'relu'

        if backend == 'pytorch':
            from models.pytorch.ctc.hierarchical_ctc import HierarchicalCTC
        elif backend == 'chainer':
            from models.chainer.ctc.hierarchical_ctc import HierarchicalCTC
        else:
            raise TypeError

        model = HierarchicalCTC(
            input_size=params['input_channel'] *
            (1 + int(params['use_delta'] + int(params['use_double_delta']))),
            encoder_type=params['encoder_type'],
            encoder_bidirectional=params['encoder_bidirectional'],
            encoder_num_units=params['encoder_num_units'],
            encoder_num_proj=params['encoder_num_proj'],
            encoder_num_layers=params['encoder_num_layers'],
            num_layers_sub=params['num_layers_sub'],
            fc_list=params['fc_list'],
            dropout=params['dropout'],
            main_loss_weight=params['main_loss_weight'],
            num_classes=params['num_classes'],
            num_classes_sub=params['num_classes_sub'],
            parameter_init=params['parameter_init'],
            subsample_list=params['subsample_list'],
            subsample_type=params['subsample_type'],
            logits_temperature=params['logits_temperature'],
            num_stack=params['num_stack'],
            splice=params['splice'],
            conv_channels=params['conv_channels'],
            conv_kernel_sizes=params['conv_kernel_sizes'],
            conv_strides=params['conv_strides'],
            poolings=params['poolings'],
            activation=params['activation'],
            batch_norm=params['batch_norm'],
            label_smoothing_prob=params['label_smoothing_prob'],
            weight_noise_std=params['weight_noise_std'],
            residual=params['residual'],
            dense_residual=params['dense_residual'])

        model.name = params['encoder_type']
        if len(params['conv_channels']) != 0:
            if params['encoder_type'] in ['cnn', 'resnet']:
                for c in params['conv_channels']:
                    model.name += '_' + str(c)
            else:
                if params['encoder_bidirectional']:
                    model.name = 'b' + model.name
                name_tmp = model.name
                model.name = 'conv_'
                for c in params['conv_channels']:
                    model.name += str(c) + '_'
                model.name = model.name + name_tmp
        if bool(params['batch_norm']):
            model.name += '_bn'
        if params['encoder_type'] not in ['cnn', 'resnet']:
            model.name += str(params['encoder_num_units']) + 'H'
            model.name += str(params['encoder_num_layers']) + 'L'
            model.name += str(params['num_layers_sub']) + 'L'
            if params['encoder_num_proj'] != 0:
                model.name += '_proj' + str(params['encoder_num_proj'])
            if sum(params['subsample_list']) > 0:
                model.name += '_' + params['subsample_type'] + \
                    str(2 ** sum(params['subsample_list']))
            if params['num_stack'] != 1:
                model.name += '_stack' + str(params['num_stack'])
        if len(params['fc_list']) != 0:
            model.name += '_fc'
            for l in params['fc_list']:
                model.name += '_' + str(l)
        model.name += '_' + params['optimizer']
        model.name += '_lr' + str(params['learning_rate'])
        if params['dropout'] != 0:
            model.name += '_drop' + str(params['dropout'])
        if params['logits_temperature'] != 1:
            model.name += '_temp' + str(params['logits_temperature'])
        if params['label_smoothing_prob'] > 0:
            model.name += '_labelsmooth' + str(params['label_smoothing_prob'])
        if params['weight_noise_std'] != 0:
            model.name += '_noise' + str(params['weight_noise_std'])
        if bool(params['residual']):
            model.name += '_res'
        if bool(params['dense_residual']):
            model.name += '_dense_res'
        model.name += '_main' + str(params['main_loss_weight'])

    elif params['model_type'] == 'hierarchical_attention':
        if 'curriculum_training' not in params.keys():
            params['curriculum_training'] = False
        # TODO: remove this

        if backend == 'pytorch':
            from models.pytorch.attention.hierarchical_attention_seq2seq import HierarchicalAttentionSeq2seq
        elif backend == 'chainer':
            from models.chainer.attention.hierarchical_attention_seq2seq import HierarchicalAttentionSeq2seq
        else:
            raise TypeError

        model = HierarchicalAttentionSeq2seq(
            input_size=params['input_channel'] *
            (1 + int(params['use_delta'] + int(params['use_double_delta']))),
            encoder_type=params['encoder_type'],
            encoder_bidirectional=params['encoder_bidirectional'],
            encoder_num_units=params['encoder_num_units'],
            encoder_num_proj=params['encoder_num_proj'],
            encoder_num_layers=params['encoder_num_layers'],
            encoder_num_layers_sub=params['encoder_num_layers_sub'],
            encoder_dropout=params['dropout_encoder'],
            attention_type=params['attention_type'],
            attention_dim=params['attention_dim'],
            decoder_type=params['decoder_type'],
            decoder_num_units=params['decoder_num_units'],
            decoder_num_layers=params['decoder_num_layers'],
            decoder_num_units_sub=params['decoder_num_units_sub'],
            decoder_num_layers_sub=params['decoder_num_layers_sub'],
            decoder_dropout=params['dropout_decoder'],
            embedding_dim=params['embedding_dim'],
            embedding_dim_sub=params['embedding_dim_sub'],
            main_loss_weight=params['main_loss_weight'],
            num_classes=params['num_classes'],
            num_classes_sub=params['num_classes_sub'],
            parameter_init=params['parameter_init'],
            subsample_list=params['subsample_list'],
            subsample_type=params['subsample_type'],
            init_dec_state=params['init_dec_state'],
            sharpening_factor=params['sharpening_factor'],
            logits_temperature=params['logits_temperature'],
            sigmoid_smoothing=params['sigmoid_smoothing'],
            coverage_weight=params['coverage_weight'],
            ctc_loss_weight_sub=params['ctc_loss_weight_sub'],
            attention_conv_num_channels=params['attention_conv_num_channels'],
            attention_conv_width=params['attention_conv_width'],
            num_stack=params['num_stack'],
            splice=params['splice'],
            conv_channels=params['conv_channels'],
            conv_kernel_sizes=params['conv_kernel_sizes'],
            conv_strides=params['conv_strides'],
            poolings=params['poolings'],
            batch_norm=params['batch_norm'],
            scheduled_sampling_prob=params['scheduled_sampling_prob'],
            scheduled_sampling_ramp_max_step=params['scheduled_sampling_ramp_max_step'],
            label_smoothing_prob=params['label_smoothing_prob'],
            weight_noise_std=params['weight_noise_std'],
            encoder_residual=params['encoder_residual'],
            encoder_dense_residual=params['encoder_dense_residual'],
            decoder_residual=params['decoder_residual'],
            decoder_dense_residual=params['decoder_dense_residual'],
            curriculum_training=params['curriculum_training'])

        model.name = model_name
        if params['encoder_type'] not in ['cnn', 'resnet']:
            model.name += str(params['encoder_num_units']) + 'H'
            model.name += str(params['encoder_num_layers']) + 'L'
            model.name += str(params['encoder_num_layers_sub']) + 'L'
            if params['encoder_num_proj'] != 0:
                model.name += '_proj' + str(params['encoder_num_proj'])
            if sum(params['subsample_list']) > 0:
                model.name += '_' + params['subsample_type'] + \
                    str(2 ** sum(params['subsample_list']))
            if params['num_stack'] != 1:
                model.name += '_stack' + str(params['num_stack'])
        model.name += '_' + params['decoder_type']
        model.name += str(params['decoder_num_units']) + 'H'
        model.name += str(params['decoder_num_layers']) + 'L'
        model.name += '_' + params['optimizer']
        model.name += '_lr' + str(params['learning_rate'])
        model.name += '_' + params['attention_type']
        if params['dropout_encoder'] != 0 or params['dropout_decoder'] != 0:
            model.name += '_drop'
            if params['dropout_encoder'] != 0:
                model.name += 'en' + str(params['dropout_encoder'])
            if params['dropout_decoder'] != 0:
                model.name += 'de' + str(params['dropout_decoder'])
        if params['sharpening_factor'] != 1:
            model.name += '_sharp' + str(params['sharpening_factor'])
        if params['logits_temperature'] != 1:
            model.name += '_temp' + str(params['logits_temperature'])
        if bool(params['sigmoid_smoothing']):
            model.name += '_smooth'
        if params['coverage_weight'] > 0:
            model.name += '_coverage' + str(params['coverage_weight'])
        if params['ctc_loss_weight_sub'] > 0:
            model.name += '_ctcsub' + str(params['ctc_loss_weight_sub'])
        if params['scheduled_sampling_prob'] > 0:
            model.name += '_scheduled' + str(params['scheduled_sampling_prob'])
        if params['label_smoothing_prob'] > 0:
            model.name += '_labelsmooth' + str(params['label_smoothing_prob'])
        if params['weight_noise_std'] != 0:
            model.name += '_noise' + str(params['weight_noise_std'])
        if bool(params['encoder_residual']):
            model.name += '_encres'
        elif bool(params['encoder_dense_residual']):
            model.name += '_encdenseres'
        if bool(params['decoder_residual']):
            model.name += '_decres'
        elif bool(params['decoder_dense_residual']):
            model.name += '_decdenseres'
        model.name += '_main' + str(params['main_loss_weight'])
        if bool(params['curriculum_training']):
            model.name += '_curriculum'

    elif params['model_type'] == 'nested_attention':

        if backend == 'pytorch':
            from models.pytorch.attention.nested_attention_seq2seq import NestedAttentionSeq2seq
        elif backend == 'chainer':
            raise NotImplementedError
        else:
            raise TypeError

        model = NestedAttentionSeq2seq(
            input_size=params['input_channel'] *
            (1 + int(params['use_delta'] + int(params['use_double_delta']))),
            encoder_type=params['encoder_type'],
            encoder_bidirectional=params['encoder_bidirectional'],
            encoder_num_units=params['encoder_num_units'],
            encoder_num_proj=params['encoder_num_proj'],
            encoder_num_layers=params['encoder_num_layers'],
            encoder_num_layers_sub=params['encoder_num_layers_sub'],
            encoder_dropout=params['dropout_encoder'],
            attention_type=params['attention_type'],
            attention_dim=params['attention_dim'],
            decoder_type=params['decoder_type'],
            decoder_num_units=params['decoder_num_units'],
            decoder_num_layers=params['decoder_num_layers'],
            decoder_num_units_sub=params['decoder_num_units_sub'],
            decoder_num_layers_sub=params['decoder_num_layers_sub'],
            decoder_dropout=params['dropout_decoder'],
            embedding_dim=params['embedding_dim'],
            embedding_dim_sub=params['embedding_dim_sub'],
            main_loss_weight=params['main_loss_weight'],
            num_classes=params['num_classes'],
            num_classes_sub=params['num_classes_sub'],
            parameter_init=params['parameter_init'],
            subsample_list=params['subsample_list'],
            subsample_type=params['subsample_type'],
            init_dec_state=params['init_dec_state'],
            sharpening_factor=params['sharpening_factor'],
            logits_temperature=params['logits_temperature'],
            sigmoid_smoothing=params['sigmoid_smoothing'],
            coverage_weight=params['coverage_weight'],
            ctc_loss_weight_sub=params['ctc_loss_weight_sub'],
            attention_conv_num_channels=params['attention_conv_num_channels'],
            attention_conv_width=params['attention_conv_width'],
            num_stack=params['num_stack'],
            splice=params['splice'],
            conv_channels=params['conv_channels'],
            conv_kernel_sizes=params['conv_kernel_sizes'],
            conv_strides=params['conv_strides'],
            poolings=params['poolings'],
            batch_norm=params['batch_norm'],
            scheduled_sampling_prob=params['scheduled_sampling_prob'],
            scheduled_sampling_ramp_max_step=params['scheduled_sampling_ramp_max_step'],
            label_smoothing_prob=params['label_smoothing_prob'],
            weight_noise_std=params['weight_noise_std'],
            composition_case=params['composition_case'],
            space_index=params['space_index'])

        model.name = model_name
        if params['encoder_type'] not in ['cnn', 'resnet']:
            model.name += str(params['encoder_num_units']) + 'H'
            model.name += str(params['encoder_num_layers']) + 'L'
            model.name += str(params['encoder_num_layers_sub']) + 'L'
            if params['encoder_num_proj'] != 0:
                model.name += '_proj' + str(params['encoder_num_proj'])
            if sum(params['subsample_list']) > 0:
                model.name += '_' + params['subsample_type'] + \
                    str(2 ** sum(params['subsample_list']))
            if params['num_stack'] != 1:
                model.name += '_stack' + str(params['num_stack'])
        model.name += '_' + params['decoder_type']
        model.name += str(params['decoder_num_units']) + 'H'
        model.name += str(params['decoder_num_layers']) + 'L'
        model.name += '_' + params['optimizer']
        model.name += '_lr' + str(params['learning_rate'])
        model.name += '_' + params['attention_type']
        if params['dropout_encoder'] != 0 or params['dropout_decoder'] != 0:
            model.name += '_drop'
            if params['dropout_encoder'] != 0:
                model.name += 'en' + str(params['dropout_encoder'])
            if params['dropout_decoder'] != 0:
                model.name += 'de' + str(params['dropout_decoder'])
        if params['sharpening_factor'] != 1:
            model.name += '_sharp' + str(params['sharpening_factor'])
        if params['logits_temperature'] != 1:
            model.name += '_temp' + str(params['logits_temperature'])
        if bool(params['sigmoid_smoothing']):
            model.name += '_smooth'
        if params['coverage_weight'] > 0:
            model.name += '_coverage' + str(params['coverage_weight'])
        if params['ctc_loss_weight_sub'] > 0:
            model.name += '_ctcsub' + str(params['ctc_loss_weight_sub'])
        if params['scheduled_sampling_prob'] > 0:
            model.name += '_scheduled' + str(params['scheduled_sampling_prob'])
        if params['label_smoothing_prob'] > 0:
            model.name += '_labelsmooth' + str(params['label_smoothing_prob'])
        if params['weight_noise_std'] != 0:
            model.name += '_noise' + str(params['weight_noise_std'])
        model.name += '_main' + str(params['main_loss_weight'])
        model.name += '_' + params['composition_case']

    return model
