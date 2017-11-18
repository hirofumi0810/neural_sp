#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.pytorch.ctc.ctc import CTC
from models.pytorch.attention.attention_seq2seq import AttentionSeq2seq
from models.pytorch.ctc.hierarchical_ctc import HierarchicalCTC
from models.pytorch.attention.hierarchical_attention_seq2seq import HierarchicalAttentionSeq2seq


def load(model_type, params):
    """Load an encoder.
    Args:
        model_type (string): ctc or attention or hierarchical_ctc or
            hierarchical_attention or joint_ctc_attention
    Returns:
        model (nn.Module): An encoder class
    """

    if model_type == 'ctc':
        model = CTC(
            input_size=params['input_size'],
            num_stack=params['num_stack'],
            splice=params['splice'],
            encoder_type=params['encoder_type'],
            bidirectional=params['bidirectional'],
            num_units=params['num_units'],
            num_proj=params['num_proj'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            num_classes=params['num_classes'],
            parameter_init=params['parameter_init'],
            logits_temperature=params['logits_temperature'])

        model.name = params['encoder_type']
        if params['bidirectional']:
            model.name = 'b' + model.name
        model.name += '_' + str(params['num_units']) + 'H'
        model.name += '_' + str(params['num_layers']) + 'L'
        model.name += '_' + params['optimizer']
        model.name += '_lr' + str(params['learning_rate'])
        if params['num_proj'] != 0:
            model.name += '_proj' + str(params['num_proj'])
        if params['dropout'] != 0:
            model.name += '_drop' + str(params['dropout'])
        if params['num_stack'] != 1:
            model.name += '_stack' + str(params['num_stack'])
        if params['weight_decay'] != 0:
            model.name += '_wd' + str(params['weight_decay'])
        if params['bottleneck_dim'] != 0:
            model.name += '_bottle' + str(params['bottleneck_dim'])
        if params['logits_temperature'] != 1:
            model.name += '_temp' + str(params['logits_temperature'])

    elif model_type == 'attention':
        model = AttentionSeq2seq(
            input_size=params['input_size'],
            num_stack=params['num_stack'],
            splice=params['splice'],
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
            decoder_num_proj=params['decoder_num_proj'],
            decoder_num_layers=params['decoder_num_layers'],
            decoder_dropout=params['dropout_decoder'],
            embedding_dim=params['embedding_dim'],
            embedding_dropout=params['dropout_embedding'],
            num_classes=params['num_classes'],
            parameter_init=params['parameter_init'],
            subsample_list=params['subsample_list'],
            init_dec_state_with_enc_state=params['init_dec_state_with_enc_state'],
            sharpening_factor=params['sharpening_factor'],
            logits_temperature=params['logits_temperature'],
            sigmoid_smoothing=params['sigmoid_smoothing'],
            input_feeding_approach=params['input_feeding_approach'],
            coverage_weight=params['coverage_weight'],
            ctc_loss_weight=params['ctc_loss_weight'])

        model.name = params['encoder_type']
        if params['encoder_bidirectional']:
            model.name = 'b' + model.name
        model.name += '_' + str(params['encoder_num_units']) + 'H'
        model.name += '_' + str(params['encoder_num_layers']) + 'L'
        model.name += '_' + params['decoder_type']
        model.name += '_' + str(params['decoder_num_units']) + 'H'
        model.name += '_' + str(params['decoder_num_layers']) + 'L'
        model.name += '_' + params['optimizer']
        model.name += '_lr' + str(params['learning_rate'])
        model.name += '_' + params['attention_type']
        if params['dropout_encoder'] != 0:
            model.name += '_dropen' + str(params['dropout_encoder'])
        if params['dropout_decoder'] != 0:
            model.name += '_dropde' + str(params['dropout_decoder'])
        if params['dropout_embedding'] != 0:
            model.name += '_dropemb' + str(params['dropout_embedding'])
        if params['num_stack'] != 1:
            model.name += '_stack' + str(params['num_stack'])
        if params['weight_decay'] != 0:
            model.name += 'wd' + str(params['weight_decay'])
        if params['sharpening_factor'] != 1:
            model.name += '_sharp' + str(params['sharpening_factor'])
        if params['logits_temperature'] != 1:
            model.name += '_temp' + str(params['logits_temperature'])
        if bool(params['sigmoid_smoothing']):
            model.name += '_smooth'
        if bool(params['input_feeding_approach']):
            model.name += '_infeed'
        if params['coverage_weight'] > 0:
            model.name += '_coverage' + str(params['coverage_weight'])
        if params['ctc_loss_weight'] > 0:
            model.name += '_ctc' + str(params['ctc_loss_weight'])

    if params['model_type'] == 'hierarchical_ctc':
        model = HierarchicalCTC(
            input_size=params['input_size'],
            num_stack=params['num_stack'],
            splice=params['splice'],
            encoder_type=params['encoder_type'],
            bidirectional=params['bidirectional'],
            num_units=params['num_units'],
            num_proj=params['num_proj'],
            num_layers=params['num_layers'],
            num_layers_sub=params['num_layers_sub'],
            dropout=params['dropout'],
            main_loss_weight=params['main_loss_weight'],
            num_classes=params['num_classes'],
            num_classes_sub=params['num_classes_sub'],
            parameter_init=params['parameter_init'],
            logits_temperature=params['logits_temperature'])

        model.name = params['encoder_type']
        if params['bidirectional']:
            model.name = 'b' + model.name
        model.name += '_' + str(params['num_units']) + 'H'
        model.name += '_' + str(params['num_layers']) + 'L'
        model.name += '_' + str(params['num_layers_sub']) + 'L'
        model.name += '_' + params['optimizer']
        model.name += '_lr' + str(params['learning_rate'])
        if params['num_proj'] != 0:
            model.name += '_proj' + str(params['num_proj'])
        if params['dropout'] != 0:
            model.name += '_drop' + str(params['dropout'])
        if params['num_stack'] != 1:
            model.name += '_stack' + str(params['num_stack'])
        if params['weight_decay'] != 0:
            model.name += '_wd' + str(params['weight_decay'])
        if params['bottleneck_dim'] != 0:
            model.name += '_bottle' + str(params['bottleneck_dim'])
        if params['logits_temperature'] != 1:
            model.name += '_temp' + str(params['logits_temperature'])
        model.name += '_main' + str(params['main_loss_weight'])

    elif params['model_type'] == 'hierarchical_attention':
        model = HierarchicalAttentionSeq2seq(
            input_size=params['input_size'],
            num_stack=params['num_stack'],
            splice=params['splice'],
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
            decoder_num_proj=params['decoder_num_proj'],
            decoder_num_layers=params['decoder_num_layers'],
            decoder_num_units_sub=params['decoder_num_units_sub'],
            decoder_num_proj_sub=params['decoder_num_proj_sub'],
            decoder_num_layers_sub=params['decoder_num_layers_sub'],
            decoder_dropout=params['dropout_decoder'],
            embedding_dim=params['embedding_dim'],
            embedding_dim_sub=params['embedding_dim_sub'],
            embedding_dropout=params['dropout_embedding'],
            main_loss_weight=params['main_loss_weight'],
            num_classes=params['num_classes'],
            num_classes_sub=params['num_classes_sub'],
            parameter_init=params['parameter_init'],
            subsample_list=params['subsample_list'],
            init_dec_state_with_enc_state=params['init_dec_state_with_enc_state'],
            sharpening_factor=params['sharpening_factor'],
            logits_temperature=params['logits_temperature'],
            sigmoid_smoothing=params['sigmoid_smoothing'],
            input_feeding_approach=params['input_feeding_approach'],
            coverage_weight=params['coverage_weight'])

        model.name = params['encoder_type']
        if params['encoder_bidirectional']:
            model.name = 'b' + model.name
        model.name += '_' + str(params['encoder_num_units']) + 'H'
        model.name += '_' + str(params['encoder_num_layers']) + 'L'
        model.name += '_' + str(params['encoder_num_layers_sub']) + 'L'
        model.name += '_' + params['decoder_type']
        model.name += '_' + str(params['decoder_num_units']) + 'H'
        model.name += '_' + str(params['decoder_num_layers']) + 'L'
        model.name += '_' + params['optimizer']
        model.name += '_lr' + str(params['learning_rate'])
        model.name += '_' + params['attention_type']
        if params['dropout_encoder'] != 0:
            model.name += '_dropen' + str(params['dropout_encoder'])
        if params['dropout_decoder'] != 0:
            model.name += '_dropde' + str(params['dropout_decoder'])
        if params['dropout_embedding'] != 0:
            model.name += '_dropemb' + str(params['dropout_embedding'])
        if params['num_stack'] != 1:
            model.name += '_stack' + str(params['num_stack'])
        if params['weight_decay'] != 0:
            model.name += 'wd' + str(params['weight_decay'])
        if params['sharpening_factor'] != 1:
            model.name += '_sharp' + str(params['sharpening_factor'])
        if params['logits_temperature'] != 1:
            model.name += '_temp' + str(params['logits_temperature'])
        if bool(params['sigmoid_smoothing']):
            model.name += '_smooth'
        if bool(params['input_feeding_approach']):
            model.name += '_infeed'
        model.name += '_main' + str(params['main_loss_weight'])
        if params['coverage_weight'] > 0:
            model.name += '_coverage' + str(params['coverage_weight'])
        # if params['ctc_loss_weight'] > 0:
        #     model.name += '_ctc' + str(params['ctc_loss_weight'])

    return model
