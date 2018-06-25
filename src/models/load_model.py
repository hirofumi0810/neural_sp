#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import isdir


def load(model_type, config, backend):
    """Load an encoder.
    Args:
        model_type (string): ctc or student_ctc or attention or
            hierarchical_ctc or hierarchical_attention or nested_attention
        config (dict): dict of hyperparameters
        backend (string): pytorch or chainer
    Returns:
        model (nn.Module): An encoder class
    """
    if model_type != 'rnnlm':
        model_name = config['encoder_type']
        if config['encoder_type'] in ['cnn', 'resnet']:
            for c in config['conv_channels']:
                model_name += '_' + str(c)
        else:
            if config['encoder_bidirectional']:
                model_name = 'b' + model_name
            if len(config['conv_channels']) != 0:
                name_tmp = model_name
                model_name = 'conv_'
                for c in config['conv_channels']:
                    model_name += str(c) + '_'
                model_name += name_tmp

    if model_type == 'ctc':
        if backend == 'pytorch':
            from src.models.pytorch_v3.ctc.ctc import CTC
        elif backend == 'chainer':
            from src.models.chainer.ctc.ctc import CTC

        model = CTC(
            input_size=config['input_freq'] *
            (1 + int(config['use_delta'] + int(config['use_double_delta']))),
            encoder_type=config['encoder_type'],
            encoder_bidirectional=config['encoder_bidirectional'],
            encoder_num_units=config['encoder_num_units'],
            encoder_num_proj=config['encoder_num_proj'],
            encoder_num_layers=config['encoder_num_layers'],
            fc_list=config['fc_list'],
            dropout_input=config['dropout_input'],
            dropout_encoder=config['dropout_encoder'],
            num_classes=config['num_classes'],
            parameter_init_distribution=config['parameter_init_distribution'],
            parameter_init=config['parameter_init'],
            recurrent_weight_orthogonal=config['recurrent_weight_orthogonal'],
            init_forget_gate_bias_with_one=config['init_forget_gate_bias_with_one'],
            subsample_list=config['subsample_list'],
            subsample_type=config['subsample_type'],
            logits_temperature=config['logits_temperature'],
            num_stack=config['num_stack'],
            num_skip=config['num_skip'],
            splice=config['splice'],
            input_channel=config['input_channel'],
            conv_channels=config['conv_channels'],
            conv_kernel_sizes=config['conv_kernel_sizes'],
            conv_strides=config['conv_strides'],
            poolings=config['poolings'],
            activation=config['activation'],
            batch_norm=config['batch_norm'],
            label_smoothing_prob=config['label_smoothing_prob'],
            weight_noise_std=config['weight_noise_std'],
            encoder_residual=config['encoder_residual'],
            encoder_dense_residual=config['encoder_dense_residual'])

        model.name = model_name
        if config['encoder_type'] not in ['cnn', 'resnet']:
            model.name += str(config['encoder_num_units']) + 'H'
            model.name += str(config['encoder_num_layers']) + 'L'
            if config['encoder_num_proj'] != 0:
                model.name += '_proj' + str(config['encoder_num_proj'])
            if sum(config['subsample_list']) > 0:
                model.name += '_' + config['subsample_type'] + \
                    str(2 ** sum(config['subsample_list']))
            if config['num_stack'] != 1:
                model.name += '_stack' + str(config['num_stack'])
        if len(config['fc_list']) != 0:
            model.name += '_fc'
            for l in config['fc_list']:
                model.name += '_' + str(l)
        if bool(config['batch_norm']):
            model.name += '_bn'
        model.name += '_' + config['optimizer']
        model.name += '_lr' + str(config['learning_rate'])
        if config['dropout_encoder'] != 0:
            model.name += '_drop'
            if config['dropout_input'] != 0:
                model.name += 'in' + str(config['dropout_input'])
            model.name += 'en' + str(config['dropout_encoder'])
        if config['logits_temperature'] != 1:
            model.name += '_temp' + str(config['logits_temperature'])

        if config['weight_noise_std'] != 0:
            model.name += '_noise' + str(config['weight_noise_std'])
        if config['encoder_type'] == 'cnn':
            model.name += '_' + config['activation']
        if bool(config['encoder_residual']):
            model.name += '_res'
        if bool(config['encoder_dense_residual']):
            model.name += '_dense_res'
        model.name += '_input' + str(model.input_size)
        if isdir(config['pretrained_model_path']):
            model.name += '_pretrain'

    elif config['model_type'] == 'hierarchical_ctc':
        if backend == 'pytorch':
            from src.models.pytorch_v3.ctc.hierarchical_ctc import HierarchicalCTC
        elif backend == 'chainer':
            from src.models.chainer.ctc.hierarchical_ctc import HierarchicalCTC

        model = HierarchicalCTC(
            input_size=config['input_freq'] *
            (1 + int(config['use_delta'] + int(config['use_double_delta']))),
            encoder_type=config['encoder_type'],
            encoder_bidirectional=config['encoder_bidirectional'],
            encoder_num_units=config['encoder_num_units'],
            encoder_num_proj=config['encoder_num_proj'],
            encoder_num_layers=config['encoder_num_layers'],
            encoder_num_layers_sub=config['encoder_num_layers_sub'],
            fc_list=config['fc_list'],
            fc_list_sub=config['fc_list_sub'],
            dropout_input=config['dropout_input'],
            dropout_encoder=config['dropout_encoder'],
            main_loss_weight=config['main_loss_weight'],
            sub_loss_weight=config['sub_loss_weight'],
            num_classes=config['num_classes'],
            num_classes_sub=config['num_classes_sub'],
            parameter_init_distribution=config['parameter_init_distribution'],
            parameter_init=config['parameter_init'],
            recurrent_weight_orthogonal=config['recurrent_weight_orthogonal'],
            init_forget_gate_bias_with_one=config['init_forget_gate_bias_with_one'],
            subsample_list=config['subsample_list'],
            subsample_type=config['subsample_type'],
            logits_temperature=config['logits_temperature'],
            num_stack=config['num_stack'],
            num_skip=config['num_skip'],
            splice=config['splice'],
            input_channel=config['input_channel'],
            conv_channels=config['conv_channels'],
            conv_kernel_sizes=config['conv_kernel_sizes'],
            conv_strides=config['conv_strides'],
            poolings=config['poolings'],
            activation=config['activation'],
            batch_norm=config['batch_norm'],
            label_smoothing_prob=config['label_smoothing_prob'],
            weight_noise_std=config['weight_noise_std'],
            encoder_residual=config['encoder_residual'],
            encoder_dense_residual=config['encoder_dense_residual'])

        model.name = model_name
        if config['encoder_type'] not in ['cnn', 'resnet']:
            model.name += str(config['encoder_num_units']) + 'H'
            model.name += str(config['encoder_num_layers']) + 'L'
            model.name += str(config['encoder_num_layers_sub']) + 'L'
            if config['encoder_num_proj'] != 0:
                model.name += '_proj' + str(config['encoder_num_proj'])
            if sum(config['subsample_list']) > 0:
                model.name += '_' + config['subsample_type'] + \
                    str(2 ** sum(config['subsample_list']))
            if config['num_stack'] != 1:
                model.name += '_stack' + str(config['num_stack'])
        if len(config['fc_list']) != 0:
            model.name += '_fc'
            for l in config['fc_list']:
                model.name += '_' + str(l)
        if bool(config['batch_norm']):
            model.name += '_bn'
        model.name += '_' + config['optimizer']
        model.name += '_lr' + str(config['learning_rate'])
        if config['dropout_encoder'] != 0:
            model.name += '_drop'
            if config['dropout_input'] != 0:
                model.name += 'in' + str(config['dropout_input'])
            model.name += 'en' + str(config['dropout_encoder'])
        if config['logits_temperature'] != 1:
            model.name += '_temp' + str(config['logits_temperature'])
        if config['label_smoothing_prob'] > 0:
            model.name += '_ls' + str(config['label_smoothing_prob'])
        if config['weight_noise_std'] != 0:
            model.name += '_noise' + str(config['weight_noise_std'])
        if bool(config['encoder_residual']):
            model.name += '_res'
        if bool(config['encoder_dense_residual']):
            model.name += '_dense_res'
        model.name += '_main' + str(config['main_loss_weight'])
        model.name += '_sub' + str(config['sub_loss_weight'])
        model.name += '_input' + str(model.input_size)
        if isdir(config['pretrained_model_path']):
            model.name += '_pretrain'

    elif model_type == 'attention':
        if backend == 'pytorch':
            from src.models.pytorch_v3.attention.attention_seq2seq import AttentionSeq2seq
        elif backend == 'chainer':
            from src.models.chainer.attention.attention_seq2seq import AttentionSeq2seq

        # TODO: remove these later
        if 'rnnlm_fusion_type' not in config.keys():
            config['rnnlm_fusion_type'] = False
        if 'rnnlm_config' not in config.keys():
            config['rnnlm_config'] = None
        if 'rnnlm_weight' not in config.keys():
            config['rnnlm_weight'] = 0
        if 'concat_embedding' not in config.keys():
            config['concat_embedding'] = False

        model = AttentionSeq2seq(
            input_size=config['input_freq'] *
            (1 + int(config['use_delta'] + int(config['use_double_delta']))),
            encoder_type=config['encoder_type'],
            encoder_bidirectional=config['encoder_bidirectional'],
            encoder_num_units=config['encoder_num_units'],
            encoder_num_proj=config['encoder_num_proj'],
            encoder_num_layers=config['encoder_num_layers'],
            attention_type=config['attention_type'],
            attention_dim=config['attention_dim'],
            decoder_type=config['decoder_type'],
            decoder_num_units=config['decoder_num_units'],
            decoder_num_layers=config['decoder_num_layers'],
            embedding_dim=config['embedding_dim'],
            dropout_input=config['dropout_input'],
            dropout_encoder=config['dropout_encoder'],
            dropout_decoder=config['dropout_decoder'],
            dropout_embedding=config['dropout_embedding'],
            num_classes=config['num_classes'],
            parameter_init_distribution=config['parameter_init_distribution'],
            parameter_init=config['parameter_init'],
            recurrent_weight_orthogonal=config['recurrent_weight_orthogonal'],
            init_forget_gate_bias_with_one=config['init_forget_gate_bias_with_one'],
            subsample_list=config['subsample_list'],
            subsample_type=config['subsample_type'],
            bridge_layer=config['bridge_layer'],
            init_dec_state=config['init_dec_state'],
            sharpening_factor=config['sharpening_factor'],
            logits_temperature=config['logits_temperature'],
            sigmoid_smoothing=config['sigmoid_smoothing'],
            coverage_weight=config['coverage_weight'],
            ctc_loss_weight=config['ctc_loss_weight'],
            attention_conv_num_channels=config['attention_conv_num_channels'],
            attention_conv_width=config['attention_conv_width'],
            num_stack=config['num_stack'],
            num_skip=config['num_skip'],
            splice=config['splice'],
            input_channel=config['input_channel'],
            conv_channels=config['conv_channels'],
            conv_kernel_sizes=config['conv_kernel_sizes'],
            conv_strides=config['conv_strides'],
            poolings=config['poolings'],
            batch_norm=config['batch_norm'],
            scheduled_sampling_prob=config['scheduled_sampling_prob'],
            scheduled_sampling_max_step=config['scheduled_sampling_max_step'],
            label_smoothing_prob=config['label_smoothing_prob'],
            weight_noise_std=config['weight_noise_std'],
            encoder_residual=config['encoder_residual'],
            encoder_dense_residual=config['encoder_dense_residual'],
            decoder_residual=config['decoder_residual'],
            decoder_dense_residual=config['decoder_dense_residual'],
            decoding_order=config['decoding_order'],
            bottleneck_dim=config['bottleneck_dim'],
            backward_loss_weight=config['backward_loss_weight'],
            num_heads=config['num_heads'],
            rnnlm_fusion_type=config['rnnlm_fusion_type'],
            rnnlm_config=config['rnnlm_config'],
            rnnlm_weight=config['rnnlm_weight'],
            concat_embedding=config['concat_embedding'])

        model.name = model_name
        if config['encoder_type'] not in ['cnn', 'resnet']:
            model.name += str(config['encoder_num_units']) + 'H'
            model.name += str(config['encoder_num_layers']) + 'L'
            if config['encoder_num_proj'] != 0:
                model.name += '_proj' + str(config['encoder_num_proj'])
            if sum(config['subsample_list']) > 0:
                model.name += '_' + config['subsample_type'] + \
                    str(2 ** sum(config['subsample_list']))
            if config['num_stack'] != 1:
                model.name += '_stack' + str(config['num_stack'])
        if bool(config['batch_norm']):
            model.name += '_bn'
        model.name += '_' + config['decoder_type']
        model.name += str(config['decoder_num_units']) + 'H'
        model.name += str(config['decoder_num_layers']) + 'L'
        model.name += '_' + config['optimizer']
        model.name += '_lr' + str(config['learning_rate'])
        model.name += '_' + config['attention_type']
        if config['bottleneck_dim'] != config['decoder_num_units']:
            model.name += '_fc' + str(config['bottleneck_dim'])
        if config['dropout_encoder'] != 0 or config['dropout_decoder'] != 0:
            model.name += '_drop'
            if config['dropout_input'] != 0:
                model.name += 'in' + str(config['dropout_input'])
            if config['dropout_encoder'] != 0:
                model.name += 'en' + str(config['dropout_encoder'])
            if config['dropout_decoder'] != 0:
                model.name += 'de' + str(config['dropout_decoder'])
            if config['dropout_embedding'] != 0:
                model.name += 'emb' + str(config['dropout_embedding'])
        if config['sharpening_factor'] != 1:
            model.name += '_sharp' + str(config['sharpening_factor'])
        if config['logits_temperature'] != 1:
            model.name += '_temp' + str(config['logits_temperature'])
        if bool(config['sigmoid_smoothing']):
            model.name += '_sigsmooth'
        if config['coverage_weight'] > 0:
            model.name += '_coverage' + str(config['coverage_weight'])
        if config['ctc_loss_weight'] > 0:
            model.name += '_ctc' + str(config['ctc_loss_weight'])
        if config['scheduled_sampling_prob'] > 0:
            model.name += '_ss' + str(config['scheduled_sampling_prob'])
        if config['label_smoothing_prob'] > 0:
            model.name += '_ls' + str(config['label_smoothing_prob'])
        if config['weight_noise_std'] != 0:
            model.name += '_noise' + str(config['weight_noise_std'])
        if bool(config['encoder_residual']):
            model.name += '_encres'
        elif bool(config['encoder_dense_residual']):
            model.name += '_encdense'
        if bool(config['decoder_residual']):
            model.name += '_decres'
        elif bool(config['decoder_dense_residual']):
            model.name += '_decdense'
        model.name += '_input' + str(model.input_size)
        model.name += '_' + config['decoding_order']
        if isdir(config['pretrained_model_path']):
            model.name += '_pretrain'
        if float(config['backward_loss_weight']) > 0:
            model.name += '_bwd' + str(config['backward_loss_weight'])
        if int(config['num_heads']) > 1:
            model.name += '_head' + str(config['num_heads'])
        if config['rnnlm_fusion_type']:
            model.name += '_' + config['rnnlm_fusion_type']
        if float(config['rnnlm_weight']) > 0:
            model.name += '_lmjoint' + str(config['rnnlm_weight'])
        if bool(config['concat_embedding']) > 0:
            model.name += '_concatemb'

    elif config['model_type'] == 'hierarchical_attention':
        if backend == 'pytorch':
            from src.models.pytorch_v3.attention.hierarchical_attention_seq2seq import HierarchicalAttentionSeq2seq
        elif backend == 'chainer':
            from src.models.chainer.attention.hierarchical_attention_seq2seq import HierarchicalAttentionSeq2seq

        # TODO: remove these later
        if 'rnnlm_fusion_type' not in config.keys():
            config['rnnlm_fusion_type'] = False
        if 'rnnlm_config' not in config.keys():
            config['rnnlm_config'] = None
        if 'rnnlm_config_sub' not in config.keys():
            config['rnnlm_config_sub'] = None
        if 'rnnlm_weight' not in config.keys():
            config['rnnlm_weight'] = 0
        if 'rnnlm_weight_sub' not in config.keys():
            config['rnnlm_weight_sub'] = 0
        if 'concat_embedding' not in config.keys():
            config['concat_embedding'] = False

        model = HierarchicalAttentionSeq2seq(
            input_size=config['input_freq'] *
            (1 + int(config['use_delta'] + int(config['use_double_delta']))),
            encoder_type=config['encoder_type'],
            encoder_bidirectional=config['encoder_bidirectional'],
            encoder_num_units=config['encoder_num_units'],
            encoder_num_proj=config['encoder_num_proj'],
            encoder_num_layers=config['encoder_num_layers'],
            encoder_num_layers_sub=config['encoder_num_layers_sub'],
            attention_type=config['attention_type'],
            attention_dim=config['attention_dim'],
            decoder_type=config['decoder_type'],
            decoder_num_units=config['decoder_num_units'],
            decoder_num_layers=config['decoder_num_layers'],
            decoder_num_units_sub=config['decoder_num_units_sub'],
            decoder_num_layers_sub=config['decoder_num_layers_sub'],
            embedding_dim=config['embedding_dim'],
            embedding_dim_sub=config['embedding_dim_sub'],
            dropout_input=config['dropout_input'],
            dropout_encoder=config['dropout_encoder'],
            dropout_decoder=config['dropout_decoder'],
            dropout_embedding=config['dropout_embedding'],
            main_loss_weight=config['main_loss_weight'],
            sub_loss_weight=config['sub_loss_weight'],
            num_classes=config['num_classes'],
            num_classes_sub=config['num_classes_sub'],
            parameter_init_distribution=config['parameter_init_distribution'],
            parameter_init=config['parameter_init'],
            recurrent_weight_orthogonal=config['recurrent_weight_orthogonal'],
            init_forget_gate_bias_with_one=config['init_forget_gate_bias_with_one'],
            subsample_list=config['subsample_list'],
            subsample_type=config['subsample_type'],
            bridge_layer=config['bridge_layer'],
            init_dec_state=config['init_dec_state'],
            sharpening_factor=config['sharpening_factor'],
            logits_temperature=config['logits_temperature'],
            sigmoid_smoothing=config['sigmoid_smoothing'],
            coverage_weight=config['coverage_weight'],
            ctc_loss_weight_sub=config['ctc_loss_weight_sub'],
            attention_conv_num_channels=config['attention_conv_num_channels'],
            attention_conv_width=config['attention_conv_width'],
            num_stack=config['num_stack'],
            num_skip=config['num_skip'],
            splice=config['splice'],
            input_channel=config['input_channel'],
            conv_channels=config['conv_channels'],
            conv_kernel_sizes=config['conv_kernel_sizes'],
            conv_strides=config['conv_strides'],
            poolings=config['poolings'],
            batch_norm=config['batch_norm'],
            scheduled_sampling_prob=config['scheduled_sampling_prob'],
            scheduled_sampling_max_step=config['scheduled_sampling_max_step'],
            label_smoothing_prob=config['label_smoothing_prob'],
            weight_noise_std=config['weight_noise_std'],
            encoder_residual=config['encoder_residual'],
            encoder_dense_residual=config['encoder_dense_residual'],
            decoder_residual=config['decoder_residual'],
            decoder_dense_residual=config['decoder_dense_residual'],
            decoding_order=config['decoding_order'],
            bottleneck_dim=config['bottleneck_dim'],
            bottleneck_dim_sub=config['bottleneck_dim_sub'],
            backward_sub=config['backward_sub'],
            num_heads=config['num_heads'],
            num_heads_sub=config['num_heads_sub'],
            rnnlm_fusion_type=config['rnnlm_fusion_type'],
            rnnlm_config=config['rnnlm_config'],
            rnnlm_config_sub=config['rnnlm_config_sub'],
            rnnlm_weight=config['rnnlm_weight'],
            rnnlm_weight_sub=config['rnnlm_weight_sub'],
            concat_embedding=config['concat_embedding'])

        model.name = model_name
        if config['encoder_type'] not in ['cnn', 'resnet']:
            model.name += str(config['encoder_num_units']) + 'H'
            model.name += str(config['encoder_num_layers']) + 'L'
            model.name += str(config['encoder_num_layers_sub']) + 'L'
            if config['encoder_num_proj'] != 0:
                model.name += '_proj' + str(config['encoder_num_proj'])
            if sum(config['subsample_list']) > 0:
                model.name += '_' + config['subsample_type'] + \
                    str(2 ** sum(config['subsample_list']))
            if config['num_stack'] != 1:
                model.name += '_stack' + str(config['num_stack'])
        if bool(config['batch_norm']):
            model.name += '_bn'
        model.name += '_' + config['decoder_type']
        model.name += str(config['decoder_num_units']) + 'H'
        model.name += str(config['decoder_num_layers']) + 'L'
        model.name += '_' + config['optimizer']
        model.name += '_lr' + str(config['learning_rate'])
        model.name += '_' + config['attention_type']
        if config['bottleneck_dim'] != config['decoder_num_units']:
            model.name += '_fc' + str(config['bottleneck_dim'])
        if config['dropout_encoder'] != 0 or config['dropout_decoder'] != 0:
            model.name += '_drop'
            if config['dropout_input'] != 0:
                model.name += 'in' + str(config['dropout_input'])
            if config['dropout_encoder'] != 0:
                model.name += 'en' + str(config['dropout_encoder'])
            if config['dropout_decoder'] != 0:
                model.name += 'de' + str(config['dropout_decoder'])
            if config['dropout_embedding'] != 0:
                model.name += 'emb' + str(config['dropout_embedding'])
        if config['sharpening_factor'] != 1:
            model.name += '_sharp' + str(config['sharpening_factor'])
        if config['logits_temperature'] != 1:
            model.name += '_temp' + str(config['logits_temperature'])
        if bool(config['sigmoid_smoothing']):
            model.name += '_sigsmooth'
        if config['coverage_weight'] > 0:
            model.name += '_coverage' + str(config['coverage_weight'])
        if config['scheduled_sampling_prob'] > 0:
            model.name += '_ss' + str(config['scheduled_sampling_prob'])
        if config['label_smoothing_prob'] > 0:
            model.name += '_ls' + str(config['label_smoothing_prob'])
        if config['weight_noise_std'] != 0:
            model.name += '_noise' + str(config['weight_noise_std'])
        if bool(config['encoder_residual']):
            model.name += '_encres'
        elif bool(config['encoder_dense_residual']):
            model.name += '_encdense'
        if bool(config['decoder_residual']):
            model.name += '_decres'
        elif bool(config['decoder_dense_residual']):
            model.name += '_decdense'
        model.name += '_main' + str(config['main_loss_weight'])
        model.name += '_sub' + str(config['sub_loss_weight'])
        if config['ctc_loss_weight_sub'] > 0:
            model.name += '_ctcsub' + str(config['ctc_loss_weight_sub'])
        model.name += '_input' + str(model.input_size)
        model.name += '_' + config['decoding_order']
        if isdir(config['pretrained_model_path']):
            model.name += '_pretrain'
        if bool(config['backward_sub']):
            model.name += '_bwdsub'
        if int(config['num_heads']) > 1:
            model.name += '_head' + str(config['num_heads'])
        if config['rnnlm_fusion_type']:
            model.name += '_' + config['rnnlm_fusion_type']
        if float(config['rnnlm_weight']) > 0 or float(config['rnnlm_weight_sub']) > 0:
            model.name += '_lmjoint' + \
                str(config['rnnlm_weight']) + str(config['rnnlm_weight_sub'])
        if bool(config['concat_embedding']) > 0:
            model.name += '_concatemb'

    elif config['model_type'] == 'nested_attention':
        if backend == 'pytorch':
            from src.models.pytorch_v3.attention.nested_attention_seq2seq import NestedAttentionSeq2seq
        elif backend == 'chainer':
            raise NotImplementedError

        model = NestedAttentionSeq2seq(
            input_size=config['input_freq'] *
            (1 + int(config['use_delta'] + int(config['use_double_delta']))),
            encoder_type=config['encoder_type'],
            encoder_bidirectional=config['encoder_bidirectional'],
            encoder_num_units=config['encoder_num_units'],
            encoder_num_proj=config['encoder_num_proj'],
            encoder_num_layers=config['encoder_num_layers'],
            encoder_num_layers_sub=config['encoder_num_layers_sub'],
            attention_type=config['attention_type'],
            attention_dim=config['attention_dim'],
            decoder_type=config['decoder_type'],
            decoder_num_units=config['decoder_num_units'],
            decoder_num_layers=config['decoder_num_layers'],
            decoder_num_units_sub=config['decoder_num_units_sub'],
            decoder_num_layers_sub=config['decoder_num_layers_sub'],
            embedding_dim=config['embedding_dim'],
            embedding_dim_sub=config['embedding_dim_sub'],
            dropout_input=config['dropout_input'],
            dropout_encoder=config['dropout_encoder'],
            dropout_decoder=config['dropout_decoder'],
            dropout_embedding=config['dropout_embedding'],
            main_loss_weight=config['main_loss_weight'],
            sub_loss_weight=config['sub_loss_weight'],
            num_classes=config['num_classes'],
            num_classes_sub=config['num_classes_sub'],
            parameter_init_distribution=config['parameter_init_distribution'],
            parameter_init=config['parameter_init'],
            recurrent_weight_orthogonal=config['recurrent_weight_orthogonal'],
            init_forget_gate_bias_with_one=config['init_forget_gate_bias_with_one'],
            subsample_list=config['subsample_list'],
            subsample_type=config['subsample_type'],
            bridge_layer=config['bridge_layer'],
            init_dec_state=config['init_dec_state'],
            sharpening_factor=config['sharpening_factor'],
            logits_temperature=config['logits_temperature'],
            sigmoid_smoothing=config['sigmoid_smoothing'],
            coverage_weight=config['coverage_weight'],
            ctc_loss_weight_sub=config['ctc_loss_weight_sub'],
            attention_conv_num_channels=config['attention_conv_num_channels'],
            attention_conv_width=config['attention_conv_width'],
            num_stack=config['num_stack'],
            num_skip=config['num_skip'],
            splice=config['splice'],
            input_channel=config['input_channel'],
            conv_channels=config['conv_channels'],
            conv_kernel_sizes=config['conv_kernel_sizes'],
            conv_strides=config['conv_strides'],
            poolings=config['poolings'],
            batch_norm=config['batch_norm'],
            scheduled_sampling_prob=config['scheduled_sampling_prob'],
            scheduled_sampling_max_step=config['scheduled_sampling_max_step'],
            label_smoothing_prob=config['label_smoothing_prob'],
            weight_noise_std=config['weight_noise_std'],
            encoder_residual=config['encoder_residual'],
            encoder_dense_residual=config['encoder_dense_residual'],
            decoder_residual=config['decoder_residual'],
            decoder_dense_residual=config['decoder_dense_residual'],
            decoding_order=config['decoding_order'],
            bottleneck_dim=config['bottleneck_dim'],
            bottleneck_dim_sub=config['bottleneck_dim_sub'],
            backward_sub=config['backward_sub'],
            num_heads=config['num_heads'],
            num_heads_sub=config['num_heads_sub'],
            num_heads_dec=config['num_heads_dec'],
            usage_dec_sub=config['usage_dec_sub'],
            att_reg_weight=config['att_reg_weight'],
            dec_attend_temperature=config['dec_attend_temperature'],
            dec_sigmoid_smoothing=config['dec_sigmoid_smoothing'],
            relax_context_vec_dec=config['relax_context_vec_dec'],
            dec_attention_type=config['dec_attention_type'],
            logits_injection=config['logits_injection'],
            gating=config['gating'])

        model.name = model_name
        if config['encoder_type'] not in ['cnn', 'resnet']:
            model.name += str(config['encoder_num_units']) + 'H'
            model.name += str(config['encoder_num_layers']) + 'L'
            model.name += str(config['encoder_num_layers_sub']) + 'L'
            if config['encoder_num_proj'] != 0:
                model.name += '_proj' + str(config['encoder_num_proj'])
            if sum(config['subsample_list']) > 0:
                model.name += '_' + config['subsample_type'] + \
                    str(2 ** sum(config['subsample_list']))
            if config['num_stack'] != 1:
                model.name += '_stack' + str(config['num_stack'])
        if bool(config['batch_norm']):
            model.name += '_bn'
        model.name += '_' + config['decoder_type']
        model.name += str(config['decoder_num_units']) + 'H'
        model.name += str(config['decoder_num_layers']) + 'L'
        model.name += '_' + config['optimizer']
        model.name += '_lr' + str(config['learning_rate'])
        model.name += '_' + config['attention_type']
        if config['bottleneck_dim'] != config['decoder_num_units']:
            model.name += '_fc' + str(config['bottleneck_dim'])
        if config['dropout_encoder'] != 0 or config['dropout_decoder'] != 0:
            model.name += '_drop'
            if config['dropout_input'] != 0:
                model.name += 'in' + str(config['dropout_input'])
            if config['dropout_encoder'] != 0:
                model.name += 'en' + str(config['dropout_encoder'])
            if config['dropout_decoder'] != 0:
                model.name += 'de' + str(config['dropout_decoder'])
            if config['dropout_embedding'] != 0:
                model.name += 'emb' + str(config['dropout_embedding'])
        if config['sharpening_factor'] != 1:
            model.name += '_sharp' + str(config['sharpening_factor'])
        if config['logits_temperature'] != 1:
            model.name += '_temp' + str(config['logits_temperature'])
        if bool(config['sigmoid_smoothing']):
            model.name += '_sigsmooth'
        if config['coverage_weight'] > 0:
            model.name += '_coverage' + str(config['coverage_weight'])
        if config['ctc_loss_weight_sub'] > 0:
            model.name += '_ctcsub' + str(config['ctc_loss_weight_sub'])
        if config['scheduled_sampling_prob'] > 0:
            model.name += '_ss' + str(config['scheduled_sampling_prob'])
        if config['label_smoothing_prob'] > 0:
            model.name += '_ls' + str(config['label_smoothing_prob'])
        if config['weight_noise_std'] != 0:
            model.name += '_noise' + str(config['weight_noise_std'])
        if bool(config['encoder_residual']):
            model.name += '_encres'
        elif bool(config['encoder_dense_residual']):
            model.name += '_encdense'
        if bool(config['decoder_residual']):
            model.name += '_decres'
        elif bool(config['decoder_dense_residual']):
            model.name += '_decdense'
        model.name += '_main' + str(config['main_loss_weight'])
        model.name += '_sub' + str(config['sub_loss_weight'])
        model.name += '_input' + str(model.input_size)
        model.name += '_' + config['decoding_order']
        if bool(config['backward_sub']):
            model.name += '_bwdsub'
        if int(config['num_heads']) > 1:
            model.name += '_head' + str(config['num_heads'])
        model.name += '_' + config['dec_attention_type']
        model.name += '_' + config['usage_dec_sub']
        if config['dec_attend_temperature'] != 1:
            model.name += '_temp' + \
                str(config['dec_attend_temperature'])
        if bool(config['dec_sigmoid_smoothing']):
            model.name += '_sigsmooth'
        if float(config['att_reg_weight']) > 0:
            model.name += '_attreg' + \
                str(config['att_reg_weight'])
        if bool(config['relax_context_vec_dec']):
            model.name += '_relax'
        if bool(config['logits_injection']):
            model.name += '_probinj'
        if bool(config['gating']):
            model.name += '_gate'
        if isdir(config['pretrained_model_path']):
            model.name += '_pretrain'

    elif model_type == 'rnnlm':
        if backend == 'pytorch':
            from src.models.pytorch_v3.lm.rnnlm import RNNLM
        elif backend == 'chainer':
            from src.models.chainer.lm.rnnlm import RNNLM

        # TODO: remove later
        if 'residual_connection' not in config.keys():
            config['residual_connection'] = False

        model = RNNLM(
            embedding_dim=config['embedding_dim'],
            rnn_type=config['rnn_type'],
            bidirectional=config['bidirectional'],
            num_units=config['num_units'],
            num_layers=config['num_layers'],
            dropout_embedding=config['dropout_embedding'],
            dropout_hidden=config['dropout_hidden'],
            dropout_output=config['dropout_output'],
            num_classes=config['num_classes'],
            parameter_init_distribution=config['parameter_init_distribution'],
            parameter_init=config['parameter_init'],
            recurrent_weight_orthogonal=config['recurrent_weight_orthogonal'],
            init_forget_gate_bias_with_one=config['init_forget_gate_bias_with_one'],
            label_smoothing_prob=config['label_smoothing_prob'],
            tie_weights=config['tie_weights'],
            residual_connection=config['residual_connection'],
            backward=config['backward'])

        model_name = config['rnn_type']
        if config['bidirectional']:
            model_name = 'b' + model_name
        model.name = model_name
        model.name += str(config['num_units']) + 'H'
        model.name += str(config['num_layers']) + 'L'
        # if config['num_proj'] != 0:
        #     model.name += '_proj' + str(config['num_proj'])
        model.name += 'emb' + str(config['embedding_dim'])
        model.name += '_' + config['optimizer']
        model.name += '_lr' + str(config['learning_rate'])
        model.name += '_drop'
        # if config['dropout_input'] != 0:
        #     model.name += 'in' + str(config['dropout_input'])
        if config['dropout_hidden'] != 0:
            model.name += 'hidden' + str(config['dropout_hidden'])
        if config['dropout_output'] != 0:
            model.name += 'out' + str(config['dropout_output'])
        if config['dropout_embedding'] != 0:
            model.name += 'emb' + str(config['dropout_embedding'])
        if config['label_smoothing_prob'] > 0:
            model.name += '_ls' + str(config['label_smoothing_prob'])
        if bool(config['tie_weights']):
            model.name += '_tie'
        if bool(config['residual_connection']):
            model.name += '_res'
        if bool(config['backward']):
            model.name += '_backward'
        if config['vocab']:
            model.name += '_' + config['vocab']

    return model
