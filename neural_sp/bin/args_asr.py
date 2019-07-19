#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Args option for the ASR task."""

import configargparse
from distutils.util import strtobool


def parse():
    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    parser.add('--config', is_config_file=True, help='config file path')
    # general
    parser.add_argument('--corpus', type=str,
                        help='corpus name')
    parser.add_argument('--n_gpus', type=int, default=1,
                        help='number of GPUs (0 indicates CPU)')
    parser.add_argument('--model_save_dir', type=str, default=False,
                        help='directory to save a model')
    parser.add_argument('--resume', type=str, default=False, nargs='?',
                        help='model path to resume training')
    parser.add_argument('--job_name', type=str, default=False,
                        help='job name')
    parser.add_argument('--stdout', type=strtobool, default=False,
                        help='print to standard output')

    # dataset
    parser.add_argument('--train_set', type=str,
                        help='tsv file path for the training set')
    parser.add_argument('--train_set_sub1', type=str, default=False,
                        help='tsv file path for the training set for the 1st auxiliary task')
    parser.add_argument('--train_set_sub2', type=str, default=False,
                        help='tsv file path for the training set for the 2nd auxiliary task')
    parser.add_argument('--dev_set', type=str,
                        help='tsv file path for the development set')
    parser.add_argument('--dev_set_sub1', type=str, default=False,
                        help='tsv file path for the development set for the 1st auxiliary task')
    parser.add_argument('--dev_set_sub2', type=str, default=False,
                        help='tsv file path for the development set for the 2nd auxiliary task')
    parser.add_argument('--eval_sets', type=str, default=[], nargs='+',
                        help='tsv file paths for the evaluation sets')
    parser.add_argument('--nlsyms', type=str, default=False, nargs='?',
                        help='non-linguistic symbols file path')
    parser.add_argument('--dict', type=str,
                        help='dictionary file path')
    parser.add_argument('--dict_sub1', type=str, default=False,
                        help='dictionary file path for the 1st auxiliary task')
    parser.add_argument('--dict_sub2', type=str, default=False,
                        help='dictionary file path for the 2nd auxiliary task')
    parser.add_argument('--unit', type=str, default='wp',
                        choices=['word', 'wp', 'char', 'phone', 'word_char'],
                        help='output unit for the main task')
    parser.add_argument('--unit_sub1', type=str, default=False,
                        choices=['wp', 'char', 'phone'],
                        help='output unit for the 1st auxiliary task')
    parser.add_argument('--unit_sub2', type=str, default=False,
                        choices=['wp', 'char', 'phone'],
                        help='output unit for the 2nd auxiliary task')
    parser.add_argument('--wp_model', type=str, default=False, nargs='?',
                        help='wordpiece model path for the main task')
    parser.add_argument('--wp_model_sub1', type=str, default=False, nargs='?',
                        help='wordpiece model path for the 1st auxiliary task')
    parser.add_argument('--wp_model_sub2', type=str, default=False, nargs='?',
                        help='wordpiece model path for the 2nd auxiliary task')
    # features
    parser.add_argument('--input_type', type=str, default='speech',
                        choices=['speech', 'text'],
                        help='type of input features')
    parser.add_argument('--n_splices', type=int, default=1,
                        help='number of input frames to splice (both for left and right frames)')
    parser.add_argument('--n_stacks', type=int, default=1,
                        help='number of input frames to stack (frame stacking)')
    parser.add_argument('--n_skips', type=int, default=1,
                        help='number of input frames to skip')
    parser.add_argument('--max_n_frames', type=int, default=2000,
                        help='maximum number of input frames')
    parser.add_argument('--min_n_frames', type=int, default=40,
                        help='minimum number of input frames')
    parser.add_argument('--dynamic_batching', type=strtobool, default=True,
                        help='')
    parser.add_argument('--gaussian_noise', type=strtobool, default=False,
                        help='add Gaussian noise to input features')
    parser.add_argument('--sequence_summary_network', type=strtobool, default=False,
                        help='use sequence summary network')
    # topology (encoder)
    parser.add_argument('--conv_in_channel', type=int, default=1, nargs='?',
                        help='input dimension of the first CNN block')
    parser.add_argument('--conv_channels', type=str, default="", nargs='?',
                        help='delimited list of channles in each CNN block')
    parser.add_argument('--conv_kernel_sizes', type=str, default="", nargs='?',
                        help='delimited list of kernel sizes in each CNN block')
    parser.add_argument('--conv_strides', type=str, default="", nargs='?',
                        help='delimited list of strides in each CNN block')
    parser.add_argument('--conv_poolings', type=str, default="", nargs='?',
                        help='delimited list of poolings in each CNN block')
    parser.add_argument('--conv_batch_norm', type=strtobool, default=False, nargs='?',
                        help='apply batch normalization in each CNN block')
    parser.add_argument('--conv_residual', type=strtobool, default=False, nargs='?',
                        help='add residual connection between each CNN block')
    parser.add_argument('--conv_bottleneck_dim', type=int, default=0, nargs='?',
                        help='dimension of the bottleneck layer between CNN and the subsequent RNN layers')
    parser.add_argument('--enc_type', type=str, default='blstm',
                        choices=['blstm', 'lstm', 'bgru', 'gru', 'conv',
                                 'conv_blstm', 'conv_lstm', 'conv_gru', 'conv_blstm',
                                 'transformer', 'conv_transformer', 'tds', 'gated_conv'],
                        help='type of the encoder')
    parser.add_argument('--enc_n_units', type=int, default=512,
                        help='number of units in each encoder RNN layer')
    parser.add_argument('--enc_n_projs', type=int, default=0,
                        help='number of units in the projection layer after each encoder RNN layer')
    parser.add_argument('--enc_n_layers', type=int, default=5,
                        help='number of encoder RNN layers')
    parser.add_argument('--enc_n_layers_sub1', type=int, default=0,
                        help='number of encoder RNN layers in the 1st auxiliary task')
    parser.add_argument('--enc_n_layers_sub2', type=int, default=0,
                        help='number of encoder RNN layers in the 2nd auxiliary task')
    parser.add_argument('--enc_residual', type=strtobool, default=False, nargs='?',
                        help='residual connection between each encoder layer')
    parser.add_argument('--enc_nin', type=strtobool, default=False, nargs='?',
                        help='NiN (network in network) between each encoder layer')
    parser.add_argument('--subsample', type=str, default="1_1_1_1_1",
                        help='delimited list input')
    parser.add_argument('--subsample_type', type=str, default='drop',
                        choices=['drop', 'concat', 'max_pool'],
                        help='type of subsampling in the encoder')
    parser.add_argument('--freeze_encoder', type=strtobool, default=False,
                        help='freeze the encoder parameter')
    # topology (decoder)
    parser.add_argument('--attn_type', type=str, default='location',
                        choices=['no', 'location', 'add', 'dot',
                                 'luong_dot', 'luong_general', 'luong_concat',
                                 'monotonic', 'mocha'],
                        help='type of attention for RNN sequence-to-sequence models')
    parser.add_argument('--attn_dim', type=int, default=128,
                        help='dimension of the attention layer')
    parser.add_argument('--attn_conv_n_channels', type=int, default=10,
                        help='')
    parser.add_argument('--attn_conv_width', type=int, default=100,
                        help='')
    parser.add_argument('--attn_n_heads', type=int, default=1,
                        help='number of heads in the attention layer')
    parser.add_argument('--attn_sharpening', type=float, default=1.0,
                        help='')
    parser.add_argument('--attn_sigmoid', type=strtobool, default=False, nargs='?',
                        help='')
    parser.add_argument('--bridge_layer', type=strtobool, default=False,
                        help='')
    parser.add_argument('--dec_type', type=str, default='lstm',
                        choices=['lstm', 'gru', 'transformer',
                                 'lstm_transducer', 'gru_transducer', 'self_attn_transducer'],
                        help='')
    parser.add_argument('--dec_n_units', type=int, default=512,
                        help='number of units in each decoder RNN layer')
    parser.add_argument('--dec_n_projs', type=int, default=0,
                        help='number of units in the projection layer after each decoder RNN layer')
    parser.add_argument('--dec_n_layers', type=int, default=1,
                        help='number of decoder RNN layers')
    parser.add_argument('--dec_loop_type', type=str, default='normal', nargs='?',
                        choices=['normal', 'lmdecoder'],
                        help='')
    parser.add_argument('--dec_residual', type=strtobool, default=False, nargs='?',
                        help='residual connection between each decoder layer')
    parser.add_argument('--input_feeding', type=strtobool, default=False, nargs='?',
                        help='')
    parser.add_argument('--dec_bottleneck_dim', type=int, default=1024,
                        help='number of dimensions of the bottleneck layer before the softmax layer')
    parser.add_argument('--emb_dim', type=int, default=512,
                        help='number of dimensions in the embedding layer')
    parser.add_argument('--tie_embedding', type=strtobool, default=False, nargs='?',
                        help='tie weights between an embedding matrix and a linear layer before the softmax layer')
    parser.add_argument('--ctc_fc_list', type=str, default="", nargs='?',
                        help='')
    parser.add_argument('--ctc_fc_list_sub1', type=str, default="", nargs='?',
                        help='')
    parser.add_argument('--ctc_fc_list_sub2', type=str, default="", nargs='?',
                        help='')
    # optimization
    parser.add_argument('--batch_size', type=int, default=50,
                        help='mini-batch size')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adadelta', 'adagrad', 'sgd', 'momentum', 'nesterov', 'noam'],
                        help='type of optimizer')
    parser.add_argument('--n_epochs', type=int, default=25,
                        help='number of epochs to train the model')
    parser.add_argument('--convert_to_sgd_epoch', type=int, default=20,
                        help='epoch to converto to SGD fine-tuning')
    parser.add_argument('--print_step', type=int, default=200,
                        help='print log per this value')
    parser.add_argument('--metric', type=str, default='edit_distance',
                        choices=['edit_distance', 'loss', 'acc', 'ppl', 'bleu', 'mse'],
                        help='metric for evaluation during training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--lr_factor', type=float, default=10.0,
                        help='factor of learning rate for Transformer')
    parser.add_argument('--eps', type=float, default=1e-6,
                        help='epsilon parameter for Adadelta optimizer')
    parser.add_argument('--lr_decay_type', type=str, default='always',
                        choices=['always', 'metric', 'warmup'],
                        help='type of learning rate decay')
    parser.add_argument('--lr_decay_start_epoch', type=int, default=10,
                        help='epoch to start to decay learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9,
                        help='decay rate of learning rate')
    parser.add_argument('--lr_decay_patient_n_epochs', type=int, default=0,
                        help='number of epochs to tolerate learning rate decay when validation perfomance is not improved')
    parser.add_argument('--early_stop_patient_n_epochs', type=int, default=5,
                        help='number of epochs to tolerate stopping training when validation perfomance is not improved')
    parser.add_argument('--sort_stop_epoch', type=int, default=10000,
                        help='epoch to stop soring utterances by length')
    parser.add_argument('--eval_start_epoch', type=int, default=1,
                        help='first epoch to start evalaution')
    parser.add_argument('--warmup_start_lr', type=float, default=0,
                        help='initial learning rate for learning rate warm up')
    parser.add_argument('--warmup_n_steps', type=int, default=0,
                        help='number of steps to warm up learing rate')
    parser.add_argument('--accum_grad_n_tokens', type=int, default=0,
                        help='total number of tokens to accumulate gradients')
    # initialization
    parser.add_argument('--param_init', type=float, default=0.1,
                        help='')
    parser.add_argument('--rec_weight_orthogonal', type=strtobool, default=False,
                        help='')
    parser.add_argument('--pretrained_model', type=str, default=False, nargs='?',
                        help='pretrained seq2seq model path')
    # knowledge distillation
    parser.add_argument('--teacher', default=False, nargs='?',
                        help='')
    parser.add_argument('--teacher_lm', default=False, nargs='?',
                        help='')
    parser.add_argument('--distillation_temperature', type=float, default=1.0,
                        help='')
    # regularization
    parser.add_argument('--clip_grad_norm', type=float, default=5.0,
                        help='')
    parser.add_argument('--dropout_in', type=float, default=0.0,
                        help='dropout probability for the input')
    parser.add_argument('--dropout_enc', type=float, default=0.0,
                        help='dropout probability for the encoder')
    parser.add_argument('--dropout_dec', type=float, default=0.0,
                        help='dropout probability for the decoder')
    parser.add_argument('--dropout_emb', type=float, default=0.0,
                        help='dropout probability for the embedding')
    parser.add_argument('--dropout_att', type=float, default=0.0,
                        help='dropout probability for the attention weights')
    parser.add_argument('--zoneout', type=float, default=0.0,
                        help='zoneout probability for the RNNCell in the decoder')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay parameter')
    parser.add_argument('--ss_prob', type=float, default=0.0,
                        help='probability of scheduled sampling')
    parser.add_argument('--ss_type', type=str, default='constant',
                        choices=['constant', 'saturation'],
                        help='type of scheduled sampling')
    parser.add_argument('--lsm_prob', type=float, default=0.0,
                        help='probability of label smoothing')
    parser.add_argument('--ctc_lsm_prob', type=float, default=0.0,
                        help='probability of label smoothing for CTC')
    parser.add_argument('--focal_loss_weight', type=float, default=0.0,
                        help='')
    parser.add_argument('--focal_loss_gamma', type=float, default=2.0,
                        help='')
    parser.add_argument('--adaptive_softmax', type=strtobool, default=False,
                        help='use adaptive softmax')
    # SpecAugment
    parser.add_argument('--freq_width', type=int, default=27,
                        help='')
    parser.add_argument('--n_freq_masks', type=int, default=0,
                        help='')
    parser.add_argument('--time_width', type=int, default=70,
                        help='')
    parser.add_argument('--n_time_masks', type=int, default=0,
                        help='')
    parser.add_argument('--time_width_upper', type=float, default=0.2,
                        help='')
    # MTL
    parser.add_argument('--ctc_weight', type=float, default=0.0,
                        help='CTC loss weight for the main task')
    parser.add_argument('--ctc_weight_sub1', type=float, default=0.0,
                        help='CTC loss weight for the 1st auxiliary task')
    parser.add_argument('--ctc_weight_sub2', type=float, default=0.0,
                        help='CTC loss weight for the 2nd auxiliary task')
    parser.add_argument('--sub1_weight', type=float, default=0.0,
                        help='total loss weight for the 1st auxiliary task')
    parser.add_argument('--sub2_weight', type=float, default=0.0,
                        help='total loss weight for the 2nd auxiliary task')
    parser.add_argument('--mtl_per_batch', type=strtobool, default=False, nargs='?',
                        help='change mini-batch per task')
    parser.add_argument('--task_specific_layer', type=strtobool, default=False, nargs='?',
                        help='insert a task-specific encoder layer per task')
    # foroward-backward
    parser.add_argument('--bwd_weight', type=float, default=0.0,
                        help='cross etnropy loss weight for the backward decoder in the main task')
    # cold fusion
    parser.add_argument('--lm_fusion_type', type=str, default='cold', nargs='?',
                        choices=['cold', 'cold_prob', 'cold_recurrency',
                                 'deep_original', 'deep',
                                 'cache', 'cache_bi'],
                        help='type of LM fusion')
    parser.add_argument('--lm_fusion', type=str, default=False, nargs='?',
                        help='LM path for LM fusion during training')
    # LM initialization, objective
    parser.add_argument('--lm_init', type=str, default=False, nargs='?',
                        help='LM path for initialization of the decoder network')
    parser.add_argument('--lmobj_weight', type=float, default=0.0, nargs='?',
                        help='LM objective weight for the main task')
    parser.add_argument('--share_lm_softmax', type=strtobool, default=False, nargs='?',
                        help='share the softmax layer between ASR and LM')
    # transformer
    parser.add_argument('--d_model', type=int, default=256,
                        help='number of units in self-attention layers in Transformer')
    parser.add_argument('--d_ff', type=int, default=2048,
                        help='number of units in feed-forward fully-conncected layers in Transformer')
    parser.add_argument('--transformer_attn_type', type=str, default='scaled_dot',
                        choices=['scaled_dot', 'add', 'average'],
                        help='type of attention for Transformer')
    parser.add_argument('--transformer_attn_n_heads', type=int, default=4,
                        help='number of heads in the attention layer for Transformer')
    parser.add_argument('--pe_type', type=str, default='add',
                        choices=['add', 'concat', 'learned_add', 'learned_concat', ''],
                        help='type of positional encoding')
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12,
                        help='epsilon value for layer normalization')
    # contextualization
    parser.add_argument('--discourse_aware', type=str, default=False, nargs='?',
                        choices=['state_carry_over', 'hierarchical', ''],
                        help='')
    # decoding parameters
    parser.add_argument('--recog_sets', type=str, default=[], nargs='+',
                        help='tsv file paths for the evaluation sets')
    parser.add_argument('--recog_model', type=str, default=False, nargs='+',
                        help='model path')
    parser.add_argument('--recog_model_bwd', type=str, default=False, nargs='?',
                        help='model path in the reverse direction')
    parser.add_argument('--recog_dir', type=str, default=False,
                        help='directory to save decoding results')
    parser.add_argument('--recog_unit', type=str, default=False, nargs='?',
                        choices=['word', 'wp', 'char', 'phone', 'word_char'],
                        help='')
    parser.add_argument('--recog_metric', type=str, default='edit_distance',
                        choices=['edit_distance', 'loss', 'acc', 'ppl', 'bleu'],
                        help='metric for evaluation')
    parser.add_argument('--recog_oracle', type=strtobool, default=False,
                        help='recognize by teacher-forcing')
    parser.add_argument('--recog_batch_size', type=int, default=1,
                        help='size of mini-batch in evaluation')
    parser.add_argument('--recog_beam_width', type=int, default=1,
                        help='size of beam')
    parser.add_argument('--recog_max_len_ratio', type=float, default=1,
                        help='')
    parser.add_argument('--recog_min_len_ratio', type=float, default=0.0,
                        help='')
    parser.add_argument('--recog_length_penalty', type=float, default=0.0,
                        help='length penalty')
    parser.add_argument('--recog_coverage_penalty', type=float, default=0.0,
                        help='coverage penalty')
    parser.add_argument('--recog_coverage_threshold', type=float, default=0.0,
                        help='coverage threshold')
    parser.add_argument('--recog_gnmt_decoding', type=strtobool, default=False, nargs='?',
                        help='adopt Google NMT beam search decoding')
    parser.add_argument('--recog_eos_threshold', type=float, default=1.5,
                        help='threshold for emitting a EOS token')
    parser.add_argument('--recog_lm_weight', type=float, default=0.0,
                        help='weight of LM score')
    parser.add_argument('--recog_ctc_weight', type=float, default=0.0,
                        help='weight of CTC score')
    parser.add_argument('--recog_lm', type=str, default=False, nargs='?',
                        help='LM path')
    parser.add_argument('--recog_lm_bwd', type=str, default=False, nargs='?',
                        help='LM path in the reverse direction')
    parser.add_argument('--recog_lm_usage', type=str, default='shallow_fusion', nargs='?',
                        choices=['shallow_fusion', 'rescoring'],
                        help='usage of the external LM')
    parser.add_argument('--recog_resolving_unk', type=strtobool, default=False,
                        help='resolving UNK for the word-based model')
    parser.add_argument('--recog_fwd_bwd_attention', type=strtobool, default=False,
                        help='forward-backward attention decoding')
    parser.add_argument('--recog_bwd_attention', type=strtobool, default=False,
                        help='backward attention decoding')
    parser.add_argument('--recog_reverse_lm_rescoring', type=strtobool, default=False,
                        help='rescore with another LM in the reverse direction')
    parser.add_argument('--recog_asr_state_carry_over', type=strtobool, default=False,
                        help='carry over ASR decoder state')
    parser.add_argument('--recog_lm_state_carry_over', type=strtobool, default=False,
                        help='carry over LM state')
    parser.add_argument('--recog_wordlm', type=strtobool, default=False,
                        help='')
    # cache parameters
    parser.add_argument('--recog_n_caches', type=int, default=0,
                        help='number of tokens for cache')
    parser.add_argument('--recog_cache_theta_speech', type=float, default=0.1,
                        help='theta paramter for acoustic cache')
    parser.add_argument('--recog_cache_lambda_speech', type=float, default=0.1,
                        help='lambda paramter for acoustic cache')
    parser.add_argument('--recog_cache_theta_lm', type=float, default=0.1,
                        help='theta paramter for LM cache')
    parser.add_argument('--recog_cache_lambda_lm', type=float, default=0.1,
                        help='lambda paramter for LM cache')
    parser.add_argument('--recog_cache_type', type=str, default='speech',
                        choices=['speech_fifo', 'speech_fifo_online',
                                 'speech_dict', 'speech_dict_overwrite',
                                 'lm_fifo', 'lm_fifo_online',
                                 'lm_dict', 'lm_dict_overwrite', ],
                        help='cache type')
    # distillation related
    parser.add_argument('--recog_nbest', type=float, default=1,
                        help='N-best list for sampling')
    parser.add_argument('--recog_softmax_temperature', type=float, default=1,
                        help='Temperature parameter for the final softmax layer')
    parser.add_argument('--distillation_type', type=str, default='prob',
                        choices=['teacher_forcing', 'beam_search'],
                        help='')
    # pre-training
    parser.add_argument('--am_pretrain_type', type=str, default='masked_audio_lm',
                        choices=['audio_lm', 'masked_audio_lm',
                                 'dae', 'mass'],
                        help='')
    # special label
    parser.add_argument('--replace_sos', type=strtobool, default=False,
                        help='')

    args = parser.parse_args()
    # args, _ = parser.parse_known_args(parser)
    return args
