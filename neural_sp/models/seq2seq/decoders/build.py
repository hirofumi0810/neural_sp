#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Select an decoder network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def build_decoder(args, special_symbols, enc_n_units, vocab,
                  ctc_weight, ctc_fc_list, global_weight,
                  lm_fusion=None, lm_init=None):
    if args.dec_type == 'transformer':
        if args.attn_type == 'cif':
            raise NotImplementedError
        else:
            from neural_sp.models.seq2seq.decoders.transformer import TransformerDecoder
            decoder = TransformerDecoder(
                special_symbols=special_symbols,
                enc_n_units=enc_n_units,
                attn_type=args.transformer_attn_type,
                n_heads=args.transformer_n_heads,
                n_layers=args.dec_n_layers,
                d_model=args.transformer_d_model,
                d_ff=args.transformer_d_ff,
                layer_norm_eps=args.transformer_layer_norm_eps,
                ffn_activation=args.transformer_ffn_activation,
                pe_type=args.transformer_dec_pe_type,
                vocab=vocab,
                tie_embedding=args.tie_embedding,
                dropout=args.dropout_dec,
                dropout_emb=args.dropout_emb,
                dropout_att=args.dropout_att,
                lsm_prob=args.lsm_prob,
                ctc_weight=ctc_weight,
                ctc_lsm_prob=args.ctc_lsm_prob,
                ctc_fc_list=ctc_fc_list,
                backward=(dir == 'bwd'),
                global_weight=global_weight,
                mtl_per_batch=args.mtl_per_batch,
                param_init=args.transformer_param_init,
                sync_bidir_attention=args.sync_bidir_attention,
                half_pred=args.half_pred)

    elif args.dec_type == 'transformer_transducer':
        raise NotImplementedError
        from neural_sp.models.seq2seq.decoders.transformer_transducer import TrasformerTransducer
        decoder = TrasformerTransducer(
            special_symbols=special_symbols,
            enc_n_units=enc_n_units,
            attn_type=args.transformer_attn_type,
            n_heads=args.transformer_n_heads,
            n_layers=args.dec_n_layers,
            d_model=args.transformer_d_model,
            d_ff=args.transformer_d_ff,
            layer_norm_eps=args.transformer_layer_norm_eps,
            ffn_activation=args.transformer_ffn_activation,
            pe_type=args.transformer_dec_pe_type,
            vocab=vocab,
            dropout=args.dropout_dec,
            dropout_emb=args.dropout_emb,
            dropout_att=args.dropout_att,
            lsm_prob=args.lsm_prob,
            ctc_weight=ctc_weight,
            ctc_lsm_prob=args.ctc_lsm_prob,
            ctc_fc_list=ctc_fc_list,
            lm_init=lm_init,
            global_weight=global_weight,
            mtl_per_batch=args.mtl_per_batch,
            param_init=args.transformer_param_init)

    elif args.dec_type in ['lstm_transducer', 'gru_transducer']:
        from neural_sp.models.seq2seq.decoders.rnn_transducer import RNNTransducer
        decoder = RNNTransducer(
            special_symbols=special_symbols,
            enc_n_units=enc_n_units,
            rnn_type=args.dec_type,
            n_units=args.dec_n_units,
            n_projs=args.dec_n_projs,
            n_layers=args.dec_n_layers,
            bottleneck_dim=args.dec_bottleneck_dim,
            emb_dim=args.emb_dim,
            vocab=vocab,
            dropout=args.dropout_dec,
            dropout_emb=args.dropout_emb,
            lsm_prob=args.lsm_prob,
            ctc_weight=ctc_weight,
            ctc_lsm_prob=args.ctc_lsm_prob,
            ctc_fc_list=ctc_fc_list,
            lm_init=lm_init,
            global_weight=global_weight,
            mtl_per_batch=args.mtl_per_batch,
            param_init=args.param_init)

    else:
        from neural_sp.models.seq2seq.decoders.las import RNNDecoder
        decoder = RNNDecoder(
            special_symbols=special_symbols,
            enc_n_units=enc_n_units,
            rnn_type=args.dec_type,
            n_units=args.dec_n_units,
            n_projs=args.dec_n_projs,
            n_layers=args.dec_n_layers,
            bottleneck_dim=args.dec_bottleneck_dim,
            emb_dim=args.emb_dim,
            vocab=vocab,
            tie_embedding=args.tie_embedding,
            attn_type=args.attn_type,
            attn_dim=args.attn_dim,
            attn_sharpening_factor=args.attn_sharpening_factor,
            attn_sigmoid_smoothing=args.attn_sigmoid,
            attn_conv_out_channels=args.attn_conv_n_channels,
            attn_conv_kernel_size=args.attn_conv_width,
            attn_n_heads=args.attn_n_heads,
            dropout=args.dropout_dec,
            dropout_emb=args.dropout_emb,
            dropout_att=args.dropout_att,
            lsm_prob=args.lsm_prob,
            ss_prob=args.ss_prob,
            ss_type=args.ss_type,
            ctc_weight=ctc_weight,
            ctc_lsm_prob=args.ctc_lsm_prob,
            ctc_fc_list=ctc_fc_list,
            mbr_weight=args.mbr_weight,
            mbr_nbest=args.mbr_nbest,
            mbr_softmax_smoothing=args.mbr_softmax_smoothing,
            lm_fusion=lm_fusion,
            lm_fusion_type=args.lm_fusion_type,
            lm_init=lm_init,
            backward=(dir == 'bwd'),
            global_weight=global_weight,
            mtl_per_batch=args.mtl_per_batch,
            param_init=args.param_init,
            mocha_chunk_size=args.mocha_chunk_size,
            mocha_adaptive=args.mocha_adaptive,
            mocha_1dconv=args.mocha_1dconv,
            mocha_quantity_loss_weight=args.mocha_quantity_loss_weight,
            mocha_ctc_sync=args.mocha_ctc_sync,
            gmm_attn_n_mixtures=args.gmm_attn_n_mixtures,
            replace_sos=args.replace_sos,
            soft_label_weight=args.soft_label_weight,
            discourse_aware=args.discourse_aware)

    return decoder
