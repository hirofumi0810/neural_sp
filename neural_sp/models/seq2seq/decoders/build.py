# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Select an decoder network."""


def build_decoder(args, special_symbols, enc_n_units, vocab,
                  ctc_weight, ctc_fc_list, global_weight, external_lm=None):

    # safeguard
    if not hasattr(args, 'transformer_dec_d_model') and hasattr(args, 'transformer_d_model'):
        args.transformer_dec_d_model = args.transformer_d_model
    if not hasattr(args, 'transformer_dec_d_ff') and hasattr(args, 'transformer_d_ff'):
        args.transformer_dec_d_ff = args.transformer_d_ff
    if not hasattr(args, 'transformer_dec_n_heads') and hasattr(args, 'transformer_n_heads'):
        args.transformer_dec_n_heads = args.transformer_n_heads
    if not hasattr(args, 'transformer_dec_attn_type') and hasattr(args, 'transformer_attn_type'):
        args.transformer_dec_attn_type = args.transformer_attn_type

    if args.dec_type in ['transformer', 'transformer_xl']:
        from neural_sp.models.seq2seq.decoders.transformer import TransformerDecoder
        decoder = TransformerDecoder(
            special_symbols=special_symbols,
            enc_n_units=enc_n_units,
            attn_type=args.transformer_dec_attn_type,
            n_heads=args.transformer_dec_n_heads,
            n_layers=args.dec_n_layers,
            d_model=args.transformer_dec_d_model,
            d_ff=args.transformer_dec_d_ff,
            ffn_bottleneck_dim=args.transformer_ffn_bottleneck_dim,
            pe_type=args.transformer_dec_pe_type,
            layer_norm_eps=args.transformer_layer_norm_eps,
            ffn_activation=args.transformer_ffn_activation,
            vocab=vocab,
            tie_embedding=args.tie_embedding,
            dropout=args.dropout_dec,
            dropout_emb=args.dropout_emb,
            dropout_att=args.dropout_att,
            dropout_layer=args.dropout_dec_layer,
            dropout_head=args.dropout_head,
            lsm_prob=args.lsm_prob,
            ctc_weight=ctc_weight,
            ctc_lsm_prob=args.ctc_lsm_prob,
            ctc_fc_list=ctc_fc_list,
            backward=(dir == 'bwd'),
            global_weight=global_weight,
            mtl_per_batch=args.mtl_per_batch,
            param_init=args.transformer_param_init,
            mma_chunk_size=args.mocha_chunk_size,
            mma_n_heads_mono=args.mocha_n_heads_mono,
            mma_n_heads_chunk=args.mocha_n_heads_chunk,
            mma_init_r=args.mocha_init_r,
            mma_eps=args.mocha_eps,
            mma_std=args.mocha_std,
            mma_no_denominator=args.mocha_no_denominator,
            mma_1dconv=args.mocha_1dconv,
            mma_quantity_loss_weight=args.mocha_quantity_loss_weight,
            mma_headdiv_loss_weight=args.mocha_head_divergence_loss_weight,
            latency_metric=args.mocha_latency_metric,
            latency_loss_weight=args.mocha_latency_loss_weight,
            mma_first_layer=args.mocha_first_layer,
            share_chunkwise_attention=args.share_chunkwise_attention,
            external_lm=external_lm,
            lm_fusion=args.lm_fusion)

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
            ctc_weight=ctc_weight,
            ctc_lsm_prob=args.ctc_lsm_prob,
            ctc_fc_list=ctc_fc_list,
            external_lm=external_lm if args.lm_init else None,
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
            ctc_weight=ctc_weight,
            ctc_lsm_prob=args.ctc_lsm_prob,
            ctc_fc_list=ctc_fc_list,
            mbr_training=args.mbr_training,
            mbr_ce_weight=args.mbr_ce_weight,
            external_lm=external_lm,
            lm_fusion=args.lm_fusion,
            lm_init=args.lm_init,
            backward=(dir == 'bwd'),
            global_weight=global_weight,
            mtl_per_batch=args.mtl_per_batch,
            param_init=args.param_init,
            mocha_chunk_size=args.mocha_chunk_size,
            mocha_n_heads_mono=args.mocha_n_heads_mono,
            mocha_init_r=args.mocha_init_r,
            mocha_eps=args.mocha_eps,
            mocha_std=args.mocha_std,
            mocha_no_denominator=args.mocha_no_denominator,
            mocha_1dconv=args.mocha_1dconv,
            mocha_decot_lookahead=args.mocha_decot_lookahead,
            quantity_loss_weight=args.mocha_quantity_loss_weight,
            latency_metric=args.mocha_latency_metric,
            latency_loss_weight=args.mocha_latency_loss_weight,
            gmm_attn_n_mixtures=args.gmm_attn_n_mixtures,
            replace_sos=args.replace_sos,
            distillation_weight=args.distillation_weight,
            discourse_aware=args.discourse_aware)

    return decoder
