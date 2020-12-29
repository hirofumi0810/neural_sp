#!/usr/bin/env bash

root=$(pwd)
cd ./examples/ci_test || exit 1;
ln -sf ${root}/coverage.xml .

# LM
./run.sh --stop_stage 3 --lm_conf conf/lm/rnnlm.yaml || exit 1;
./run.sh --stage 3 --stop_stage 3 --lm_conf conf/lm/transformerlm.yaml || exit 1;
./run.sh --stage 3 --stop_stage 3 --lm_conf conf/lm/transformer_xl.yaml || exit 1;

# ASR
./run.sh --stage 4 --conf conf/asr/blstm_las.yaml || exit 1;
./score.sh --model results/asr/train_char/conv2Lblstm16H8P1L_sumfwdbwd_chunkL-1R40_drop4_lstm16H8P1L_location_ss0.1_adam_lr0.001_bs1_ls0.1_warmup2_ctc0.3/model.epoch-1 || exit 1;
./plot_attention.sh --model results/asr/train_char/conv2Lblstm16H8P1L_sumfwdbwd_chunkL-1R40_drop4_lstm16H8P1L_location_ss0.1_adam_lr0.001_bs1_ls0.1_warmup2_ctc0.3/model.epoch-1 || exit 1;

# resume
./run.sh --stage 4 --conf conf/asr/blstm_las.yaml --resume results/asr/train_char/conv2Lblstm16H8P1L_sumfwdbwd_chunkL-1R40_drop4_lstm16H8P1L_location_ss0.1_adam_lr0.001_bs1_ls0.1_warmup2_ctc0.3/model.epoch-1 || exit 1;

# output unit (default: char)
./run.sh --stage 0 --conf conf/asr/blstm_las.yaml --unit word || exit 1;
./run.sh --stage 0 --conf conf/asr/blstm_las.yaml --unit wp || exit 1;
# ./run.sh --stage 0 --conf conf/asr/blstm_las.yaml --unit phone || exit 1;

# Architecture
./run.sh --stage 4 --conf conf/asr/blstm_ctc.yaml || exit 1;
./score.sh --model results/asr/train_char/conv2Lblstm16H8P1L_sumfwdbwd_chunkL-1R40_drop4_adam_lr0.001_bs1_warmup2_ctc1.0/model.epoch-1 || exit 1;
./plot_ctc.sh --model results/asr/train_char/conv2Lblstm16H8P1L_sumfwdbwd_chunkL-1R40_drop4_adam_lr0.001_bs1_warmup2_ctc1.0/model.epoch-1 || exit 1;
./run.sh --stage 4 --conf conf/asr/blstm_transducer.yaml || exit 1;
./score.sh --model results/asr/train_char/conv2Lblstm16H8P1L_sumfwdbwd_chunkL-1R40_drop4_lstm_transducer16H8P1L_adam_lr0.001_bs1_ls0.1_warmup2_ctc0.3/model.epoch-1 || exit 1;
./run.sh --stage 4 --conf conf/asr/transformer.yaml || exit 1;
./score.sh --model results/asr/train_char/conv2Ltransformer8dmodel32dff1L4Hpenone_max_pool4_transformer8dmodel32dff1L4Hpe1dconv3Lscaled_dot_noam_lr5.0_bs1_ls0.1_warmup2_accum2_ctc0.3/model.epoch-2 || exit 1;
./run.sh --stage 4 --conf conf/asr/transformer_ctc.yaml || exit 1;
./run.sh --stage 4 --conf conf/asr/conformer.yaml || exit 1;
./run.sh --stage 4 --conf conf/asr/tds_las.yaml || exit 1;
./run.sh --stage 4 --conf conf/asr/transformer_las.yaml || exit 1;
./run.sh --stage 4 --conf conf/asr/blstm_transformer.yaml || exit 1;

# data augmentation
./run.sh --stage 0 --conf conf/asr/blstm_las.yaml --speed_perturb true || exit 1;
./run.sh --stage 4 --conf conf/asr/blstm_las.yaml --conf2 conf/data/spec_augment.yaml || exit 1;
./run.sh --stage 4 --conf conf/asr/blstm_las.yaml --conf2 conf/data/adaptive_spec_augment.yaml || exit 1;

# streaming ASR
./run.sh --stage 4 --conf conf/asr/lc_transformer_mma_ma4H_ca4H_w16_from4L_64_128_64.yaml || exit 1;

# multi-task
./run_2mtl.sh --stage 0 --conf conf/asr/blstm_las_2mtl.yaml --unit wp --unit_sub1 char || exit 1;
./run_2mtl.sh --stage 0 --conf conf/asr/blstm_las_2mtl.yaml --speed_perturb true --unit wp --unit_sub1 char || exit 1;
./run_2mtl.sh --stage 0 --conf conf/asr/transformer_2mtl.yaml --unit wp --unit_sub1 char || exit 1;
# ./run_2mtl.sh --stage 0 --conf conf/asr/blstm_las_2mtl_per_batch.yaml --unit wp --unit_sub1 char || exit 1;

pip install coverage
coverage report --include ${root}/neural_sp
coverage xml