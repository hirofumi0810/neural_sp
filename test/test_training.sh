#!/usr/bin/env bash

cd ./examples/ci_test || exit 1;

# LM
./run.sh --stop_stage 3 --lm_conf conf/lm/rnnlm.yaml || exit 1;
./run.sh --stage 3 --stop_stage 3 --lm_conf conf/lm/transformerlm.yaml || exit 1;
./run.sh --stage 3 --stop_stage 3 --lm_conf conf/lm/transformer_xl.yaml || exit 1;

# ASR
./run.sh --stage 4 --conf conf/asr/blstm_las.yaml || exit 1;
./run.sh --stage 4 --conf conf/asr/blstm_las.yaml --conf2 conf/data/spec_augment.yaml || exit 1;
./run.sh --stage 4 --conf conf/asr/blstm_las.yaml --conf2 conf/data/adaptive_spec_augment.yaml || exit 1;
./run.sh --stage 4 --conf conf/asr/blstm_ctc.yaml || exit 1;
./run.sh --stage 4 --conf conf/asr/blstm_transducer.yaml || exit 1;
./run.sh --stage 4 --conf conf/asr/transformer.yaml || exit 1;
./run.sh --stage 4 --conf conf/asr/transformer_ctc.yaml || exit 1;
./run.sh --stage 4 --conf conf/asr/conformer.yaml || exit 1;
./run.sh --stage 4 --conf conf/asr/tds_las.yaml || exit 1;
./run.sh --stage 4 --conf conf/asr/transformer_las.yaml || exit 1;
./run.sh --stage 4 --conf conf/asr/blstm_transformer.yaml || exit 1;

# output unit (default: char)
./run.sh --stage 0 --conf conf/asr/blstm_las.yaml --unit wp || exit 1;
# ./run.sh --stage 0 --conf conf/asr/blstm_las.yaml --unit phone || exit 1;

# speed perturbation
./run.sh --stage 0 --conf conf/asr/blstm_las.yaml --speed_perturb true || exit 1;

# streaming ASR
# ./run.sh --stage 4 --conf conf/asr/lcblstm_mocha_chunk4040.yaml || exit 1;
./run.sh --stage 4 --conf conf/asr/lc_transformer_mma_ma4H_ca4H_w16_from4L_64_128_64.yaml || exit 1;

# multi-task
./run_2mtl.sh --stage 0 --conf conf/asr/blstm_las_2mtl.yaml --unit wp --unit_sub1 char || exit 1;
./run_2mtl.sh --stage 0 --conf conf/asr/transformer_2mtl.yaml --unit wp --unit_sub1 char || exit 1;
./run_2mtl.sh --stage 0 --conf conf/asr/blstm_las_2mtl.yaml --speed_perturb true --unit wp --unit_sub1 char || exit 1;
# ./run_2mtl.sh --stage 0 --conf conf/asr/blstm_las_2mtl_per_batch.yaml --unit wp --unit_sub1 char || exit 1;
