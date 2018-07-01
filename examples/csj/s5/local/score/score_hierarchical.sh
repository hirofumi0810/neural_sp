#!/bin/bash

. ./path.sh
set -e

### Select GPU
if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./score_hierarchical.sh path_to_saved_model gpu_id" 1>&2
  exit 1
fi

# main task
beam_width=4
length_penalty=0
coverage_penalty=0
rnnlm_weight=0.3
rnnlm_path=
# rnnlm_path=/n/sd8/inaguma/result/csj/pytorch/rnnlm/word/aps_other/
# rnnlm_path=/n/sd8/inaguma/result/csj/pytorch/rnnlm/word/all/

# sub task
beam_width_sub=4
length_penalty_sub=0
coverage_penalty_sub=0
rnnlm_weight_sub=0.3
rnnlm_path_sub=
# rnnlm_path_sub=/n/sd8/inaguma/result/csj/pytorch/rnnlm/character_wb/aps_other/
# rnnlm_path_sub=/n/sd8/inaguma/result/csj/pytorch/rnnlm/character_wb/all/

joint_decoding=false
resolving_unk=true
score_sub_weight=0.3

CUDA_VISIBLE_DEVICES=$2 ${PYTHON} ../../../src/bin/evaluation/eval_hierarchical.py \
  --corpus ${corpus} \
  --eval_sets eval1 eval2 eval3 \
  --data_save_path ${data} \
  --model_path $1 \
  --epoch -1 \
  --eval_batch_size 1 \
  --beam_width ${beam_width} \
  --length_penalty ${length_penalty} \
  --coverage_penalty ${coverage_penalty} \
  --rnnlm_weight ${rnnlm_weight} \
  --rnnlm_path ${rnnlm_path} \
  --beam_width_sub ${beam_width_sub} \
  --length_penalty_sub ${length_penalty_sub} \
  --coverage_penalty_sub ${coverage_penalty_sub} \
  --rnnlm_weight_sub ${rnnlm_weight_sub} \
  --rnnlm_path_sub ${rnnlm_path_sub} \
  --resolving_unk ${resolving_unk} \
  --joint_decoding ${joint_decoding} \
  --score_sub_weight ${score_sub_weight} \
  --score_sub_task false
