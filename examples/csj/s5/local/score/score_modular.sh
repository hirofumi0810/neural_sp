#!/bin/bash

. ./path.sh
set -e

### Select GPU
if [ $# -ne 3 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./score_modular.sh path_to_A2P path_to_P2W gpu_id" 1>&2
  exit 1
fi

# a2p
beam_width_a2p=1
length_penalty_a2p=0
coverage_penalty_a2p=0

# p2w
beam_width_p2w=1
length_penalty_p2w=0
coverage_penalty_p2w=0
rnnlm_weight=0.3
rnnlm_path=

CUDA_VISIBLE_DEVICES=$3 ${PYTHON} ../../../src/bin/evaluation/eval_modular.py \
  --corpus ${corpus} \
  --eval_sets eval1 eval2 eval3 \
  --data_save_path ${data} \
  --model_path_a2p $1 \
  --model_path_p2w $2 \
  --epoch_a2p -1 \
  --epoch_p2w -1 \
  --eval_batch_size 1 \
  --beam_width_a2p ${beam_width_a2p} \
  --length_penalty_a2p ${length_penalty_a2p} \
  --coverage_penalty_a2p ${coverage_penalty_a2p} \
  --beam_width_p2w ${beam_width_p2w} \
  --length_penalty_p2w ${length_penalty_p2w} \
  --coverage_penalty_p2w ${coverage_penalty_p2w} \
  --rnnlm_weight ${rnnlm_weight} \
  --rnnlm_path ${rnnlm_path}
