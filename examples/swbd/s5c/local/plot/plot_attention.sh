#!/bin/bash

. ./path.sh
set -e

### Select GPU
if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./plot_attention.sh path_to_saved_model gpu_id" 1>&2
  exit 1
fi

beam_width=4
length_penalty=0
coverage_penalty=0
rnnlm_weight=0.3
rnnlm_path=
# word
# rnnlm_path=/n/sd8/inaguma/result/swbd/pytorch/rnnlm/word/swbd_fisher/

# char
# rnnlm_path=/n/sd8/inaguma/result/swbd/pytorch/rnnlm/character/swbd_fisher/

CUDA_VISIBLE_DEVICES=$2 ${PYTHON} ../../../src/bin/visualization/plot_attention_weights.py \
  --corpus ${corpus} \
  --data_type eval2000_swbd \
  --data_save_path ${data} \
  --model_path $1 \
  --epoch -1 \
  --eval_batch_size 1 \
  --beam_width ${beam_width} \
  --length_penalty ${length_penalty} \
  --coverage_penalty ${coverage_penalty} \
  --rnnlm_weight ${rnnlm_weight} \
  --rnnlm_path ${rnnlm_path}
