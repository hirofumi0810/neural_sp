#!/bin/bash

. ./path.sh
set -e

### Select GPU
if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./score_hierarchical_attention_joint_decoding.sh path_to_saved_model gpu_id" 1>&2
  exit 1
fi

### Set path to save dataset
data=/n/sd8/inaguma/corpus/wsj/kaldi

beam_width=4
length_penalty=0.1
coverage_penalty=0
resolving_unk=true
score_sub_weight=0.3

CUDA_VISIBLE_DEVICES=$2 ${PYTHON} ex2/evaluation/eval_hierarchical.py \
  --data_save_path ${data} \
  --model_path $1 \
  --epoch -1 \
  --eval_batch_size 1 \
  --beam_width ${beam_width} \
  --beam_width_sub 1 \
  --length_penalty ${length_penalty} \
  --coverage_penalty ${coverage_penalty} \
  --resolving_unk ${resolving_unk} \
  --joint_decoding true \
  --score_sub_weight ${score_sub_weight}
