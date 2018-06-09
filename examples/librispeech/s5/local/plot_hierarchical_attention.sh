#!/bin/bash

. ./path.sh
set -e

### Select GPU
if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./plot_hierarchical_attention.sh path_to_saved_model gpu_index" 1>&2
  exit 1
fi

### Set path to save dataset
DATA="/n/sd8/inaguma/corpus/librispeech/kaldi"

saved_model_path=$1
gpu_index=$2

beam_width=4
beam_width_sub=4
length_penalty=0.1
coverage_penalty=0

CUDA_VISIBLE_DEVICES=$gpu_index \
$PYTHON exp/visualization/plot_hierarchical_attention_weights.py \
  --data_save_path $DATA \
  --model_path $saved_model_path \
  --epoch -1 \
  --eval_batch_size 1 \
  --beam_width $beam_width \
  --beam_width_sub $beam_width_sub \
  --length_penalty $length_penalty \
  --coverage_penalty $coverage_penalty
