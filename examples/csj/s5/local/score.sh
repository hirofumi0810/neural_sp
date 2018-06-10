#!/bin/bash

. ./path.sh
set -e

### Select GPU
if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./score.sh path_to_saved_model gpu_index" 1>&2
  exit 1
fi

### Set path to save dataset
DATA="/n/sd8/inaguma/corpus/csj/kaldi"

saved_model_path=$1
gpu_index=$2

beam_width=4
length_penalty=0
coverage_penalty=0
rnnlm_weight=0.3
# rnnlm_path=
rnnlm_path="/n/sd8/inaguma/result/csj/pytorch/rnnlm/word/aps_other/lstm1024H1Lemb128_adam_lr1e-3_drophidden0.2out0.2emb0.2"

CUDA_VISIBLE_DEVICES=$gpu_index \
$PYTHON exp/evaluation/eval.py \
  --data_save_path $DATA \
  --model_path $saved_model_path \
  --epoch -1 \
  --eval_batch_size 1 \
  --beam_width $beam_width \
  --length_penalty $length_penalty \
  --coverage_penalty $coverage_penalty \
  --rnnlm_weight $rnnlm_weight \
  --rnnlm_path $rnnlm_path
