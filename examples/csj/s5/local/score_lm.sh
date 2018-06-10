#!/bin/bash

. ./path.sh
set -e

### Select GPU
if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./score_lm.sh path_to_saved_model gpu_index" 1>&2
  exit 1
fi

### Set path to save dataset
DATA="/n/sd8/inaguma/corpus/csj/kaldi"

saved_model_path=$1
gpu_index=$2

CUDA_VISIBLE_DEVICES=$gpu_index \
$PYTHON exp/evaluation/eval_lm.py \
  --data_save_path $DATA \
  --model_path $saved_model_path \
  --epoch -1
