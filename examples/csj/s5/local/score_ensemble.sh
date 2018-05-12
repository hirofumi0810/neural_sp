#!/bin/bash

. ./path.sh
set -e

### Select GPU
if [ $# -ne 1 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./score_ensemble.sh path_to_saved_model gpu_index" 1>&2
  exit 1
fi

### Set path to save dataset
DATA_SAVEPATH="/n/sd8/inaguma/corpus/csj/kaldi"

# saved_model_path=$1
gpu_index=$1

CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
$PYTHON exp/evaluation/eval_ensemble.py \
  --data_save_path $DATA_SAVEPATH \
  --epoch -1 \
  --beam_width 1 \
  --eval_batch_size 1

  # --model_path $saved_model_path \
