#!/bin/bash

. ./path.sh
set -e

### Select GPU
if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./plot_ctc.sh path_to_saved_model gpu_index" 1>&2
  exit 1
fi

### Set path to save dataset
DATA_SAVEPATH="/n/sd8/inaguma/corpus/swbd/kaldi"

saved_model_path=$1
gpu_index=$2

CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
$PYTHON exp/visualization/plot_ctc_probs.py \
  --model_path $saved_model_path \
  --epoch -1 \
  --eval_batch_size 1 \
  --data_save_path $DATA_SAVEPATH
