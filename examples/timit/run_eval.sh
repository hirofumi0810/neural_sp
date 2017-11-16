#!/bin/bash

MODEL_SAVE_PATH="/n/sd8/inaguma/result/pytorch/timit"

# Select GPU
if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./run_eval.sh path_to_saved_model gpu_index" 1>&2
  exit 1
fi

# Set path to CUDA
export PATH=$PATH:/usr/local/cuda-8.0/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64

# Set path to python
PYTHON=/home/lab5/inaguma/.pyenv/versions/anaconda3-4.1.1/bin/python

config_path=$1
gpu_index=$2
filename=$(basename $config_path | awk -F. '{print $1}')

CUDA_VISIBLE_DEVICES=$gpu_index $PYTHON eval.py \
  $config_path $MODEL_SAVE_PATH
