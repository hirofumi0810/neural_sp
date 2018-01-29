#!/bin/bash

ROOT=../../../pytorch_speech_recognition

# Select GPU
if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./run_retrain.sh path_to_config_file gpu_index" 1>&2
  exit 1
fi

# Set path to CUDA
export PATH=$PATH:/usr/local/cuda-8.0/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64

# Set path to python
# PYTHON=/home/lab5/inaguma/.pyenv/versions/anaconda3-4.1.1/bin/python
PYTHON=/home/lab5/inaguma/.pyenv/versions/anaconda3-4.1.1/envs/`hostname`/bin/python

saved_model_path=$1
gpu_index=$2
filename=$(basename $saved_model_path | awk -F. '{print $1}')

mkdir -p log
#
CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
nohup $PYTHON train.py \
  --gpu $gpu_index \
  --saved_model_path $saved_model_path > log/$filename".log" &
