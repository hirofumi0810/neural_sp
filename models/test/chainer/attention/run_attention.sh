#!/bin/bash

# Select GPU
if [ $# -ne 1 ]; then
  echo "Error: set GPU index." 1>&2
  echo "Usage: ./run_attention.sh gpu_index" 1>&2
  exit 1
fi

# Set path to CUDA
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64

# Set path to python
PYTHON=/home/lab5/inaguma/.pyenv/versions/anaconda3-4.1.1/envs/`hostname`/bin/python

gpu_index=$1

# Background job version
CUDA_VISIBLE_DEVICES=$gpu_index $PYTHON test_attention.py
