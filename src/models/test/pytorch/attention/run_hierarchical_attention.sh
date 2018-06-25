#!/bin/bash

# Select GPU
if [ $# -lt 1 ]; then
  echo "Error: set GPU number." 1>&2
  echo "Usage: ./run_hierarchical_attention.sh gpu_id1 gpu_id2... (arbitrary number)" 1>&2
  exit 1
fi

ngpus=`expr $#`
gpu_ids=$1

if [ $# -gt 1 ]; then
  rest_ngpus=`expr $ngpus - 1`
  for i in `seq 1 $rest_ngpus`
  do
    gpu_ids=$gpu_ids","${2}
    shift
  done
fi

# Set path to CUDA
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64

# Set path to python
PYTHON=/home/inaguma/.pyenv/versions/anaconda3-4.1.1/envs/`hostname`/bin/python

# Background job version
CUDA_VISIBLE_DEVICES=${gpu_ids} $PYTHON test_hierarchical_attention.py --ngpus ${ngpus}
