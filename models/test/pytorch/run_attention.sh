#!/bin/bash

# Select GPU
if [ $# -ne 1 ]; then
  echo "Error: set GPU index." 1>&2
  echo "Usage: ./run_attention_pytorch.sh gpu_index" 1>&2
  exit 1
fi

# Set path to CUDA
# export PATH=$PATH:/usr/local/cuda-8.0/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64

# Set path to python
# PYTHON=/home/lab5/inaguma/.pyenv/versions/anaconda3-4.1.1/bin/python

export PATH=$PATH:/opt/share/cuda-8.0/x86_64/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/x86_64/lib64:/usr/local/cuda-8.0/x86_64/extras/CUPTI/lib64:/opt/share/cuDNN-v5.1-8.0/cuda/lib64
PYTHON_PATH=/u/jp573469/.pyenv/shims/python

gpu_index=$1

# Background job version
CUDA_VISIBLE_DEVICES=$gpu_index $PYTHON test_att_pytorch.py


# CUDA_ROOT=/opt/share/cuda-8.0/x86_64
# PATH=$PATH:/opt/share/cuda-8.0/x86_64
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/share/cuda-8.0/x86_64/lib64:/opt/share/cuda-8.0/x86_64/lib
# CPATH=$CPATH:/opt/share/cuda-8.0/x86_64/include
# CUDA_INC_DIR=/opt/share/cuda-8.0/x86_64/bin:$CUDA_INC_DIR
# CUDA_HOME=/opt/share/cuda-8.0/x86_64

CUDA_ROOT=/opt/share/cuda-7.5
PATH=$PATH:/opt/share/cuda-7.5
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/share/cuda-7.5/lib64:/opt/share/cuda-7.5/lib
CPATH=$CPATH:/opt/share/cuda-7.5/include
CUDA_INC_DIR=/opt/share/cuda-7.5/bin:$CUDA_INC_DIR
CUDA_HOME=/opt/share/cuda-7.5

TORCH_BUILD_DIR=/u/jp573469/.pyenv/versions/anaconda3-2.5.0/lib/python3.5/site-packages/torch/lib
TORCH_TH_INCLUDE_DIR=$TORCH_BUILD_DIR/include/TH
TORCH_THC_INCLUDE_DIR=$TORCH_BUILD_DIR/include/THC

# TORCH_THC_UTILS_INCLUDE_DIR=$ENV{HOME}/pytorch/torch/lib/THC

python setup.py install
