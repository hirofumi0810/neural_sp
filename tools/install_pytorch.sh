#!/bin/bash

# Install optional dependencies
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" # [anaconda root directory]

# Install basic dependencies
conda install numpy pyyaml mkl setuptools cmake cffi

# Add LAPACK support for the GPU
conda install -c soumith magma-cuda80 # or magma-cuda75 if CUDA 7.5

current_dir=`pwd`
cd ~/tool

# Get the PyTorch source
if [ ! -e pytorch_`hostname` ]; then
  git clone --recursive https://github.com/pytorch/pytorch
  mv pytorch pytorch_`hostname`
fi
cd pytorch_`hostname`
git submodule update --init

# Install PyTorch
python setup.py install

cd $current_dir
