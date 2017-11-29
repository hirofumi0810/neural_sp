#!/bin/bash

current_dir=`pwd`
cd ~/tool

if [ ! -e warp-ctc_`hostname` ]; then
  git clone https://github.com/SeanNaren/warp-ctc.git
mv warp-ctc warp-ctc_`hostname`
fi
cd warp-ctc_`hostname`

mkdir build
cd build
cmake ..
make
export CUDA_HOME="/usr/local/cuda"
cd ../pytorch_binding
python setup.py install

cd $current_dir
