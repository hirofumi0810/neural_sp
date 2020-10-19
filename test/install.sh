#!/usr/bin/env bash

set -euo pipefail

$CXX -v

TOOL=$(pwd)/tools/tmp
CONDA=$TOOL/miniconda

python --version
pip install -U pip wheel

# install neural_sp
pip list
pip install pip --upgrade
pip install torch==$PYTORCH_VERSION
pip install -e .  # setup.py
pip install -e ".[test]"  # install test dependencies
# conda install -y $(CONDA_PYTORCH) -c pytorch

# [FIXME] hacking==1.1.0 requires flake8<2.7.0,>=2.6.0, but that version has a problem around fstring
# pip install -U flake8 flake8-docstrings

pip install numpy --upgrade
# install matplotlib
# pip install matplotlib

# install kaldi
git clone https://github.com/kaldi-asr/kaldi.git tools/kaldi

# download pre-built kaldi binary (copy from espnet)
[ ! -e ubuntu16-featbin.tar.gz ] && wget --tries=3 https://github.com/espnet/kaldi-bin/releases/download/v0.0.1/ubuntu16-featbin.tar.gz
tar -xf ./ubuntu16-featbin.tar.gz
cp featbin/* tools/kaldi/src/featbin/

use_warpctc=$(python3 <<EOF
from distutils.version import LooseVersion as V
import torch

if V(torch.__version__) < V("1.2"):
    print("true")
else:
    print("false")
EOF
)

# install warp-ctc (use @jnishi patched version)
if [ $use_warpctc ]; then
    git clone https://github.com/jnishi/warp-ctc.git -b pytorch-1.0.0
    cd warp-ctc && mkdir build && cd build && cmake .. && make -j4 && cd ..
    pip install cffi
    cd pytorch_binding && python setup.py install && cd ../..
fi

# install warp-transducer
git clone https://github.com/HawkAaron/warp-transducer.git
cd warp-transducer && mkdir build && cd build && cmake .. && make && cd ..
cd pytorch_binding && python setup.py install && cd ../..

# install sentencepiece
git clone https://github.com/google/sentencepiece.git tools/sentencepiece
cd tools/sentencepiece && mkdir build && cd build && (cmake3 .. || cmake ..) && make && cd ../../..

# log
pip freeze
