#!/usr/bin/env bash

set -euo pipefail

TOOL=$(pwd)/tools/tmp
CONDA=$TOOL/miniconda

$CXX -v

python --version
pip install -U pip wheel

# install neural_sp
pip list
pip install pip --upgrade
pip install torch==$(PYTORCH_VERSION)
pip install -e .  # setup.py
# pip install -e ".[test]"
# conda install -y $(CONDA_PYTORCH) -c pytorch

# [FIXME] hacking==1.1.0 requires flake8<2.7.0,>=2.6.0, but that version has a problem around fstring
# pip install -U flake8 flake8-docstrings

# install matplotlib
# pip install matplotlib

# install kaldi
git clone https://github.com/kaldi-asr/kaldi.git tools/kaldi

# install warp-ctc (use @jnishi patched version)
git clone https://github.com/jnishi/warp-ctc.git -b pytorch-1.0.0
cd warp-ctc && mkdir build && cd build && cmake .. && make -j4 && cd ..
pip install cffi
cd pytorch_binding && python setup.py install && cd ../..

# log
pip freeze
