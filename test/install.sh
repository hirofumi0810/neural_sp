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
# pip install -e ".[test]"
# conda install -y $(CONDA_PYTORCH) -c pytorch

# install other tools
pip install pycodestyle
pip install pytest-cov

# [FIXME] hacking==1.1.0 requires flake8<2.7.0,>=2.6.0, but that version has a problem around fstring
# pip install -U flake8 flake8-docstrings

# install matplotlib
# pip install matplotlib

# install kaldi
git clone https://github.com/kaldi-asr/kaldi.git tools/kaldi

# download pre-built kaldi binary (copy from espnet)
[ ! -e ubuntu16-featbin.tar.gz ] && wget --tries=3 https://github.com/espnet/kaldi-bin/releases/download/v0.0.1/ubuntu16-featbin.tar.gz
tar -xf ./ubuntu16-featbin.tar.gz
cp featbin/* tools/kaldi/src/featbin/

# install warp-ctc (use @jnishi patched version)
git clone https://github.com/jnishi/warp-ctc.git -b pytorch-1.0.0
cd warp-ctc && mkdir build && cd build && cmake .. && make -j4 && cd ..
pip install cffi
cd pytorch_binding && python setup.py install && cd ../..

# install warp-transducer
git clone https://github.com/HawkAaron/warp-transducer.git
cd warp-transducer && mkdir build && cd build && cmake .. && make && cd ..
cd pytorch_binding && python setup.py install && cd ../..

# install sentencepiece
git clone https://github.com/google/sentencepiece.git tools/sentencepiece
cd tools/sentencepiece && mkdir build && cd build && (cmake3 .. || cmake ..) && make && cd ../../..

# log
pip freeze
