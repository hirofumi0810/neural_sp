#!/usr/bin/env bash

set -euo pipefail

$CXX -v

ROOT=$(pwd)
TOOL=${ROOT}/tools/neural_sp

# install kaldi (not compiled)
git clone https://github.com/kaldi-asr/kaldi.git ${TOOL}/kaldi

# download pre-built kaldi binary (copy from espnet)
[ ! -e ubuntu16-featbin.tar.gz ] && wget --tries=3 https://github.com/espnet/kaldi-bin/releases/download/v0.0.1/ubuntu16-featbin.tar.gz
tar -xf ./ubuntu16-featbin.tar.gz
cp featbin/* ${TOOL}/kaldi/src/featbin/

cd tools
make PYTORCH_VERSION="${PYTORCH_VERSION}" PYTHON_VERSION="${TRAVIS_PYTHON_VERSION}" TOOL="${TOOL}" KALDI=${TOOL}/kaldi
cd ${ROOT}

source tools/neural_sp/miniconda/bin/activate

pip install -e ".[test]"  # install test dependencies (setup.py)

# log
pip freeze