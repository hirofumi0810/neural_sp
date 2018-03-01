export KALDI_ROOT="/home/lab5/inaguma/tool/kaldi"

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh

### python
# export PYTHON=/home/lab5/inaguma/.pyenv/versions/anaconda3-4.1.1/bin/python
export PYTHON=/home/lab5/inaguma/.pyenv/versions/anaconda3-4.1.1/envs/`hostname`/bin/python

### CUDA
# export PATH=$PATH:/usr/local/cuda/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib:/usr/local/lib64:/usr/local/cuda/bin/nvcc
export PATH=$PATH:/usr/local/cuda-8.0/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64

export LC_ALL=C

# NOTE: set when using HTK toolkit
export HCOPY='/home/lab5/inaguma/htk-3.4/bin/HCopy'

ln -s $KALDI_ROOT/egs/wsj/s5/steps .
ln -s $KALDI_ROOT/egs/wsj/s5/utils .

export PYTHONIOENCODING='utf-8'


### Below are the paths used by the optional parts of the recipe

# We only need the Festival stuff below for the optional text normalization(for LM-training) step
FEST_ROOT=tools/festival
NSW_PATH=${FEST_ROOT}/festival/bin:${FEST_ROOT}/nsw/bin
export PATH=$PATH:$NSW_PATH

# SRILM is needed for LM model building
SRILM_ROOT=$KALDI_ROOT/tools/srilm
SRILM_PATH=$SRILM_ROOT/bin:$SRILM_ROOT/bin/i686-m64
export PATH=$PATH:$SRILM_PATH

# Sequitur G2P executable
sequitur=$KALDI_ROOT/tools/sequitur/g2p.py
sequitur_path="$(dirname $sequitur)/lib/$PYTHON/site-packages"

# Directory under which the LM training corpus should be extracted
LM_CORPUS_ROOT=./lm-corpus
