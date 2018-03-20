if [ -z $KALDI_ROOT ]; then
  export KALDI_ROOT="/home/inaguma/tool/kaldi"
fi

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh

### python
# export PYTHON=/home/inaguma/.pyenv/versions/anaconda3-4.1.1/bin/python
export PYTHON=/home/inaguma/.pyenv/versions/anaconda3-4.1.1/envs/`hostname`/bin/python

### CUDA
# export PATH=$PATH:/usr/local/cuda/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib:/usr/local/lib64:/usr/local/cuda/bin/nvcc
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64

export LC_ALL=C

# NOTE: set when using HTK toolkit
export HCOPY='/home/inaguma/htk-3.4/bin/HCopy'

ln -s $KALDI_ROOT/egs/wsj/s5/steps .
ln -s $KALDI_ROOT/egs/wsj/s5/utils .

export PYTHONIOENCODING='utf-8'
