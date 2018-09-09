### Set the root path of kaldi-asr
if [ -z $KALDI_ROOT ]; then
  export KALDI_ROOT="/n/sd8/inaguma/kaldi"
fi
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
NEURALSP_ROOT=../../..

export PATH=$PWD/utils/:$NEURALSP_ROOT/neural_sp/utils/bin/:$KALDI_ROOT/tools/sctk/bin/:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

### Python
source ~/espnet/tools/venv/bin/activate
export PYTHONPATH=$NEURALSP_ROOT/:~/espnet/tools/kaldi-io-for-python/:$PYTHONPATH

### CUDA
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64

export PYTHONDONTWRITEBYTECODE=1
find $NEURALSP_ROOT -name "*.pyc" -exec rm -f {} \;
