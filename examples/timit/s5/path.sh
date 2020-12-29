export NEURALSP_ROOT=$PWD/../../..
export TOOL=$NEURALSP_ROOT/tools/neural_sp
export CONDA=$TOOL/miniconda
export KALDI_ROOT=$TOOL/kaldi

# Kaldi
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$NEURALSP_ROOT/utils:$PWD/utils/:$KALDI_ROOT/tools/sctk/bin/:$TOOL/sentencepiece/build/src:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

### Python
source $CONDA/etc/profile.d/conda.sh && conda deactivate && conda activate
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1

### CUDA
CUDAROOT=/usr/local/cuda
NCCL_ROOT=/usr/local/nccl
export CPATH=$NCCL_ROOT/include:$CPATH
export LD_LIBRARY_PATH=$NCCL_ROOT/lib/:$CUDAROOT/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$NCCL_ROOT/lib/:$LIBRARY_PATH
export CUDA_HOME=$CUDAROOT
export CUDA_PATH=$CUDAROOT
