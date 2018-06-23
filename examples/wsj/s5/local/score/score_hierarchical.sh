#!/bin/bash

. ./path.sh
set -e

### Select GPU
if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./score_hierarchical.sh path_to_saved_model gpu_id" 1>&2
  exit 1
fi

beam_width=4
beam_width_sub=4
length_penalty=0.1
coverage_penalty=0

CUDA_VISIBLE_DEVICES=$2 ${PYTHON} ../../../src/bin/evaluation/eval_hierarchical.py \
  --corpus ${corpus} \
  --eval_sets test_dev93 test_eval92 \
  --data_save_path ${data} \
  --model_path $1 \
  --epoch -1 \
  --eval_batch_size 1 \
  --beam_width ${beam_width} \
  --beam_width_sub ${beam_width_sub} \
  --length_penalty ${length_penalty} \
  --coverage_penalty ${coverage_penalty}
