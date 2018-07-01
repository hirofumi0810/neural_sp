#!/bin/bash

. ./path.sh
set -e

### Select GPU
if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./gen_a2p_feat.sh path_to_A2P gpu_id" 1>&2
  exit 1
fi

beam_width=1
length_penalty=0
coverage_penalty=0

CUDA_VISIBLE_DEVICES=$2 ${PYTHON} ../../../src/bin/feature_generation/generate_a2p_feature.py \
  --corpus ${corpus} \
  --data_type train \
  --data_save_path ${data} \
  --model_path $1 \
  --epoch -1 \
  --eval_batch_size 1 \
  --beam_width ${beam_width} \
  --length_penalty ${length_penalty} \
  --coverage_penalty ${coverage_penalty}
