#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

### Select GPU
if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./decode.sh model_path gpu_id" 1>&2
  exit 1
fi

model=$1
gpu_id=$2

### path to save dataset
data=/n/sd8/inaguma/corpus/swbd

epoch=-1
batch_size=1
beam_width=5
max_len_ratio=1
min_len_ratio=0
length_penalty=0
coverage_penalty=0.6
coverage_threshold=0
rnnlm_weight=0
rnnlm=

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail


# for eval_set in eval2000 rt03; do
for eval_set in eval2000; do
  decode_dir=`CUDA_VISIBLE_DEVICES=${gpu_id} ../../../neural_sp/bin/asr/eval.py \
    --eval_sets ${data}/dataset/${eval_set}_word15000.csv \
    --model ${model} \
    --epoch ${epoch} \
    --batch_size ${batch_size} \
    --beam_width ${beam_width} \
    --max_len_ratio ${max_len_ratio} \
    --min_len_ratio ${min_len_ratio} \
    --length_penalty ${length_penalty} \
    --coverage_penalty ${coverage_penalty} \
    --coverage_threshold ${coverage_threshold} \
    --rnnlm_weight ${rnnlm_weight} \
    --rnnlm ${rnnlm}`

  local/score_sclite.sh ${data} ${decode_dir} ${eval_set}
done
