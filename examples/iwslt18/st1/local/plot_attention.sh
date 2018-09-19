#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

### Select GPU
if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./plot_attention.sh model_path gpu_id" 1>&2
  exit 1
fi

model=$1
gpu_id=$2

### vocabulary
unit=wordpiece
vocab_size=5000

# for wordpiece
wp_model_type=unigram  # or bpe

### path to save dataset
data=/n/sd8/inaguma/corpus/iwslt18

epoch=-1
batch_size=1
beam_width=1
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


for eval_set in dev; do
  CUDA_VISIBLE_DEVICES=${gpu_id} ../../../neural_sp/bin/asr/plot_attention.py \
    --eval_sets ${data}/dataset/dev.de_${unit}${wp_model_type}${vocab_size}.csv \
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
    --rnnlm ${rnnlm}
done
