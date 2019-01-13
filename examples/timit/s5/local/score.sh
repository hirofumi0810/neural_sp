#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

model=
gpu=

### path to save preproecssed data
data=/n/sd8/inaguma/corpus/timit

epoch=-1
batch_size=1
beam_width=5
min_len_ratio=0.0
max_len_ratio=1.0
length_penalty=0.0
coverage_penalty=0.6
coverage_threshold=0.0

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

if [ -z ${gpu} ]; then
  echo "Error: set GPU number." 1>&2
  echo "Usage: ./run.sh --gpu 0" 1>&2
  exit 1
fi
gpu=`echo ${gpu} | cut -d "," -f 1`

for set in dev test; do
  recog_dir=${model}/decode_${set}_ep${epoch}_beam${beam_width}_lp${length_penalty}_cp${coverage_penalty}_${min_len_ratio}_${max_len_ratio}
  mkdir -p ${recog_dir}

  CUDA_VISIBLE_DEVICES=${gpu} ../../../neural_sp/bin/asr/eval.py \
    --recog_sets ${data}/dataset/${set}.csv \
    --recog_model ${model} \
    --recog_epoch ${epoch} \
    --recog_batch_size ${batch_size} \
    --recog_beam_width ${beam_width} \
    --recog_max_len_ratio ${max_len_ratio} \
    --recog_min_len_ratio ${min_len_ratio} \
    --recog_length_penalty ${length_penalty} \
    --recog_coverage_penalty ${coverage_penalty} \
    --recog_coverage_threshold ${coverage_threshold} \
    --recog_dir ${recog_dir} || exit 1;

  echo ${set}
  local/score_sclite.sh ${recog_dir} > ${recog_dir}/RESULTS
  cat ${recog_dir}/RESULTS
done
