#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

model=
model_bwd=
gpu=

### path to save preproecssed data
data=/n/sd8/inaguma/corpus/csj

epoch=-1
batch_size=1
beam_width=5
min_len_ratio=0.0
max_len_ratio=1.0
length_penalty=0.0
coverage_penalty=0.6
coverage_threshold=0.0
rnnlm=
rnnlm_bwd=
rnnlm_weight=0.0
resolving_unk=0
fwd_bwd_attention=0
recog_unit=

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

for set in eval1 eval2 eval3; do
  plot_dir=${model}/plot_${set}_ep${epoch}_beam${beam_width}_lp${length_penalty}_cp${coverage_penalty}_${min_len_ratio}_${max_len_ratio}_rnnlm${rnnlm_weight}
  if [ ! -z ${recog_unit} ]; then
      plot_dir=${plot_dir}_${recog_unit}
  fi
  mkdir -p ${plot_dir}

  CUDA_VISIBLE_DEVICES=${gpu} ../../../neural_sp/bin/asr/plot_attention.py \
    --recog_sets ${data}/dataset/${set}_aps_other_word12500.csv \
    --recog_model ${model} \
    --recog_model_bwd ${model_bwd} \
    --recog_epoch ${epoch} \
    --recog_batch_size ${batch_size} \
    --beam_width ${beam_width} \
    --max_len_ratio ${max_len_ratio} \
    --min_len_ratio ${min_len_ratio} \
    --length_penalty ${length_penalty} \
    --coverage_penalty ${coverage_penalty} \
    --coverage_threshold ${coverage_threshold} \
    --rnnlm ${rnnlm} \
    --rnnlm_bwd ${rnnlm_bwd} \
    --rnnlm_weight ${rnnlm_weight} \
    --resolving_unk ${resolving_unk} \
    --fwd_bwd_attention ${fwd_bwd_attention} \
    --recog_unit ${recog_unit} \
    --plot_dir ${plot_dir} || exit 1;
done
