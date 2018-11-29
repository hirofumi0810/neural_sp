#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


model=$1
gpu_id=$2

### path to save preproecssed data
data=/n/sd8/inaguma/corpus/librispeech

epoch=-1
batch_size=1
beam_width=5
min_len_ratio=0.0
max_len_ratio=1.0
length_penalty=0.0
coverage_penalty=0.6
coverage_threshold=0.0
rnnlm=
rnnlm_rev=
rnnlm_weight=0.0
resolving_unk=0
fwd_bwd_attention=0

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

for eval_set in dev_clean dev_other test_clean test_other; do
  decode_dir=${model}/decode_${eval_set}_ep${epoch}_beam${beam_width}_lp${length_penalty}_cp${coverage_penalty}_${min_len_ratio}_${max_len_ratio}_rnnlm${rnnlm_weight}
  if [ ! -z ${fwd_bwd_attention} ]; then
    decode_dir=${decode_dir}_fwdbwd
  fi
  mkdir -p ${decode_dir}

  CUDA_VISIBLE_DEVICES=${gpu} ../../../neural_sp/bin/asr/eval.py \
    --eval_sets ${data}/dataset_csv/${eval_set}_960_word30000.csv \
    --model ${model} \
    --model_rev ${model_rev} \
    --epoch ${epoch} \
    --batch_size ${batch_size} \
    --beam_width ${beam_width} \
    --max_len_ratio ${max_len_ratio} \
    --min_len_ratio ${min_len_ratio} \
    --length_penalty ${length_penalty} \
    --coverage_penalty ${coverage_penalty} \
    --coverage_threshold ${coverage_threshold} \
    --rnnlm ${rnnlm} \
    --rnnlm_rev ${rnnlm_rev} \
    --rnnlm_weight ${rnnlm_weight} \
    --resolving_unk ${resolving_unk} \
    --fwd_bwd_attention ${fwd_bwd_attention} \
    --decode_dir ${decode_dir} || exit 1;

    sclite -r ${decode_dir}/ref.trn trn -h ${decode_dir}/hyp.trn trn -i rm -o all stdout > ${decode_dir}/result.txt
    grep -e Avg -e SPKR -m 2 ${decode_dir}/result.txt > ${decode_dir}/RESULTS
    cat ${decode_dir}/RESULTS
done
