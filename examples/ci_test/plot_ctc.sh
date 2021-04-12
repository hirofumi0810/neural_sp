#!/usr/bin/env bash

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

model=
gpu=
stdout=false
n_threads=1
eval_set="train"
cmd_coverage="coverage run -a"

### path to save preproecssed data
data=./data

batch_size=1
n_average=2  # for Transformer

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

if [ -z ${gpu} ]; then
    # CPU
    n_gpus=0
    export OMP_NUM_THREADS=${n_threads}
else
    n_gpus=$(echo ${gpu} | tr "," "\n" | wc -l)
fi

for set in ${eval_set}; do
    recog_dir=$(dirname ${model})/plot_${set}
    if [ ${n_average} != 1 ]; then
        recog_dir=${recog_dir}_average${n_average}
    fi
    mkdir -p ${recog_dir}

    CUDA_VISIBLE_DEVICES=${gpu} ${cmd_coverage} ${NEURALSP_ROOT}/neural_sp/bin/asr/plot_ctc.py \
        --recog_n_gpus ${n_gpus} \
        --recog_sets ${data}/dataset/${set}_char.tsv \
        --recog_dir ${recog_dir} \
        --recog_model ${model} \
        --recog_batch_size ${batch_size} \
        --recog_n_average ${n_average} \
        --recog_stdout ${stdout} || exit 1;
done
