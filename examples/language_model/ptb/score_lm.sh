#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

model=
gpu=

### path to save preproecssed data
data=/n/work2/inaguma/corpus/ptb

batch_size=1
n_caches=0
cache_theta=0.1
cache_lambda=0.1
mem_len=0  # for TransformerXL
n_average=1  # for Transformer

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

if [ -z ${gpu} ]; then
    n_gpus=0
else
    n_gpus=$(echo ${gpu} | tr "," "\n" | wc -l)
fi

for set in valid test; do
    recog_dir=$(dirname ${model})/decode_${set}
    if [ ${n_caches} != 0 ]; then
        recog_dir=${recog_dir}_cache${n_caches}_theta${cache_theta}_lambda${cache_lambda}
    fi
    if [ ${n_average} != 1 ]; then
        recog_dir=${recog_dir}_average${n_average}
    fi
    if [ ${mem_len} != 0 ]; then
        recog_dir=${recog_dir}_mem${mem_len}
    fi
    mkdir -p ${recog_dir}

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/lm/eval.py \
        --recog_n_gpus ${n_gpus} \
        --recog_sets ${data}/dataset/${set}_word10000.tsv \
        --recog_model ${model} \
        --recog_batch_size ${batch_size} \
        --recog_n_caches ${n_caches} \
        --recog_cache_theta ${cache_theta} \
        --recog_cache_lambda ${cache_lambda} \
        --recog_mem_len ${mem_len} \
        --recog_n_average ${n_average} \
        --recog_dir ${recog_dir} || exit 1;
done
