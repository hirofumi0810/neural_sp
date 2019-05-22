#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

model=
gpu=

### path to save preproecssed data
data=/n/sd3/inaguma/corpus/ptb

batch_size=1
n_caches=100
cache_theta=0.1
cache_lambda=0.1

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

if [ -z ${gpu} ]; then
    echo "Error: set GPU number." 1>&2
    echo "Usage: local/plot_lm_cache.sh --gpu 0" 1>&2
    exit 1
fi
gpu=$(echo ${gpu} | cut -d "," -f 1)

for set in test; do
    recog_dir=$(dirname ${model})/plot_${set}
    if [ ${n_caches} != 0 ]; then
        recog_dir=${recog_dir}_cache${n_caches}_theta${cache_theta}_lambda${cache_lambda}
    fi
    mkdir -p ${recog_dir}

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/lm/plot_cache.py \
        --recog_sets ${data}/dataset/${set}_word10000.tsv \
        --recog_model ${model} \
        --recog_batch_size ${batch_size} \
        --recog_n_caches ${n_caches} \
        --recog_cache_theta ${cache_theta} \
        --recog_cache_lambda ${cache_lambda} \
        --recog_dir ${recog_dir} || exit 1;
done
