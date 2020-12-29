#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

model=
gpu=
n_threads=1

### path to save preproecssed data
data=/n/work2/inaguma/inaguma/corpus/swbd

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
    # CPU
    n_gpus=0
    export OMP_NUM_THREADS=${n_threads}
else
    n_gpus=$(echo ${gpu} | tr "," "\n" | wc -l)
fi

for set in eval2000_swbd; do
    recog_dir=$(dirname ${model})/plot_${set}
    if [ ${n_caches} != 0 ]; then
        recog_dir=${recog_dir}_cache${n_caches}_theta${cache_theta}_lambda${cache_lambda}
    fi
    mkdir -p ${recog_dir}

    if [ $(echo ${model} | grep 'fisher_swbd_train_fisher_swbd') ]; then
        recog_set=${data}/dataset_lm/${set}_fisher_swbd_train_fisher_swbd_wpbpe30000.tsv
    elif [ $(echo ${model} | grep 'fisher_swbd_train_swbd') ]; then
        recog_set=${data}/dataset_lm/${set}_fisher_swbd_train_swbd_wpbpe10000.tsv
    elif [ $(echo ${model} | grep 'fisher_swbd_train_fisher_swbd') ]; then
        recog_set=${data}/dataset_lm/${set}_swbd_train_swbd_wpbpe10000.tsv
    fi

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/lm/plot_cache.py \
        --recog_sets ${recog_set} \
        --recog_model ${model} \
        --recog_batch_size ${batch_size} \
        --recog_n_caches ${n_caches} \
        --recog_cache_theta ${cache_theta} \
        --recog_cache_lambda ${cache_lambda} \
        --recog_dir ${recog_dir} || exit 1;
done
