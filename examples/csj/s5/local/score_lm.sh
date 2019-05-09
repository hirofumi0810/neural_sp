#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

model=
gpu=

### path to save preproecssed data
data=/n/sd8/inaguma/corpus/csj

batch_size=1
n_caches=0
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
    echo "Usage: local/score.sh --gpu 0" 1>&2
    exit 1
fi
gpu=$(echo ${gpu} | cut -d "," -f 1)

for set in eval1 eval2 eval3; do
    recog_dir=$(dirname ${model})/decode_${set}
    if [ ${n_caches} != 0 ]; then
        recog_dir=${recog_dir}_cache${n_caches}_theta${cache_theta}_lambda${cache_lambda}
    fi
    mkdir -p ${recog_dir}

    if [ $(echo ${model} | grep 'all') ]; then
        if [ $(echo ${model} | grep 'aps_other') ]; then
            recog_set=${data}/dataset_lm/${set}_all_train_aps_other_wpbpe10000.tsv
        elif [ $(echo ${model} | grep 'sps') ]; then
            recog_set=${data}/dataset_lm/${set}_all_train_sps_wpbpe10000.tsv
        else
            recog_set=${data}/dataset_lm/${set}_all_train_all_wpbpe10000.tsv
        fi
    elif [ $(echo ${model} | grep 'aps_other') ]; then
        recog_set=${data}/dataset_lm/${set}_aps_other_train_aps_other_wpbpe10000.tsv
    elif [ $(echo ${model} | grep 'sps') ]; then
        recog_set=${data}/dataset_lm/${set}_sps_train_sps_wpbpe10000.tsv
    fi

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/lm/eval.py \
        --recog_sets ${recog_set} \
        --recog_model ${model} \
        --recog_batch_size ${batch_size} \
        --recog_n_caches ${n_caches} \
        --recog_cache_theta ${cache_theta} \
        --recog_cache_lambda ${cache_lambda} \
        --recog_dir ${recog_dir} || exit 1;
done
