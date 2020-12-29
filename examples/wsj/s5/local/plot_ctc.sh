#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

model=
model1=
model2=
model3=
gpu=
stdout=false
n_threads=1

### path to save preproecssed data
data=/n/work2/inaguma/corpus/wsj

unit=
batch_size=1

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

for set in test_dev93 test_eval92; do
    recog_dir=$(dirname ${model})/plot_${set}
    if [ ! -z ${unit} ]; then
        recog_dir=${recog_dir}_${unit}
    fi
    if [ ! -z ${model3} ]; then
        recog_dir=${recog_dir}_ensemble4
    elif [ ! -z ${model2} ]; then
        recog_dir=${recog_dir}_ensemble3
    elif [ ! -z ${model1} ]; then
        recog_dir=${recog_dir}_ensemble2
    fi
    mkdir -p ${recog_dir}

    if [ $(echo ${model} | grep 'train_si284_sp') ]; then
        recog_set=${data}/dataset/${set}_si284_sp_wpbpe1000.tsv
    elif [ $(echo ${model} | grep 'train_si284') ]; then
        recog_set=${data}/dataset/${set}_si284_wpbpe1000.tsv
    elif [ $(echo ${model} | grep 'train_si84_sp') ]; then
        recog_set=${data}/dataset/${set}_si84_sp_char.tsv
    elif [ $(echo ${model} | grep 'train_si84') ]; then
        recog_set=${data}/dataset/${set}_si84_char.tsv
    fi

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/asr/plot_ctc.py \
        --recog_n_gpus ${n_gpus} \
        --recog_sets ${recog_set} \
        --recog_dir ${recog_dir} \
        --recog_unit ${unit} \
        --recog_model ${model} ${model1} ${model2} ${model3} \
        --recog_batch_size ${batch_size} \
        --recog_stdout ${stdout} || exit 1;
done
