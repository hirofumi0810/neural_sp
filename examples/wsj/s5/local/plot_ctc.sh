#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

model=
model1=
model2=
model3=
gpu=
stdout=false

### path to save preproecssed data
data=/n/sd3/inaguma/corpus/wsj

unit=
batch_size=1

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

if [ -z ${gpu} ]; then
    echo "Error: set GPU number." 1>&2
    echo "Usage: local/plot_ctc.sh --gpu 0" 1>&2
    exit 1
fi
gpu=$(echo ${gpu} | cut -d "," -f 1)

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
    else
        exit 1
    fi

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/asr/plot_ctc.py \
        --recog_sets ${recog_set} \
        --recog_dir ${recog_dir} \
        --recog_unit ${unit} \
        --recog_model ${model} ${model1} ${model2} ${model3} \
        --recog_batch_size ${batch_size} \
        --recog_stdout ${stdout} || exit 1;
done
