#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

model=
gpu=

### path to save preproecssed data
data=/n/sd8/inaguma/corpus/wsj

recog_unit=
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
    if [ ! -z ${recog_unit} ]; then
        recog_dir=${recog_dir}_${recog_unit}
    fi
    mkdir -p ${recog_dir}

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/asr/plot_ctc.py \
        --recog_sets ${data}/dataset/${set}_wpbpe1000.tsv \
        --recog_dir ${recog_dir} \
        --recog_unit ${recog_unit} \
        --recog_model ${model} \
        --recog_batch_size ${batch_size} \
        || exit 1;
done
