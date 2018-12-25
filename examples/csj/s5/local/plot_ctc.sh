#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

model=
gpu=

### path to save preproecssed data
data=/n/sd8/inaguma/corpus/csj

epoch=-1
batch_size=1

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
  plot_dir=${model}/plot_${set}_ep${epoch}
  mkdir -p ${plot_dir}

  CUDA_VISIBLE_DEVICES=${gpu} ../../../neural_sp/bin/asr/plot_ctc.py \
    --eval_sets ${data}/dataset/${set}_aps_other_word12500.csv \
    --model ${model} \
    --epoch ${epoch} \
    --batch_size ${batch_size} \
    --plot_dir ${plot_dir} || exit 1;
done
