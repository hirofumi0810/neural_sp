#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

model=
gpu=

### path to save preproecssed data
data=/n/sd8/inaguma/corpus/swbd

epoch=-1
batch_size=1
recog_unit=

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
gpu=`echo ${gpu} | cut -d "," -f 1`

for set in eval2000; do
  recog_dir=${model}/plot_${set}_ep${epoch}
  if [ ! -z ${recog_unit} ]; then
      recog_dir=${recog_dir}_${recog_unit}
  fi
  mkdir -p ${recog_dir}

  if [ `echo ${model} | grep 'train_sp'` ]; then
    if [ `echo ${model} | grep 'fisher_swbd'` ]; then
      recog_set=${data}/dataset/${set}_sp_fisher_swbd_wpbpe30000.csv
    else
      recog_set=${data}/dataset/${set}_sp_swbd_wpbpe10000.csv
    fi
  else
    if [ `echo ${model} | grep 'fisher_swbd'` ]; then
      recog_set=${data}/dataset/${set}_fisher_swbd_wpbpe30000.csv
    else
      recog_set=${data}/dataset/${set}_swbd_wpbpe10000.csv
    fi
  fi

  CUDA_VISIBLE_DEVICES=${gpu} ../../../neural_sp/bin/asr/plot_ctc.py \
    --recog_sets ${recog_set} \
    --recog_model ${model} \
    --recog_epoch ${epoch} \
    --recog_batch_size ${batch_size} \
    --recog_unit ${recog_unit} \
    --recog_dir ${recog_dir} || exit 1;
done
