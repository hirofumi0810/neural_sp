#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

# . utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./run_rnnlm.sh path_to_config_file gpu_ids" 1>&2
  exit 1
fi


echo ============================================================================
echo "                                   WSJ                                     "
echo ============================================================================

stage=3
run_background=true
# run_background=false

### Set path to original data
wsj0="/n/rd21/corpora_1/WSJ/wsj0"
wsj1="/n/rd21/corpora_1/WSJ/wsj1"

# Sometimes, we have seen WSJ distributions that do not have subdirectories
# like '11-13.1', but instead have 'doc', 'si_et_05', etc. directly under the
# wsj0 or wsj1 directories. In such cases, try the following:
CSTR_WSJTATATOP="/n/rd21/corpora_1/WSJ"
# $CSTR_WSJTATATOP must contain a 'wsj0' and a 'wsj1' subdirectory for this to work.

directory_type=cstr # or original

### Set path to save the model
model="/n/sd8/inaguma/result/wsj"


if [ ${stage} -le 0 ] && [ ! -e ${data}/.stage_0 ]; then
  echo "Run ./run.sh at first"
  exit 1;
fi


if [ ${stage} -le 1 ] && [ ! -e ${data}/.stage_1 ]; then
  echo "Run ./run.sh at first"
  exit 1;
fi


if [ ${stage} -le 2 ] && [ ! -e ${data}/.stage_2 ]; then
  echo "Run ./run.sh at first"
  exit 1;
fi


if [ ${stage} -le 3 ]; then
  echo ============================================================================
  echo "                             LM training                                 "
  echo ============================================================================

  config_path=$1
  gpu_ids=$2
  filename=$(basename ${config_path} | awk -F. '{print $1}')

  mkdir -p log
  mkdir -p ${model}

  echo "Start training..."

  if [ `echo ${config_path} | grep 'result'` ]; then
    if $run_background; then
      CUDA_VISIBLE_DEVICES=${gpu_ids} \
      nohup ${PYTHON} ../../../src/bin/training/train_lm.py \
        --corpus ${corpus} \
        --gpu_ids ${gpu_ids} \
        --train_set train_lm \
        --dev_set test_dev93 \
        --eval_sets test_eval92 \
        --saved_model_path ${config_path} \
        --data_save_path ${data} > log/${filename}".log" &
    else
      CUDA_VISIBLE_DEVICES=${gpu_ids} \
      ${PYTHON} ../../../src/bin/training/train_lm.py \
        --corpus ${corpus} \
        --gpu_ids ${gpu_ids} \
        --train_set train_lm \
        --dev_set test_dev93 \
        --eval_sets test_eval92 \
        --saved_model_path ${config_path} \
        --data_save_path ${data} || exit 1;
    fi
  else
    if $run_background; then
      CUDA_VISIBLE_DEVICES=${gpu_ids} \
      nohup ${PYTHON} ../../../src/bin/training/train_lm.py \
        --corpus ${corpus} \
        --gpu_ids ${gpu_ids} \
        --train_set train_lm \
        --dev_set test_dev93 \
        --eval_sets test_eval92 \
        --config_path ${config_path} \
        --model_save_path ${model} \
        --data_save_path ${data} > log/${filename}".log" &
    else
      CUDA_VISIBLE_DEVICES=${gpu_ids} \
      ${PYTHON} ../../../src/bin/training/train_lm.py \
        --corpus ${corpus} \
        --gpu_ids ${gpu_ids} \
        --train_set train_lm \
        --dev_set test_dev93 \
        --eval_sets test_eval92 \
        --config_path ${config_path} \
        --model_save_path ${model} \
        --data_save_path ${data} || exit 1;
    fi
  fi

  echo "Finish LM training (stage: 4)."
fi


echo "Done."
