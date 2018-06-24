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
echo "                              Switchboard                                 "
echo ============================================================================

stage=0
run_background=true
# run_background=false

### Set path to original data
SWBD_AUDIOPATH="/n/rd21/corpora_7/swb"
EVAL2000_AUDIOPATH="/n/rd21/corpora_7/hub5_english/LDC2002S09"
EVAL2000_TRANSPATH="/n/rd21/corpora_7/hub5_english/LDC2002T43"
FISHER_PATH="/n/rd7/fisher_english"

### Set path to save the model
model="/n/sd8/inaguma/result/swbd"

# train_set=train
train_set=train_fisher

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
        --train_set ${train_set} \
        --dev_set dev \
        --eval_sets eval2000_swbd eval2000_ch \
        --saved_model_path ${config_path} \
        --data_save_path ${data} > log/${filename}".log" &
    else
      CUDA_VISIBLE_DEVICES=${gpu_ids} \
      ${PYTHON} ../../../src/bin/training/train_lm.py \
        --corpus ${corpus} \
        --gpu_ids ${gpu_ids} \
        --train_set ${train_set} \
        --dev_set dev \
        --eval_sets eval2000_swbd eval2000_ch \
        --saved_model_path ${config_path} \
        --data_save_path ${data} || exit 1;
    fi
  else
    if $run_background; then
      CUDA_VISIBLE_DEVICES=${gpu_ids} \
      nohup ${PYTHON} ../../../src/bin/training/train_lm.py \
        --corpus ${corpus} \
        --gpu_ids ${gpu_ids} \
        --train_set ${train_set} \
        --dev_set dev \
        --eval_sets eval2000_swbd eval2000_ch \
        --config_path ${config_path} \
        --model_save_path ${model} \
        --data_save_path ${data} > log/${filename}".log" &
    else
      CUDA_VISIBLE_DEVICES=${gpu_ids} \
      ${PYTHON} ../../../src/bin/training/train_lm.py \
        --corpus ${corpus} \
        --gpu_ids ${gpu_ids} \
        --train_set ${train_set} \
        --dev_set dev \
        --eval_sets eval2000_swbd eval2000_ch \
        --config_path ${config_path} \
        --model_save_path ${model} \
        --data_save_path ${data} || exit 1;
    fi
  fi

  echo "Finish LM training (stage: 4)."
fi


echo "Done."
