#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

# . utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./run_rnnlm.sh path_to_config_file gpu_index or ./run.sh path_to_saved_model gpu_index" 1>&2
  exit 1
fi


echo ============================================================================
echo "                              LibriSpeech                                 "
echo ============================================================================

stage=0
run_background=true

### Select data size
# datasize=100
# datasize=460
datasize=960

### Set path to save the model
model="/n/sd8/inaguma/result/librispeech"

### Set path to save dataset
export data="/n/sd8/inaguma/corpus/librispeech/kaldi"

if [ ${stage} -le 0 ] && [ ! -e ${data}/.stage_0_${datasize} ]; then
  exit "Run ./run.sh at first"
  exit 1;
fi


if [ ${stage} -le 1 ] && [ ! -e ${data}/.stage_1_${datasize} ]; then
  exit "Run ./run.sh at first"
  exit 1;
fi


if [ ${stage} -le 2 ] && [ ! -e ${data}/.stage_2_${datasize} ]; then
  exit "Run ./run.sh at first"
  exit 1;
fi


if [ ${stage} -le 3 ]; then
  echo ============================================================================
  echo "                             LM training                                 "
  echo ============================================================================

  config_path=$1
  gpu_id=$2
  filename=$(basename ${config_path} | awk -F. '{print $1}')

  mkdir -p log
  mkdir -p ${model}

  echo "Start training..."

  if [ `echo ${config_path} | grep 'result'` ]; then
    if $run_background; then
      CUDA_VISIBLE_DEVICES=${gpu_id} \
      nohup $PYTHON exp/training/train_lm.py \
        --gpu ${gpu_id} \
        --saved_model_path ${config_path} \
        --data_save_path ${data} > log/${filename}".log" &
    else
      CUDA_VISIBLE_DEVICES=${gpu_id} \
      $PYTHON exp/training/train_lm.py \
        --gpu ${gpu_id} \
        --saved_model_path ${config_path} \
        --data_save_path ${data} || exit 1;
    fi
  else
    if $run_background; then
      CUDA_VISIBLE_DEVICES=${gpu_id} \
      nohup $PYTHON exp/training/train_lm.py \
        --gpu ${gpu_id} \
        --config_path ${config_path} \
        --model_save_path ${model} \
        --data_save_path ${data} > log/${filename}".log" &
    else
      CUDA_VISIBLE_DEVICES=${gpu_id} \
      $PYTHON exp/training/train_lm.py \
        --gpu ${gpu_id} \
        --config_path ${config_path} \
        --model_save_path ${model} \
        --data_save_path ${data} || exit 1;
    fi
  fi

  echo "Finish LM training (stage: 4)."
fi


echo "Done."
