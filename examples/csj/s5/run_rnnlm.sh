#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

# . utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./run.sh path_to_config_file gpu_index or ./run.sh path_to_saved_model gpu_index" 1>&2
  exit 1
fi


echo ============================================================================
echo "                                   CSJ                                     "
echo ============================================================================

stage=0
run_background=true

### Select data size
export DATASIZE=aps_other
# export DATASIZE=aps
# export DATASIZE=all_except_dialog
# export DATASIZE=all

# NOTE:
# aps_other=default using "Academic lecture" and "other" data,
# aps=using "Academic lecture" data,
# all_except_dialog=using All data except for "dialog" data,
# all=using All data

### Set path to save the model
MODEL="/n/sd8/inaguma/result/csj"

### Set path to save dataset
export DATA="/n/sd8/inaguma/corpus/csj/kaldi"

train=train_$DATASIZE

if [ $stage -le 0 ] && [ ! -e $DATA/.stage_0_$DATASIZE ]; then
  exit "Run ./run.sh at first"
  exit 1;
fi


if [ $stage -le 1 ] && [ ! -e $DATA/.stage_1_$DATASIZE ]; then
  exit "Run ./run.sh at first"
  exit 1;
fi


if [ $stage -le 2 ] && [ ! -e $DATA/.stage_2_$DATASIZE ]; then
  exit "Run ./run.sh at first"
  exit 1;
fi


if [ $stage -le 3 ]; then
  echo ============================================================================
  echo "                             LM training                                 "
  echo ============================================================================

  config_path=$1
  gpu_index=$2
  filename=$(basename $config_path | awk -F. '{print $1}')

  mkdir -p log
  mkdir -p $MODEL

  echo "Start training..."

  if [ `echo $config_path | grep 'result'` ]; then
    if $run_background; then
      CUDA_VISIBLE_DEVICES=$gpu_index \
      nohup $PYTHON exp/training/train_lm.py \
        --gpu $gpu_index \
        --saved_model_path $config_path \
        --data_save_path $DATA > log/$filename".log" &
    else
      CUDA_VISIBLE_DEVICES=$gpu_index \
      $PYTHON exp/training/train_lm.py \
        --gpu $gpu_index \
        --saved_model_path $config_path \
        --data_save_path $DATA || exit 1;
    fi
  else
    if $run_background; then
      CUDA_VISIBLE_DEVICES=$gpu_index \
      nohup $PYTHON exp/training/train_lm.py \
        --gpu $gpu_index \
        --config_path $config_path \
        --model_save_path $MODEL \
        --data_save_path $DATA > log/$filename".log" &
    else
      CUDA_VISIBLE_DEVICES=$gpu_index \
      $PYTHON exp/training/train_lm.py \
        --gpu $gpu_index \
        --config_path $config_path \
        --model_save_path $MODEL \
        --data_save_path $DATA || exit 1;
    fi
  fi

  echo "Finish LM training (stage: 4)."
fi


echo "Done."
