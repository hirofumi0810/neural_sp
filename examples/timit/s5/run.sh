#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./run.sh path_to_config_file gpu_ids" 1>&2
  exit 1
fi


echo ============================================================================
echo "                                  TIMIT                                    "
echo ============================================================================

stage=0
run_background=true
# run_background=false

### Set path to original data
TIMITDATATOP="/n/rd21/corpora_1/TIMIT"

### Set path to save dataset
export data="/n/sd8/inaguma/corpus/timit/kaldi"

### Set path to save the model
model="/n/sd8/inaguma/result/timit"

### Select one tool to extract features (HTK is the fastest)
# tool=kaldi
tool=htk
# tool=python_speech_features
# tool=librosa

### Configuration of feature extranction
channels=40
window=0.025
slide=0.01
energy=1
delta=1
deltadelta=1
# normalize=global
normalize=speaker
# normalize=utterance


if [ ${stage} -le 0 ] && [ ! -e ${data}/.stage_0 ]; then
  echo ============================================================================
  echo "                           Data Preparation                               "
  echo ============================================================================

  # Data preparation based on kaldi-asr
  local/timit_data_prep.sh ${TIMITDATATOP} || exit 1;
  local/timit_format_data.sh || exit 1;

  touch ${data}/.stage_0
  echo "Finish data preparation (stage: 0)."
fi


if [ ${stage} -le 1 ] && [ ! -e ${data}/.stage_1 ]; then
  echo ============================================================================
  echo "                        Feature extranction                               "
  echo ============================================================================

  if [ ${tool} = 'kaldi' ]; then
    for x in train dev test; do
      steps/make_fbank.sh --nj 8 --cmd run.pl ${data}/$x ${data}/make_fbank/$x ${data}/fbank || exit 1;
      steps/compute_cmvn_stats.sh ${data}/$x ${data}/make_fbank/$x ${data}/fbank || exit 1;
      utils/fix_data_dir.sh ${data}/$x || exit 1;
    done

  elif [ ${tool} = 'htk' ]; then
    # Make a config file to covert from wav to htk file
    ${PYTHON} local/make_htk_config.py \
        --data_save_path ${data} \
        --config_save_path ./conf/fbank_htk.conf \
        --audio_file_type nist \
        --channels ${channels} \
        --sampling_rate 16000 \
        --window ${window} \
        --slide ${slide} \
        --energy ${energy} \
        --delta ${delta} \
        --deltadelta ${deltadelta} || exit 1;

    for data_type in train dev test ; do
      mkdir -p ${data}/htk/$data_type
      [ -e ${data}/$data_type/htk.scp ] && rm ${data}/$data_type/htk.scp
      touch ${data}/$data_type/htk.scp
      cat ${data}/$data_type/wav.scp | while read line
      do
        # Convert from wav to htk files
        wav_path=`echo $line | awk -F " " '{ print $(NF - 1) }'`
        speaker=`echo $line | awk -F "/" '{ print $(NF - 1) }'`
        mkdir -p ${data}/htk/$data_type/$speaker
        file_name=`basename $wav_path`
        base=${file_name%.*}
        # ext=${file_name##*.}
        htk_path=${data}/htk/$data_type/$speaker/$base".htk"
        if [ ! -e $htk_path ]; then
          echo $wav_path $htk_path > ./tmp.scp
          $HCOPY -T 1 -C ./conf/fbank_htk.conf -S ./tmp.scp || exit 1;
          rm ./tmp.scp
        fi
        echo $htk_path >> ${data}/$data_type/htk.scp
      done
    done

  else
    if ! which sox >&/dev/null; then
      echo "This script requires you to first install sox";
      exit 1;
    fi
  fi

  ${PYTHON} local/feature_extraction.py \
    --data_save_path ${data} \
    --tool ${tool} \
    --normalize ${normalize} \
    --channels ${channels} \
    --window ${window} \
    --slide ${slide} \
    --energy ${energy} \
    --delta ${delta} \
    --deltadelta ${deltadelta} || exit 1;

  touch ${data}/.stage_1
  echo "Finish feature extranction (stage: 1)."
fi


if [ ${stage} -le 2 ] && [ ! -e ${data}/.stage_2 ]; then
  echo ============================================================================
  echo "                            Create dataset                                "
  echo ============================================================================

  ${PYTHON} local/make_dataset_csv.py \
    --data_save_path ${data} \
    --phone_map_file_path ./conf/phones.60-48-39.map \
    --tool ${tool} || exit 1;

  touch ${data}/.stage_2
  echo "Finish creating dataset (stage: 2)."
fi


if [ ${stage} -le 3 ]; then
  echo ============================================================================
  echo "                             Training stage                               "
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
      nohup ${PYTHON} ../../../src/bin/training/train.py \
        --corpus ${corpus} \
        --gpu_ids ${gpu_ids} \
        --train_set train \
        --dev_set dev \
        --eval_sets test \
        --saved_model_path ${config_path} \
        --data_save_path ${data} > log/${filename}".log" &
    else
      CUDA_VISIBLE_DEVICES=${gpu_ids} \
      ${PYTHON} ../../../src/bin/training/train.py \
        --corpus ${corpus} \
        --gpu_ids ${gpu_ids} \
        --train_set train \
        --dev_set dev \
        --eval_sets test \
        --saved_model_path ${config_path} \
        --data_save_path ${data} || exit 1;
    fi
  else
    if $run_background; then
      CUDA_VISIBLE_DEVICES=${gpu_ids} \
      nohup ${PYTHON} ../../../src/bin/training/train.py \
        --corpus ${corpus} \
        --gpu_ids ${gpu_ids} \
        --train_set train \
        --dev_set dev \
        --eval_sets test \
        --config_path ${config_path} \
        --model_save_path ${model} \
        --data_save_path ${data} > log/${filename}".log" &
    else
      CUDA_VISIBLE_DEVICES=${gpu_ids} \
      ${PYTHON} ../../../src/bin/training/train.py \
        --corpus ${corpus} \
        --gpu_ids ${gpu_ids} \
        --train_set train \
        --dev_set dev \
        --eval_sets test \
        --config_path ${config_path} \
        --model_save_path ${model} \
        --data_save_path ${data} || exit 1;
    fi
  fi

  echo "Finish model training (stage: 3)."
fi


echo "Done."
