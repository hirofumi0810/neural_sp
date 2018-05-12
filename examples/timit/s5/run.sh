#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./run.sh path_to_config_file gpu_index" 1>&2
  exit 1
fi


echo ============================================================================
echo "                                  TIMIT                                    "
echo ============================================================================

stage=2
run_background=true
restart=false

### Set path to original data
timit="/n/sd8/inaguma/corpus/timit/data"

### Set path to save dataset
export DATA_SAVEPATH="/n/sd8/inaguma/corpus/timit/kaldi"

### Set path to save the model
MODEL_SAVEPATH="/n/sd8/inaguma/result/timit"

### Select one tool to extract features (HTK is the fastest)
# TOOL=kaldi
TOOL=htk
# TOOL=python_speech_features
# TOOL=librosa

### Configuration of feature extranction
CHANNELS=40
WINDOW=0.025
SLIDE=0.01
ENERGY=1
DELTA=1
DELTADELTA=1
# NORMALIZE=global
NORMALIZE=speaker
# NORMALIZE=utterance


if [ $stage -le 0 ]; then
  echo ============================================================================
  echo "                           Data Preparation                               "
  echo ============================================================================

  # Data preparation based on kaldi-asr
  local/timit_data_prep.sh $timit || exit 1;
  local/timit_prepare_dict.sh || exit 1;
  local/timit_format_data.sh || exit 1;

  echo "Finish data preparation (stage: 0)."
fi


if [ $stage -le 1 ]; then
  echo ============================================================================
  echo "                        Feature extranction                               "
  echo ============================================================================

  if [ $TOOL = 'kaldi' ]; then
    for x in train dev test; do
      steps/make_fbank.sh --nj 8 --cmd run.pl $DATA_SAVEPATH/$x $DATA_SAVEPATH/make_fbank/$x $DATA_SAVEPATH/fbank || exit 1;
      steps/compute_cmvn_stats.sh $DATA_SAVEPATH/$x $DATA_SAVEPATH/make_fbank/$x $DATA_SAVEPATH/fbank || exit 1;
      utils/fix_data_dir.sh $DATA_SAVEPATH/$x || exit 1;
    done

  elif [ $TOOL = 'htk' ]; then
    # Make a config file to covert from wav to htk file
    python local/make_htk_config.py \
        --data_save_path $DATA_SAVEPATH \
        --config_save_path ./conf/fbank_htk.conf \
        --audio_file_type nist \
        --channels $CHANNELS \
        --sampling_rate 16000 \
        --window $WINDOW \
        --slide $SLIDE \
        --energy $ENERGY \
        --delta $DELTA \
        --deltadelta $DELTADELTA || exit 1;

    for data_type in train dev test ; do
      if [ ! -e $DATA_SAVEPATH/htk/$data_type/.done_make_htk ]; then
        mkdir -p $DATA_SAVEPATH/htk/$data_type
        touch $DATA_SAVEPATH/$data_type/htk.scp
        cat $DATA_SAVEPATH/$data_type/wav.scp | while read line
        do
          # Convert from wav to htk files
          wav_path=`echo $line | awk -F " " '{ print $(NF - 1) }'`
          speaker=`echo $line | awk -F "/" '{ print $(NF - 1) }'`
          mkdir -p $DATA_SAVEPATH/htk/$data_type/$speaker
          file_name=`basename $wav_path`
          base=${file_name%.*}
          # ext=${file_name##*.}
          htk_path=$DATA_SAVEPATH/htk/$data_type/$speaker/$base".htk"
          echo $wav_path $htk_path > ./tmp.scp
          $HCOPY -T 1 -C ./conf/fbank_htk.conf -S ./tmp.scp || exit 1;
          echo $htk_path >> $DATA_SAVEPATH/$data_type/htk.scp
        done
        rm ./tmp.scp
        touch $DATA_SAVEPATH/htk/$data_type/.done_make_htk
      fi
    done

  else
    if ! which sox >&/dev/null; then
      echo "This script requires you to first install sox";
      exit 1;
    fi
  fi

  if [ ! -e $DATA_SAVEPATH/feature/$TOOL/.done_feature_extraction ]; then
    python local/feature_extraction.py \
      --data_save_path $DATA_SAVEPATH \
      --tool $TOOL \
      --normalize $NORMALIZE \
      --channels $CHANNELS \
      --window $WINDOW \
      --slide $SLIDE \
      --energy $ENERGY \
      --delta $DELTA \
      --deltadelta $DELTADELTA || exit 1;
    touch $DATA_SAVEPATH/feature/$TOOL/.done_feature_extraction
  fi

  echo "Finish feature extranction (stage: 1)."
fi


if [ $stage -le 2 ]; then
  echo ============================================================================
  echo "                            Create dataset                                "
  echo ============================================================================

  if [ ! -e $DATA_SAVEPATH/dataset/$TOOL/.done_dataset ]; then
    python local/make_dataset_csv.py \
      --data_save_path $DATA_SAVEPATH \
      --phone_map_file_path ./conf/phones.60-48-39.map \
      --tool $TOOL || exit 1;
    touch $DATA_SAVEPATH/dataset/$TOOL/.done_dataset
  fi

  echo "Finish creating dataset (stage: 2)."
fi


if [ $stage -le 3 ]; then
  echo ============================================================================
  echo "                             Training stage                               "
  echo ============================================================================

  config_path=$1
  gpu_index=$2
  filename=$(basename $config_path | awk -F. '{print $1}')

  mkdir -p log
  mkdir -p $MODEL_SAVEPATH

  echo "Start training..."

  if $restart; then
    if $run_background; then
      CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
      nohup $PYTHON exp/training/train.py \
        --gpu $gpu_index \
        --saved_model_path $config_path \
        --data_save_path $DATA_SAVEPATH > log/$filename".log" &
    else
      CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
      $PYTHON exp/training/train.py \
        --gpu $gpu_index \
        --saved_model_path $config_path \
        --data_save_path $DATA_SAVEPATH || exit 1;
    fi
  else
    if $run_background; then
      CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
      nohup $PYTHON exp/training/train.py \
        --gpu $gpu_index \
        --config_path $config_path \
        --model_save_path $MODEL_SAVEPATH \
        --data_save_path $DATA_SAVEPATH > log/$filename".log" &
    else
      CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
      $PYTHON exp/training/train.py \
        --gpu $gpu_index \
        --config_path $config_path \
        --model_save_path $MODEL_SAVEPATH \
        --data_save_path $DATA_SAVEPATH || exit 1;
    fi
  fi

  echo "Finish model training (stage: 3)."
fi


echo "Done."
