#!/bin/bash

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh
set -e

. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

if [ $# -lt 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./run.sh path_to_config_file (or path_to_saved_model) gpu_id1 gpu_id2... (arbitrary number)" 1>&2
  exit 1
fi

ngpus=`expr $# - 1`
config_path=$1
gpu_ids=$2

if [ $# -gt 2 ]; then
  rest_ngpus=`expr $ngpus - 1`
  for i in `seq 1 $rest_ngpus`
  do
    gpu_ids=$gpu_ids","${3}
    shift
  done
fi



echo ============================================================================
echo "                                  Babel                                    "
echo ============================================================================

stage=0
run_background=true
# run_background=false

### Set path to save the model
model=/n/sd8/inaguma/result/babel

### Set language IDs
# train_langs="101 103 104 105 107 201 204 205 207 404"  # 10
train_langs="101 102 103 104 105 106 107 201 202 203 204 205 206 207 404"  # 15
eval_langs="102 106 202 203 206"

train_set="train"
dev_set="dev"
for l in ${train_langs}; do
  train_set="${train_set}_${l}"
  dev_set="${dev_set}_${l}"
done

eval_set=""
for l in ${eval_langs}; do
  eval_set="eval_${l} ${eval_set}"
done
eval_set=${eval_set%% }

### Select one tool to extract features (HTK is the fastest)
# tool=kaldi
tool=htk
# tool=python_speech_features
# tool=librosa

### Configuration of feature extranction
channels=80
window=0.025
slide=0.01
energy=1
delta=1
deltadelta=1
# normalize=global
normalize=speaker
# normalize=utterance


if [ ! -e ${KALDI_ROOT}/tools/sph2pipe_v2.5/sph2pipe ]; then
  echo ============================================================================
  echo "                           Install sph2pipe                               "
  echo ============================================================================
  cur_dir=`pwd`
  # Install instructions for sph2pipe_v2.5.tar.gz
  if ! which wget >&/dev/null; then
    echo "This script requires you to first install wget";
    exit 1;
  fi
  if ! which automake >&/dev/null; then
    echo "Warning: automake not installed (IRSTLM installation will not work)"
    sleep 1
  fi
  if ! which libtoolize >&/dev/null && ! which glibtoolize >&/dev/null; then
    echo "Warning: libtoolize or glibtoolize not installed (IRSTLM installation probably will not work)"
    sleep 1
  fi

  if [ ! -e ${KALDI_ROOT}/tools/sph2pipe_v2.5.tar.gz ]; then
    wget -T 3 -t 3 http://www.openslr.org/resources/3/sph2pipe_v2.5.tar.gz -P ${KALDI_ROOT}/tools
  else
    echo "sph2pipe_v2.5.tar.gz is already downloaded."
  fi
  tar -xovzf ${KALDI_ROOT}/tools/sph2pipe_v2.5.tar.gz -C ${KALDI_ROOT}/tools
  rm ${KALDI_ROOT}/tools/sph2pipe_v2.5.tar.gz
  echo "Enter into ${KALDI_ROOT}/tools/sph2pipe_v2.5 ..."
  cd ${KALDI_ROOT}/tools/sph2pipe_v2.5
  gcc -o sph2pipe *.c -lm
  echo "Get out of ${KALDI_ROOT}/tools/sph2pipe_v2.5 ..."
  cd ${cur_dir}
fi


if [ $stage -le 0 ] && [ ! -e ${data}/.stage_0 ]; then
  echo ============================================================================
  echo "                           Data Preparation                               "
  echo ============================================================================

  ./local/setup_languages.sh --langs "${train_langs}" --recog "${recog}"

  touch ${data}/.stage_0
  echo "Finish data preparation (stage: 0)."
fi


if [ $stage -le 1 ] && [ ! -e ${data}/.stage_1 ]; then
  echo ============================================================================
  echo "                        Feature extranction                               "
  echo ============================================================================

  if [ ${tool} = "kaldi" ]; then
    for x in ${train_set} ${dev_set} ${eval_set}; do
      steps/make_fbank_pitch.sh --nj 8 --cmd run.pl ${data}/$x ${data}/make_fbank/$x ${data}/fbank || exit 1;
      steps/compute_cmvn_stats.sh ${data}/$x ${data}/make_fbank/$x ${data}/fbank || exit 1;
      utils/fix_data_dir.sh ${data}/$x || exit 1;
    done

  elif [ ${tool} = "htk" ]; then
    # Make a config file to covert from wav to htk file
    ${PYTHON} local/make_htk_config.py \
        --data_save_path ${data} \
        --config_save_path ./conf/fbank_htk.conf \
        --audio_file_type wav \
        --channels $channels \
        --sampling_rate 16000 \
        --window $window \
        --slide $slide \
        --energy $energy \
        --delta $delta \
        --deltadelta $deltadelta || exit 1;

    for x in ${train_set} ${dev_set} ${eval_set}; do
      mkdir -p ${data}/wav
      mkdir -p ${data}/htk
      [ -e ${data}/$x/htk.scp ] && rm ${data}/$x/htk.scp
      touch ${data}/$x/htk.scp
      cat ${data}/$x/wav.scp | while read line
      do
        # Convert from sph to wav files
        sph_path=`echo $line | awk -F " " '{ print $(NF - 1) }'`
        if [ ! `echo ${sph_path} | grep 'downsample'` ]; then
          file_name=`basename ${sph_path}`
          base=${file_name%.*}
          # ext=${file_name##*.}
          lang_id=`echo $base | awk -F "_" '{ print $3 }'`
          mkdir -p ${data}/wav/${lang_id}
          wav_path=${data}/wav/${lang_id}/$base".wav"
          echo $wav_path
          if [ ! -e ${wav_path} ]; then
            ${KALDI_ROOT}/tools/sph2pipe_v2.5/sph2pipe -f wav ${sph_path} ${wav_path} || exit 1;
          fi

          # Convert from wav to htk files
          mkdir -p ${data}/htk/${lang_id}
          htk_path=${data}/htk/${lang_id}/$base".htk"
          if [ ! -e ${htk_path} ]; then
            echo ${wav_path}  ${htk_path} > ./tmp.scp
            $HCOPY -T 1 -C ./conf/fbank_htk.conf -S ./tmp.scp || exit 1;
            rm ./tmp.scp
          fi
          echo ${htk_path} >> ${data}/$x/htk.scp
        fi
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
    --normalize $normalize \
    --channels $channels \
    --window $window \
    --slide $slide \
    --energy $energy \
    --delta $delta \
    --deltadelta $deltadelta || exit 1;

  touch ${data}/.stage_1
  echo "Finish feature extranction (stage: 1)."
fi


if [ $stage -le 2 ] && [ ! -e ${data}/.stage_2 ]; then
  echo ============================================================================
  echo "                            Create dataset                                "
  echo ============================================================================

  ${PYTHON} local/make_dataset_csv.py \
    --data_save_path ${data} \
    --tool ${tool} || exit 1;

  touch ${data}/.stage_2
  echo "Finish creating dataset (stage: 2)."
fi


if [ $stage -le 3 ]; then
  echo ============================================================================
  echo "                             Training stage                               "
  echo ============================================================================

  filename=$(basename ${config_path} | awk -F. '{print $1}')
  mkdir -p log
  mkdir -p ${model}

  echo "Start training..."

  if [ `echo ${config_path} | grep 'result'` ]; then
    if $run_background; then
      CUDA_VISIBLE_DEVICES=${gpu_ids} \
      nohup ${PYTHON} ../../../src/bin/training/train.py \
        --corpus ${corpus} \
        --ngpus ${ngpus} \
        --train_set train \
        --dev_set dev \
        --eval_sets eval1 \
        --saved_model_path ${config_path} \
        --data_save_path ${data} > log/$filename".log" &
    else
      CUDA_VISIBLE_DEVICES=${gpu_ids} \
      ${PYTHON} ../../../src/bin/training/train.py \
        --corpus ${corpus} \
        --ngpus ${ngpus} \
        --train_set train \
        --dev_set dev \
        --eval_sets eval1 \
        --saved_model_path ${config_path} \
        --data_save_path ${data} || exit 1;
    fi
  else
    if $run_background; then
      CUDA_VISIBLE_DEVICES=${gpu_ids} \
      nohup ${PYTHON} ../../../src/bin/training/train.py \
        --corpus ${corpus} \
        --ngpus ${ngpus} \
        --train_set train \
        --dev_set dev \
        --eval_sets eval1 \
        --config_path ${config_path} \
        --model_save_path ${model} \
        --data_save_path ${data} > log/$filename".log" &
    else
      CUDA_VISIBLE_DEVICES=${gpu_ids} \
      ${PYTHON} ../../../src/bin/training/train.py \
        --corpus ${corpus} \
        --ngpus ${ngpus} \
        --train_set train \
        --dev_set dev \
        --eval_sets eval1 \
        --config_path ${config_path} \
        --model_save_path ${model} \
        --data_save_path ${data} || exit 1;
    fi
  fi

  echo "Finish model training (stage: 3)."
fi
