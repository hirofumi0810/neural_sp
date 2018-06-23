#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

# . utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./run.sh path_to_config_file gpu_ids or ./run.sh path_to_saved_model gpu_ids" 1>&2
  exit 1
fi


echo ============================================================================
echo "                                   CSJ                                     "
echo ============================================================================

stage=0
run_background=true
# run_background=false

### Set path to save the model
model=/n/sd8/inaguma/result/csj

### Set path to original data
CSJDATATOP="/n/rd25/mimura/corpus/CSJ"
#CSJDATATOP=/db/laputa1/${data}/processed/public/CSJ ## CSJ database top directory.
CSJVER=dvd  ## Set your CSJ format (dvd or usb).
            ## Usage    :
            ## Case DVD : We assume CSJ DVDs are copied in this directory with the names dvd1, dvd2,...,dvd17.
            ##            Neccesary directory is dvd3 - dvd17.
            ##            e.g. $ ls $CSJDATATOP(DVD) => 00README.txt dvd1 dvd2 ... dvd17
            ##
            ## Case USB : Neccesary directory is MORPH/SDB and WAV
            ##            e.g. $ ls $CSJDATATOP(USB) => 00README.txt DOC MORPH ... WAV fileList.csv
            ## Case merl :MERL setup. Neccesary directory is WAV and sdb

### Select data size
export datasize=aps_other
# export datasize=aps
# export datasize=all_except_dialog
# export datasize=all
# NOTE: aps_other=default using "Academic lecture" and "other" data,
#       aps=using "Academic lecture" data,
#       all_except_dialog=using All data except for "dialog" data,
#       all=using All data

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


train=train_${datasize}

if [ $stage -le 0 ] && [ ! -e ${data}/.stage_0_${datasize} ]; then
  echo ============================================================================
  echo "                           Data Preparation                               "
  echo ============================================================================

  local/csj_make_trans/csj_autorun.sh $CSJDATATOP ${data}/csj-data $CSJVER || exit 1;

  # Prepare Corpus of Spontaneous Japanese (CSJ) data.
  # Processing CSJ data to KALDI format based on switchboard recipe.
  local/csj_data_prep.sh ${data}/csj-data ${datasize} || exit 1;

  local/csj_prepare_dict.sh || exit 1;

  # Use the first 4k sentences from training data as dev set. (39 speakers.)
  # NOTE: when we trained the LM, we used the 1st 10k sentences as dev set,
  # so the 1st 4k won't have been used in the LM training data.
  # However, they will be in the lexicon, plus speakers may overlap,
  # so it's still not quite equivalent to a test set.
  utils/subset_data_dir.sh --first ${data}/$train 4000 ${data}/dev || exit 1; # 6hr 31min
  n=$[`cat ${data}/$train/segments | wc -l` - 4000]
  utils/subset_data_dir.sh --last ${data}/$train $n ${data}/tmp || exit 1;

  # Finally, the full training set:
  rm -rf ${data}/$train
  utils/data/remove_dup_utts.sh 300 ${data}/tmp ${data}/$train || exit 1;  # 233hr 36min
  rm -rf ${data}/tmp

  # Data preparation and formatting for evaluation set.
  # CSJ has 3 types of evaluation data
  #local/csj_eval_data_prep.sh <SPEECH_and_TRANSCRIPTION_DATA_DIRECTORY_ABOUT_EVALUATION_DATA> <EVAL_NUM>
  for eval_num in eval1 eval2 eval3 ; do
    local/csj_eval_data_prep.sh ${data}/csj-data/eval $eval_num || exit 1;
  done

  touch ${data}/.stage_0_${datasize}
  echo "Finish data preparation (stage: 0)."
fi


if [ $stage -le 1 ] && [ ! -e ${data}/.stage_1_${datasize} ]; then
  echo ============================================================================
  echo "                        Feature extranction                               "
  echo ============================================================================

  if [ ${tool} = "kaldi" ]; then
    for x in train dev eval1 eval2 eval3; do
      steps/make_fbank.sh --nj 8 --cmd run.pl ${data}/$x ${data}/make_fbank/$x ${data}/fbank || exit 1;
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

    for data_type in $train dev eval1 eval2 eval3; do
      if [ `echo $data_type | grep 'train'` ]; then
        mkdir -p ${data}/htk/train
      else
        mkdir -p ${data}/htk/$data_type
      fi
      [ -e ${data}/$data_type/htk.scp ] && rm ${data}/$data_type/htk.scp
      touch ${data}/$data_type/htk.scp
      cat ${data}/$data_type/wav.scp | while read line
      do
        wav_path=`echo $line | awk -F " " '{ print $(NF - 1) }'`
        file_name=`basename $wav_path`
        base=${file_name%.*}
        # ext=${file_name##*.}

        # Convert from wav to htk files
        if [ `echo $data_type | grep 'train'` ]; then
          htk_path=${data}/htk/train/$base".htk"
        else
          htk_path=${data}/htk/$data_type/$base".htk"
        fi
        if [ ! -e $htk_path ]; then
          echo $wav_path  $htk_path > ./tmp.scp
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
    --data_size ${datasize} \
    --tool ${tool} \
    --normalize $normalize \
    --channels $channels \
    --window $window \
    --slide $slide \
    --energy $energy \
    --delta $delta \
    --deltadelta $deltadelta || exit 1;

  touch ${data}/.stage_1_${datasize}
  echo "Finish feature extranction (stage: 1)."
fi


if [ $stage -le 2 ] && [ ! -e ${data}/.stage_2_${datasize} ]; then
  echo ============================================================================
  echo "                            Create dataset                                "
  echo ============================================================================

  ${PYTHON} local/make_dataset_csv.py \
    --data_save_path ${data} \
    --data_size ${datasize} \
    --tool ${tool} || exit 1;

  touch ${data}/.stage_2_${datasize}
  echo "Finish creating dataset (stage: 2)."
fi


if [ $stage -le 3 ]; then
  echo ============================================================================
  echo "                             Training stage                               "
  echo ============================================================================

  config_path=$1
  gpu_ids=$2
  filename=$(basename ${config_path} | awk -F. '{print $1}')

  mkdir -p log
  mkdir -p ${model}

  echo "Start training..."

  if [ `echo ${config_path} | grep 'hierarchical'` ]; then
    if [ `echo ${config_path} | grep 'result'` ]; then
      if $run_background; then
        CUDA_VISIBLE_DEVICES=${gpu_ids} \
        nohup ${PYTHON} ../../../src/bin/training/train_hierarchical.py \
          --corpus ${corpus} \
          --train_set train \
          --dev_set dev \
          --eval_sets eval1 \
          --gpu_ids ${gpu_ids} \
          --saved_model_path ${config_path} \
          --data_save_path ${data} > log/$filename".log" &
      else
        CUDA_VISIBLE_DEVICES=${gpu_ids} \
        nohup ${PYTHON} ../../../src/bin/training/train_hierarchical.py \
          --corpus ${corpus} \
          --gpu_ids ${gpu_ids} \
          --train_set train \
          --dev_set dev \
          --eval_sets eval1 \
          --saved_model_path ${config_path} \
          --data_save_path ${data} || exit 1;
      fi
    else
      if $run_background; then
        CUDA_VISIBLE_DEVICES=${gpu_ids} \
        nohup ${PYTHON} ../../../src/bin/training/train_hierarchical.py \
          --corpus ${corpus} \
          --gpu_ids ${gpu_ids} \
          --train_set train \
          --dev_set dev \
          --eval_sets eval1 \
          --config_path ${config_path} \
          --model_save_path ${model} \
          --data_save_path ${data} > log/$filename".log" &
      else
        CUDA_VISIBLE_DEVICES=${gpu_ids} \
        ${PYTHON} ../../../src/bin/training/train_hierarchical.py \
          --corpus ${corpus} \
          --gpu_ids ${gpu_ids} \
          --train_set train \
          --dev_set dev \
          --eval_sets eval1 \
          --config_path ${config_path} \
          --model_save_path ${model} \
          --data_save_path ${data} || exit 1;
      fi
    fi
  else
    if [ `echo ${config_path} | grep 'result'` ]; then
      if $run_background; then
        CUDA_VISIBLE_DEVICES=${gpu_ids} \
        nohup ${PYTHON} ../../../src/bin/training/train.py \
          --corpus ${corpus} \
          --gpu_ids ${gpu_ids} \
          --train_set train \
          --dev_set dev \
          --eval_sets eval1 \
          --saved_model_path ${config_path} \
          --data_save_path ${data} > log/$filename".log" &
      else
        CUDA_VISIBLE_DEVICES=${gpu_ids} \
        ${PYTHON} ../../../src/bin/training/train.py \
          --corpus ${corpus} \
          --gpu_ids ${gpu_ids} \
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
          --gpu_ids ${gpu_ids} \
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
          --gpu_ids ${gpu_ids} \
          --train_set train \
          --dev_set dev \
          --eval_sets eval1 \
          --config_path ${config_path} \
          --model_save_path ${model} \
          --data_save_path ${data} || exit 1;
      fi
    fi
  fi

  echo "Finish model training (stage: 3)."
fi


if [ $stage -le 4 ]; then
  echo ============================================================================
  echo "                             LM training                                 "
  echo ============================================================================

  echo "Finish LM training (stage: 4)."
fi


if [ $stage -le 5 ]; then
  echo ============================================================================
  echo "                              Rescoring                                   "
  echo ============================================================================

  echo "Finish rescoring (stage: 5)."
fi


echo "Done."


# utils/prepare_lang.sh --num-sil-states 4 ${data}/local/dict_nosp "<unk>" ${data}/local/lang_nosp ${data}/lang_nosp

# Now train the language models.
# local/csj_train_lms.sh ${data}/local/train/text ${data}/local/dict_nosp/lexicon.txt ${data}/local/lm

# We don't really need all these options for SRILM, since the LM training script
# does some of the same processing (e.g. -subset -tolower)
# srilm_opts="-subset -prune-lowprobs -unk -tolower -order 3"
# LM=${data}/local/lm/csj.o3g.kn.gz
# utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
#   ${data}/lang_nosp $LM ${data}/local/dict_nosp/lexicon.txt ${data}/lang_nosp_csj_tg


# getting results (see RESULTS file)
# for eval_num in eval1 eval2 eval3 $dev_set ; do
#     echo "=== evaluation set $eval_num ===" ;
#     for x in ../../../src/bin/{tri,dnn}*/decode_${eval_num}*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done ;
# done
