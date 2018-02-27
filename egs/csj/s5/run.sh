#!/bin/bash

# Copyright  2015 Tokyo Institute of Technology
#                 (Authors: Takafumi Moriya, Tomohiro Tanaka and Takahiro Shinozaki)
#            2015 Mitsubishi Electric Research Laboratories (Author: Shinji Watanabe)
# Apache 2.0
# Acknowledgement  This work was supported by JSPS KAKENHI Grant Number 26280055.

# This recipe is based on the Switchboard corpus recipe, by Arnab Ghoshal,
# in the egs/swbd/s5c/ directory.

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.
# Caution: some of the graph creation steps use quite a bit of memory, so you
# should run this on a machine that has sufficient memory.

. ./cmd.sh
. ./path.sh
set -e # exit on error

if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./run.sh path_to_config_file gpu_index" 1>&2
  exit 1
fi


echo ============================================================================
echo "                                   CSJ                                     "
echo ============================================================================

stage=2

### Set path to original data
CSJDATATOP="/n/rd25/mimura/corpus/CSJ"
#CSJDATATOP=/db/laputa1/$CSJDATA_SAVEPATH/processed/public/CSJ ## CSJ database top directory.
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
DATASIZE=subset
# DATASIZE=aps
# DATASIZE=fullset
# DATASIZE=all

### Set path to save dataset
CSJDATA_SAVEPATH="/n/sd8/inaguma/corpus/csj/kaldi"

### Select one tool to extract features (HTK is the fastest)
TOOL='htk'
# TOOL='python_speech_features'
# TOOL='librosa'

# NOTE: set when using HTK toolkit
HCOPY_PATH='/home/lab5/inaguma/htk-3.4/bin/HCopy'

### Configuration of feature extranction
FEATURE_TYPE='fbank'
# FEATURE_TYPE='mfcc'
# FEATURE_TYPE='wav'
CHANNELS=80
WINDOW=0.025
SLIDE=0.01
ENERGY=0
DELTA=1
DELTADELTA=1
# NORMALIZE='global'
NORMALIZE='speaker'
# NORMALIZE='utterance'
# NORMALIZE='no'
# NOTE: normalize in [-1, 1] in case of wav


# Set path to CUDA
export PATH=$PATH:/usr/local/cuda-8.0/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64

# Set path to python
# PYTHON=/home/lab5/inaguma/.pyenv/versions/anaconda3-4.1.1/bin/python
PYTHON=/home/lab5/inaguma/.pyenv/versions/anaconda3-4.1.1/envs/`hostname`/bin/python

# Set path to save the model
MODEL_SAVE_PATH="/n/sd8/inaguma/result"



echo ============================================================================
echo "                           Data Preparation                               "
echo ============================================================================
export CSJDATA_SAVEPATH=$CSJDATA_SAVEPATH/$DATASIZE
if [ $stage -le 0 ]; then
  rm -rf $CSJDATA_SAVEPATH/local
  if [ ! -e $CSJDATA_SAVEPATH/csj-data/.done_make_all ]; then
   echo "CSJ transcription file does not exist"
   #local/csj_make_trans/csj_autorun.sh <RESOUCE_DIR> <MAKING_PLACE(no change)> || exit 1;
   local/csj_make_trans/csj_autorun.sh $CSJDATATOP $CSJDATA_SAVEPATH/csj-data $CSJVER
  fi
  wait

  [ ! -e $CSJDATA_SAVEPATH/csj-data/.done_make_all ]\
      && echo "Not finished processing CSJ data" && exit 1;

  # Prepare Corpus of Spontaneous Japanese (CSJ) data.
  # Processing CSJ data to KALDI format based on switchboard recipe.
  # local/csj_data_prep.sh <SPEECH_and_TRANSCRIPTION_DATA_DIRECTORY> [ <mode_number> ]
  # mode_number can be 0, 1, 2, 3 (0=default using "Academic lecture" and "other" data,
  #                                1=using "Academic lecture" data,
  #                                2=using All data except for "dialog" data, 3=using All data )
  if [ $DATASIZE = 'subset' ]; then
    local/csj_data_prep.sh $CSJDATA_SAVEPATH/csj-data  # subset (240h)
  elif [ $DATASIZE = 'aps' ]; then
    local/csj_data_prep.sh $CSJDATA_SAVEPATH/csj-data 1  # aps
  elif [ $DATASIZE = 'fullset' ]; then
    local/csj_data_prep.sh $CSJDATA_SAVEPATH/csj-data 2  # fullset (586h)
  elif [ $DATASIZE = 'all' ]; then
    local/csj_data_prep.sh $CSJDATA_SAVEPATH/csj-data 3  # all
  fi

  local/csj_prepare_dict.sh

  # Data preparation and formatting for evaluation set.
  # CSJ has 3 types of evaluation data
  #local/csj_eval_data_prep.sh <SPEECH_and_TRANSCRIPTION_DATA_DIRECTORY_ABOUT_EVALUATION_DATA> <EVAL_NUM>
  for eval_num in eval1 eval2 eval3 ; do
      local/csj_eval_data_prep.sh $CSJDATA_SAVEPATH/csj-data/eval $eval_num
  done

  # Use the first 4k sentences from training data as dev set. (39 speakers.)
  # NOTE: when we trained the LM, we used the 1st 10k sentences as dev set,
  # so the 1st 4k won't have been used in the LM training data.
  # However, they will be in the lexicon, plus speakers may overlap,
  # so it's still not quite equivalent to a test set.
  utils/subset_data_dir.sh --first $CSJDATA_SAVEPATH/train 4000 $CSJDATA_SAVEPATH/dev # 6hr 31min
  n=$[`cat $CSJDATA_SAVEPATH/train/segments | wc -l` - 4000]
  utils/subset_data_dir.sh --last $CSJDATA_SAVEPATH/train $n $CSJDATA_SAVEPATH/train_nodev

  # Now-- there are 162k utterances (240hr 8min), and we want to start the
  # monophone training on relatively short utterances (easier to align), but want
  # to exclude the shortest ones.
  # Therefore, we first take the 100k shortest ones;
  # remove most of the repeated utterances, and
  # then take 10k random utterances from those (about 8hr 9mins)
  # utils/subset_data_dir.sh --shortest $CSJDATA_SAVEPATH/train_nodev 100000 $CSJDATA_SAVEPATH/train_100kshort
  # utils/subset_data_dir.sh $CSJDATA_SAVEPATH/train_100kshort 30000 $CSJDATA_SAVEPATH/train_30kshort

  # Take the first 100k utterances (about half the data); we'll use
  # this for later stages of training.
  utils/subset_data_dir.sh --first $CSJDATA_SAVEPATH/train_nodev 100000 $CSJDATA_SAVEPATH/train_100k
  utils/data/remove_dup_utts.sh 200 $CSJDATA_SAVEPATH/train_100k $CSJDATA_SAVEPATH/train_100k_tmp  # 147hr 6min
  rm -rf $CSJDATA_SAVEPATH/train_100k
  mv $CSJDATA_SAVEPATH/train_100k_tmp $CSJDATA_SAVEPATH/train_100k

  # Finally, the full training set:
  rm -rf $CSJDATA_SAVEPATH/train
  utils/data/remove_dup_utts.sh 300 $CSJDATA_SAVEPATH/train_nodev $CSJDATA_SAVEPATH/train  # 233hr 36min
  rm -rf $CSJDATA_SAVEPATH/train_nodev

  echo "Finish data preparation (stage: 0)."
fi

# Calculate the amount of utterance segmentations.
# perl -ne 'split; $s+=($_[3]-$_[2]); END{$h=int($s/3600); $r=($s-$h*3600); $m=int($r/60); $r-=$m*60; printf "%.1f sec -- %d:%d:%.1f\n", $s, $h, $m, $r;}' $CSJDATA_SAVEPATH/train/segments
# perl -ne 'split; $s+=($_[3]-$_[2]); END{$h=int($s/3600); $r=($s-$h*3600); $m=int($r/60); $r-=$m*60; printf "%.1f sec -- %d:%d:%.1f\n", $s, $h, $m, $r;}' $CSJDATA_SAVEPATH/eval1/segments
# perl -ne 'split; $s+=($_[3]-$_[2]); END{$h=int($s/3600); $r=($s-$h*3600); $m=int($r/60); $r-=$m*60; printf "%.1f sec -- %d:%d:%.1f\n", $s, $h, $m, $r;}' $CSJDATA_SAVEPATH/eval2/segments
# perl -ne 'split; $s+=($_[3]-$_[2]); END{$h=int($s/3600); $r=($s-$h*3600); $m=int($r/60); $r-=$m*60; printf "%.1f sec -- %d:%d:%.1f\n", $s, $h, $m, $r;}' $CSJDATA_SAVEPATH/eval3/segments



echo ============================================================================
echo "                        Feature extranction                               "
echo ============================================================================
if [ $stage -le 1 ]; then
  if [ $TOOL = 'htk' ]; then
    # Make a config file to covert from wav to htk file
    python local/make_htk_config.py \
        --data_save_path $CSJDATA_SAVEPATH \
        --config_save_path ./conf \
        --feature_type $FEATURE_TYPE \
        --channels $CHANNELS \
        --window $WINDOW \
        --slide $SLIDE \
        --energy $ENERGY \
        --delta $DELTA \
        --deltadelta $DELTADELTA

    declare -A file_number
    if [ $DATASIZE = 'subset' ]; then
      # file_number["train"]=986
      file_number["train"]=947
    elif [ $DATASIZE = 'aps' ]; then
      file_number["train"]=967
    elif [ $DATASIZE = 'fullset' ]; then
      file_number["train"]=3212
    elif [ $DATASIZE = 'all' ]; then
      exit 1
      file_number["train"]=3212
    fi
    file_number["dev"]=39
    file_number["eval1"]=10
    file_number["eval2"]=10
    file_number["eval3"]=10

    # Convert from wav to htk files
    for data_type in train dev eval1 eval2 eval3 ; do
      mkdir -p $CSJDATA_SAVEPATH/$data_type/htk

      htk_paths=$(find $CSJDATA_SAVEPATH/$data_type/htk -iname '*.htk')
      htk_file_num=$(find $CSJDATA_SAVEPATH/$data_type/htk -iname '*.htk' | wc -l)

      if [ $htk_file_num -ne ${file_number[$data_type]} ]; then
        $HCOPY_PATH -T 1 -C ./conf/$FEATURE_TYPE.conf -S $CSJDATA_SAVEPATH/$data_type/wav2htk.scp
      fi
    done
  else
    if ! which sox >&/dev/null; then
      echo "This script requires you to first install sox";
      exit 1;
    fi
  fi

  python local/feature_extraction.py \
    --data_save_path $CSJDATA_SAVEPATH \
    --tool $TOOL \
    --normalize $NORMALIZE \
    --feature_type $FEATURE_TYPE \
    --channels $CHANNELS \
    --window $WINDOW \
    --slide $SLIDE \
    --energy $ENERGY \
    --delta $DELTA \
    --deltadelta $DELTADELTA

  echo "Finish feature extranction (stage: 1)."
fi

# TODO: create dastaset.csv



echo ============================================================================
echo "                            Create dataset                                "
echo ============================================================================
if [ $stage -le 2 ]; then

  echo "Finish creating dataset (stage: 2)."
fi


echo OK.
exit 1



echo ============================================================================
echo "                             Training stage                               "
echo ============================================================================
if [ $stage -le 3 ]; then
  config_path=$1
  gpu_index=$2
  filename=$(basename $config_path | awk -F. '{print $1}')


  mkdir -p log

  # CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
  # nohup $PYTHON train.py \
  #   --gpu $gpu_index \
  #   --config_path $config_path \
  #   --model_save_path $MODEL_SAVE_PATH > log/$filename".log" &

  CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
  $PYTHON train.py \
    --gpu $gpu_index \
    --config_path $config_path \
    --model_save_path $MODEL_SAVE_PATH

  echo "Finish model training (stage: 3)."
fi



# echo ============================================================================
# echo "                             LM training                                 "
# echo ============================================================================
# if [ $stage -le 4 ]; then
#
#   echo "Finish LM training (stage: 4)."
# fi



# echo ============================================================================
# echo "                              Rescoring                                   "
# echo ============================================================================
# if [ $stage -le 5 ]; then
#
#   echo "Finish rescoring (stage: 5)."
# fi


echo "Done."


# utils/prepare_lang.sh --num-sil-states 4 $CSJDATA_SAVEPATH/local/dict_nosp "<unk>" $CSJDATA_SAVEPATH/local/lang_nosp $CSJDATA_SAVEPATH/lang_nosp

# Now train the language models.
# local/csj_train_lms.sh $CSJDATA_SAVEPATH/local/train/text $CSJDATA_SAVEPATH/local/dict_nosp/lexicon.txt $CSJDATA_SAVEPATH/local/lm

# We don't really need all these options for SRILM, since the LM training script
# does some of the same processing (e.g. -subset -tolower)
# srilm_opts="-subset -prune-lowprobs -unk -tolower -order 3"
# LM=$CSJDATA_SAVEPATH/local/lm/csj.o3g.kn.gz
# utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
#   $CSJDATA_SAVEPATH/lang_nosp $LM $CSJDATA_SAVEPATH/local/dict_nosp/lexicon.txt $CSJDATA_SAVEPATH/lang_nosp_csj_tg


# getting results (see RESULTS file)
# for eval_num in eval1 eval2 eval3 $dev_set ; do
#     echo "=== evaluation set $eval_num ===" ;
#     for x in exp/{tri,dnn}*/decode_${eval_num}*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done ;
# done
