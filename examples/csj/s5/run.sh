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

stage=3
# hierarchical_model=false
hierarchical_model=true

### Set true when hiding progress bar
run_background=true

### Set true when restarting training
# restart=false
restart=true

### Set path to original data
CSJDATATOP="/n/rd25/mimura/corpus/CSJ"
#CSJDATATOP=/db/laputa1/$DATA_SAVEPATH/processed/public/CSJ ## CSJ database top directory.
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

### Set path to save the model
MODEL_SAVEPATH="/n/sd8/inaguma/result/csj"

### Set path to save dataset
DATA_SAVEPATH="/n/sd8/inaguma/corpus/csj/kaldi"

### Select one tool to extract features (HTK is the fastest)
# TOOL=kaldi
TOOL=htk
# TOOL=python_speech_features
# TOOL=librosa
# # TOOL=wav

### Configuration of feature extranction
CHANNELS=80
WINDOW=0.025
SLIDE=0.01
ENERGY=0
DELTA=1
DELTADELTA=1
# NORMALIZE=global
NORMALIZE=speaker
# NORMALIZE=utterance
# NORMALIZE=no
# NOTE: normalize in [-1, 1] in case of WAV


export DATA_SAVEPATH=$DATA_SAVEPATH/$DATASIZE
export DATA_DOWNLOADPATH=$DATA_SAVEPATH/data
if [ $stage -le 0 ]; then
  echo ============================================================================
  echo "                           Data Preparation                               "
  echo ============================================================================

  if [ ! -e $DATA_SAVEPATH/csj-data/.done_make_all ]; then
   echo "CSJ transcription file does not exist"
   #local/csj_make_trans/csj_autorun.sh <RESOUCE_DIR> <MAKING_PLACE(no change)> || exit 1;
   local/csj_make_trans/csj_autorun.sh $CSJDATATOP $DATA_SAVEPATH/csj-data $CSJVER || exit 1;
  fi
  wait

  [ ! -e $DATA_SAVEPATH/csj-data/.done_make_all ]\
      && echo "Not finished processing CSJ data" && exit 1;

  rm -rf $DATA_SAVEPATH/local

  # Prepare Corpus of Spontaneous Japanese (CSJ) data.
  # Processing CSJ data to KALDI format based on switchboard recipe.
  # local/csj_data_prep.sh <SPEECH_and_TRANSCRIPTION_DATA_DIRECTORY> [ <mode_number> ]
  # mode_number can be 0, 1, 2, 3 (0=default using "Academic lecture" and "other" data,
  #                                1=using "Academic lecture" data,
  #                                2=using All data except for "dialog" data, 3=using All data )
  if [ $DATASIZE = "subset" ]; then
    local/csj_data_prep.sh $DATA_SAVEPATH/csj-data || exit 1;  # subset (240h)
  elif [ $DATASIZE = "aps" ]; then
    local/csj_data_prep.sh $DATA_SAVEPATH/csj-data 1 || exit 1;  # aps
  elif [ $DATASIZE = "fullset" ]; then
    local/csj_data_prep.sh $DATA_SAVEPATH/csj-data 2 || exit 1;  # fullset (586h)
  elif [ $DATASIZE = "all" ]; then
    local/csj_data_prep.sh $DATA_SAVEPATH/csj-data 3 || exit 1;  # all
  fi

  local/csj_prepare_dict.sh || exit 1;

  # Use the first 4k sentences from training data as dev set. (39 speakers.)
  # NOTE: when we trained the LM, we used the 1st 10k sentences as dev set,
  # so the 1st 4k won't have been used in the LM training data.
  # However, they will be in the lexicon, plus speakers may overlap,
  # so it's still not quite equivalent to a test set.
  utils/subset_data_dir.sh --first $DATA_SAVEPATH/train 4000 $DATA_SAVEPATH/dev || exit 1; # 6hr 31min
  n=$[`cat $DATA_SAVEPATH/train/segments | wc -l` - 4000]
  utils/subset_data_dir.sh --last $DATA_SAVEPATH/train $n $DATA_SAVEPATH/train_nodev || exit 1;

  # Take the first 100k utterances (about half the data); we'll use
  # this for later stages of training.
  # utils/subset_data_dir.sh --first $DATA_SAVEPATH/train_nodev 100000 $DATA_SAVEPATH/train_100k || exit 1;
  # utils/data/remove_dup_utts.sh 200 $DATA_SAVEPATH/train_100k $DATA_SAVEPATH/train_100k_tmp || exit 1;  # 147hr 6min
  # rm -rf $DATA_SAVEPATH/train_100k
  # mv $DATA_SAVEPATH/train_100k_tmp $DATA_SAVEPATH/train_100k || exit 1;

  # Finally, the full training set:
  rm -rf $DATA_SAVEPATH/train
  utils/data/remove_dup_utts.sh 300 $DATA_SAVEPATH/train_nodev $DATA_SAVEPATH/train || exit 1;  # 233hr 36min
  rm -rf $DATA_SAVEPATH/train_nodev

  # Data preparation and formatting for evaluation set.
  # CSJ has 3 types of evaluation data
  #local/csj_eval_data_prep.sh <SPEECH_and_TRANSCRIPTION_DATA_DIRECTORY_ABOUT_EVALUATION_DATA> <EVAL_NUM>
  for eval_num in eval1 eval2 eval3 ; do
      local/csj_eval_data_prep.sh $DATA_SAVEPATH/csj-data/eval $eval_num || exit 1;
  done

  echo "Finish data preparation (stage: 0)."
fi

# Calculate the amount of utterance segmentations.
# perl -ne 'split; $s+=($_[3]-$_[2]); END{$h=int($s/3600); $r=($s-$h*3600); $m=int($r/60); $r-=$m*60; printf "%.1f sec -- %d:%d:%.1f\n", $s, $h, $m, $r;}' $DATA_SAVEPATH/train/segments
# perl -ne 'split; $s+=($_[3]-$_[2]); END{$h=int($s/3600); $r=($s-$h*3600); $m=int($r/60); $r-=$m*60; printf "%.1f sec -- %d:%d:%.1f\n", $s, $h, $m, $r;}' $DATA_SAVEPATH/dev/segments
# perl -ne 'split; $s+=($_[3]-$_[2]); END{$h=int($s/3600); $r=($s-$h*3600); $m=int($r/60); $r-=$m*60; printf "%.1f sec -- %d:%d:%.1f\n", $s, $h, $m, $r;}' $DATA_SAVEPATH/eval1/segments
# perl -ne 'split; $s+=($_[3]-$_[2]); END{$h=int($s/3600); $r=($s-$h*3600); $m=int($r/60); $r-=$m*60; printf "%.1f sec -- %d:%d:%.1f\n", $s, $h, $m, $r;}' $DATA_SAVEPATH/eval2/segments
# perl -ne 'split; $s+=($_[3]-$_[2]); END{$h=int($s/3600); $r=($s-$h*3600); $m=int($r/60); $r-=$m*60; printf "%.1f sec -- %d:%d:%.1f\n", $s, $h, $m, $r;}' $DATA_SAVEPATH/eval3/segments


if [ $stage -le 1 ]; then
  echo ============================================================================
  echo "                        Feature extranction                               "
  echo ============================================================================

  if [ $TOOL = "kaldi" ]; then
    for x in train dev eval1 eval2 eval3; do
      steps/make_fbank.sh --nj 8 --cmd run.pl $DATA_SAVEPATH/$x $DATA_SAVEPATH/make_fbank/$x $DATA_SAVEPATH/fbank || exit 1;
      steps/compute_cmvn_stats.sh $DATA_SAVEPATH/$x $DATA_SAVEPATH/make_fbank/$x $DATA_SAVEPATH/fbank || exit 1;
      utils/fix_data_dir.sh $DATA_SAVEPATH/$x || exit 1;
    done

  elif [ $TOOL = "htk" ]; then
    # Make a config file to covert from wav to htk file
    python local/make_htk_config.py \
        --data_save_path $DATA_SAVEPATH \
        --config_save_path ./conf/fbank_htk.conf \
        --channels $CHANNELS \
        --window $WINDOW \
        --slide $SLIDE \
        --energy $ENERGY \
        --delta $DELTA \
        --deltadelta $DELTADELTA || exit 1;

    # Convert from wav to htk files
    for data_type in train dev eval1 eval2 eval3; do
      mkdir -p $DATA_SAVEPATH/htk
      mkdir -p $DATA_SAVEPATH/htk/$data_type

      if [ ! -e $DATA_SAVEPATH/htk/$data_type/.done_make_htk ]; then
        $HCOPY -T 1 -C ./conf/fbank_htk.conf -S $DATA_SAVEPATH/$data_type/wav2htk.scp || exit 1;
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

  if $hierarchical_model; then
    if $restart; then
      if $run_background; then
        CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
        nohup $PYTHON exp/training/train_hierarchical.py \
          --gpu $gpu_index \
          --saved_model_path $config_path \
          --data_save_path $DATA_SAVEPATH > log/$filename".log" &
      else
        CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
        nohup $PYTHON exp/training/train_hierarchical.py \
          --gpu $gpu_index \
          --saved_model_path $config_path \
          --data_save_path $DATA_SAVEPATH || exit 1;
      fi
    else
      if $run_background; then
        CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
        nohup $PYTHON exp/training/train_hierarchical.py \
          --gpu $gpu_index \
          --config_path $config_path \
          --model_save_path $MODEL_SAVEPATH \
          --data_save_path $DATA_SAVEPATH > log/$filename".log" &
      else
        CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
        $PYTHON exp/training/train_hierarchical.py \
          --gpu $gpu_index \
          --config_path $config_path \
          --model_save_path $MODEL_SAVEPATH \
          --data_save_path $DATA_SAVEPATH || exit 1;
      fi
    fi
  else
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
          --data_save_path $DATA_SAVEPATH　|| exit 1;
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


# utils/prepare_lang.sh --num-sil-states 4 $DATA_SAVEPATH/local/dict_nosp "<unk>" $DATA_SAVEPATH/local/lang_nosp $DATA_SAVEPATH/lang_nosp

# Now train the language models.
# local/csj_train_lms.sh $DATA_SAVEPATH/local/train/text $DATA_SAVEPATH/local/dict_nosp/lexicon.txt $DATA_SAVEPATH/local/lm

# We don't really need all these options for SRILM, since the LM training script
# does some of the same processing (e.g. -subset -tolower)
# srilm_opts="-subset -prune-lowprobs -unk -tolower -order 3"
# LM=$DATA_SAVEPATH/local/lm/csj.o3g.kn.gz
# utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
#   $DATA_SAVEPATH/lang_nosp $LM $DATA_SAVEPATH/local/dict_nosp/lexicon.txt $DATA_SAVEPATH/lang_nosp_csj_tg


# getting results (see RESULTS file)
# for eval_num in eval1 eval2 eval3 $dev_set ; do
#     echo "=== evaluation set $eval_num ===" ;
#     for x in exp/{tri,dnn}*/decode_${eval_num}*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done ;
# done
