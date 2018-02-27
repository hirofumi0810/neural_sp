#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

# . utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

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
export CSJDATA_SAVEPATH="/n/sd8/inaguma/corpus/csj/kaldi"/$DATASIZE

### Set path to save the model
MODEL_SAVEPATH="/n/sd8/inaguma/result"
# MODEL_SAVEPATH="/n/sd8/inaguma/result/csj"

### Select one tool to extract features (HTK is the fastest)
TOOL = 'kaldi'
# TOOL='htk'
# TOOL='python_speech_features'
# TOOL='librosa'
# # TOOL='wav'

### Configuration of feature extranction
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
# NOTE: normalize in [-1, 1] in case of WAV


echo ============================================================================
echo "                           Data Preparation                               "
echo ============================================================================
if [ $stage -le 0 ]; then
  if [ ! -e $CSJDATA_SAVEPATH/csj-data/.done_make_all ]; then
   echo "CSJ transcription file does not exist"
   #local/csj_make_trans/csj_autorun.sh <RESOUCE_DIR> <MAKING_PLACE(no change)> || exit 1;
   local/csj_make_trans/csj_autorun.sh $CSJDATATOP $CSJDATA_SAVEPATH/csj-data $CSJVER || exit 1;
  fi
  wait

  [ ! -e $CSJDATA_SAVEPATH/csj-data/.done_make_all ]\
      && echo "Not finished processing CSJ data" && exit 1;

  rm -rf $CSJDATA_SAVEPATH/local
  
  # Prepare Corpus of Spontaneous Japanese (CSJ) data.
  # Processing CSJ data to KALDI format based on switchboard recipe.
  # local/csj_data_prep.sh <SPEECH_and_TRANSCRIPTION_DATA_DIRECTORY> [ <mode_number> ]
  # mode_number can be 0, 1, 2, 3 (0=default using "Academic lecture" and "other" data,
  #                                1=using "Academic lecture" data,
  #                                2=using All data except for "dialog" data, 3=using All data )
  if [ $DATASIZE = 'subset' ]; then
    local/csj_data_prep.sh $CSJDATA_SAVEPATH/csj-data || exit 1;  # subset (240h)
  elif [ $DATASIZE = 'aps' ]; then
    local/csj_data_prep.sh $CSJDATA_SAVEPATH/csj-data 1 || exit 1;  # aps
  elif [ $DATASIZE = 'fullset' ]; then
    local/csj_data_prep.sh $CSJDATA_SAVEPATH/csj-data 2 || exit 1;  # fullset (586h)
  elif [ $DATASIZE = 'all' ]; then
    local/csj_data_prep.sh $CSJDATA_SAVEPATH/csj-data 3 || exit 1;  # all
  fi

  local/csj_prepare_dict.sh || exit 1;

  # Use the first 4k sentences from training data as dev set. (39 speakers.)
  # NOTE: when we trained the LM, we used the 1st 10k sentences as dev set,
  # so the 1st 4k won't have been used in the LM training data.
  # However, they will be in the lexicon, plus speakers may overlap,
  # so it's still not quite equivalent to a test set.
  utils/subset_data_dir.sh --first $CSJDATA_SAVEPATH/train 4000 $CSJDATA_SAVEPATH/dev || exit 1; # 6hr 31min
  n=$[`cat $CSJDATA_SAVEPATH/train/segments | wc -l` - 4000]
  utils/subset_data_dir.sh --last $CSJDATA_SAVEPATH/train $n $CSJDATA_SAVEPATH/train_nodev || exit 1;

  # Take the first 100k utterances (about half the data); we'll use
  # this for later stages of training.
  # utils/subset_data_dir.sh --first $CSJDATA_SAVEPATH/train_nodev 100000 $CSJDATA_SAVEPATH/train_100k || exit 1;
  # utils/data/remove_dup_utts.sh 200 $CSJDATA_SAVEPATH/train_100k $CSJDATA_SAVEPATH/train_100k_tmp || exit 1;  # 147hr 6min
  # rm -rf $CSJDATA_SAVEPATH/train_100k
  # mv $CSJDATA_SAVEPATH/train_100k_tmp $CSJDATA_SAVEPATH/train_100k || exit 1;

  # Finally, the full training set:
  rm -rf $CSJDATA_SAVEPATH/train
  utils/data/remove_dup_utts.sh 300 $CSJDATA_SAVEPATH/train_nodev $CSJDATA_SAVEPATH/train || exit 1;  # 233hr 36min
  rm -rf $CSJDATA_SAVEPATH/train_nodev

  # Data preparation and formatting for evaluation set.
  # CSJ has 3 types of evaluation data
  #local/csj_eval_data_prep.sh <SPEECH_and_TRANSCRIPTION_DATA_DIRECTORY_ABOUT_EVALUATION_DATA> <EVAL_NUM>
  for eval_num in eval1 eval2 eval3 ; do
      local/csj_eval_data_prep.sh $CSJDATA_SAVEPATH/csj-data/eval $eval_num || exit 1;
  done

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
  if [ $TOOL = 'kaldi' ]; then
    steps/make_fbank.sh --cmd "$train_cmd" --nj 32 $CSJDATA_SAVEPATH/train exp/make_fbank/train $CSJDATA_SAVEPATH/fbank || exit 1;
    steps/compute_cmvn_stats.sh $CSJDATA_SAVEPATH/train exp/make_fbank/train $CSJDATA_SAVEPATH/fbank || exit 1;
    utils/fix_data_dir.sh $CSJDATA_SAVEPATH/train || exit;
    for x in dev eval1 eval2 eval3; do
      steps/make_fbank.sh --nj 10 --cmd "$train_cmd" $CSJDATA_SAVEPATH/$x exp/make_fbank/$x $CSJDATA_SAVEPATH/fbank || exit 1;
      steps/compute_cmvn_stats.sh $CSJDATA_SAVEPATH/$x exp/make_fbank/$x $CSJDATA_SAVEPATH/fbank || exit 1;
      utils/fix_data_dir.sh $CSJDATA_SAVEPATH/$x || exit 1;
    done

  exit 1

  elif [ $TOOL = 'htk' ]; then
    # Make a config file to covert from wav to htk file
    python local/make_htk_config.py \
        --data_save_path $CSJDATA_SAVEPATH \
        --config_save_path ./conf \
        --channels $CHANNELS \
        --window $WINDOW \
        --slide $SLIDE \
        --energy $ENERGY \
        --delta $DELTA \
        --deltadelta $DELTADELTA || exit 1;

    # Convert from wav to htk files
    for data_type in train dev eval1 eval2 eval3 ; do
      mkdir -p $CSJDATA_SAVEPATH/$data_type/htk

      htk_paths=$(find $CSJDATA_SAVEPATH/$data_type/htk -iname '*.htk')
      htk_file_num=$(find $CSJDATA_SAVEPATH/$data_type/htk -iname '*.htk' | wc -l)

      if [ ! -e $CSJDATA_SAVEPATH/$data_type/htk/.done_make_htk ]; then
        $HCOPY -T 1 -C ./conf/fbank.conf -S $CSJDATA_SAVEPATH/$data_type/wav2htk.scp || exit 1;
        touch $CSJDATA_SAVEPATH/$data_type/htk/.done_make_htk
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
    --channels $CHANNELS \
    --window $WINDOW \
    --slide $SLIDE \
    --energy $ENERGY \
    --delta $DELTA \
    --deltadelta $DELTADELTA || exit 1;

  echo "Finish feature extranction (stage: 1)."
fi

# TODO: data_type下に作るといちいち消えるからやめる



echo ============================================================================
echo "                            Create dataset                                "
echo ============================================================================
if [ $stage -le 2 ]; then

  echo "Finish creating dataset (stage: 2)."
fi

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
  #   --model_save_path $MODEL_SAVEPATH > log/$filename".log" &

  CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
  $PYTHON train.py \
    --gpu $gpu_index \
    --config_path $config_path \
    --model_save_path $MODEL_SAVEPATH || exit 1;

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
