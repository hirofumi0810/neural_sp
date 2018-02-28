#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./run.sh path_to_config_file gpu_index" 1>&2
  exit 1
fi


echo ============================================================================
echo "                                   WSJ                                     "
echo ============================================================================

stage=0

### Set path to original data
wsj0="/export/corpora5/LDC/LDC93S6B"
wsj1="/export/corpora5/LDC/LDC94S13B"

### Set path to save dataset
export WSJDATA_SAVEPATH="/n/sd8/inaguma/corpus/wsj/kaldi"

### Set path to save the model
MODEL_SAVEPATH="/n/sd8/inaguma/result"
# MODEL_SAVEPATH="/n/sd8/inaguma/result/wsj"

### Select one tool to extract features (HTK is the fastest)
TOOL = 'kaldi'
# TOOL='htk'
# TOOL='python_speech_features'
# TOOL='librosa'

### Configuration of feature extranction
FEATURE_TYPE='fbank'
# FEATURE_TYPE='mfcc'
# FEATURE_TYPE='wav'

CHANNELS=40
WINDOW=0.025
SLIDE=0.01
ENERGY=1
DELTA=1
DELTADELTA=1
# NORMALIZE='global'
NORMALIZE='speaker'
# NORMALIZE='utterance'
# NORMALIZE='no'
# NOTE: normalize in [-1, 1] in case of wav


echo ============================================================================
echo "                           Data Preparation                               "
echo ============================================================================
if [ $stage -le 0 ]; then
  # data preparation.
  local/wsj_data_prep.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.?  || exit 1;

  # Sometimes, we have seen WSJ distributions that do not have subdirectories
  # like '11-13.1', but instead have 'doc', 'si_et_05', etc. directly under the
  # wsj0 or wsj1 directories. In such cases, try the following:
  #
  # corpus=/exports/work/inf_hcrc_cstr_general/corpora/wsj
  # local/cstr_wsj_data_prep.sh $corpus
  # rm data/local/dict/lexiconp.txt
  # $corpus must contain a 'wsj0' and a 'wsj1' subdirectory for this to work.
  #
  # "nosp" refers to the dictionary before silence probabilities and pronunciation
  # probabilities are added.
  local/wsj_prepare_dict.sh --dict-suffix "_nosp" || exit 1;

  # utils/prepare_lang.sh data/local/dict_nosp \
  #                       "<SPOKEN_NOISE>" data/local/lang_tmp_nosp data/lang_nosp || exit 1;

  local/wsj_format_data.sh --lang-suffix "_nosp" || exit 1;

  # We suggest to run the next three commands in the background,
  # as they are not a precondition for the system building and
  # most of the tests: these commands build a dictionary
  # containing many of the OOVs in the WSJ LM training data,
  # and an LM trained directly on that data (i.e. not just
  # copying the arpa files from the disks from LDC).
  # Caution: the commands below will only work if $decode_cmd
  # is setup to use qsub.  Else, just remove the --cmd option.
  # NOTE: If you have a setup corresponding to the older cstr_wsj_data_prep.sh style,
  # use local/cstr_wsj_extend_dict.sh --dict-suffix "_nosp" $corpus/wsj1/doc/ instead.
  # (
  #   local/wsj_extend_dict.sh --dict-suffix "_nosp" $wsj1/13-32.1  && \
  #     utils/prepare_lang.sh data/local/dict_nosp_larger \
  #                           "<SPOKEN_NOISE>" data/local/lang_tmp_nosp_larger data/lang_nosp_bd && \
  #     local/wsj_train_lms.sh --dict-suffix "_nosp" &&
  #     local/wsj_format_local_lms.sh --lang-suffix "_nosp" # &&
  # ) &

  utils/subset_data_dir.sh --first data/train_si284 7138 data/train_si84 || exit 1;

  echo "Finish data preparation (stage: 0)."
fi

# Calculate the amount of utterance segmentations.
# perl -ne 'split; $s+=($_[3]-$_[2]); END{$h=int($s/3600); $r=($s-$h*3600); $m=int($r/60); $r-=$m*60; printf "%.1f sec -- %d:%d:%.1f\n", $s, $h, $m, $r;}' $CSJDATA_SAVEPATH/train/segments


echo ============================================================================
echo "                        Feature extranction                               "
echo ============================================================================
if [ $stage -le 1 ]; then
  if [ $TOOL = 'kaldi' ]; then
    steps/make_fbank.sh --cmd "$train_cmd" --nj 32 $WSJDATA_SAVEPATH/train exp/make_fbank/train $WSJDATA_SAVEPATH/fbank || exit 1;
    steps/compute_cmvn_stats.sh $WSJDATA_SAVEPATH/train exp/make_fbank/train $WSJDATA_SAVEPATH/fbank || exit 1;
    utils/fix_data_dir.sh $WSJDATA_SAVEPATH/train || exit;
    for x in dev eval1 eval2 eval3; do
      steps/make_fbank.sh --nj 10 --cmd "$train_cmd" $WSJDATA_SAVEPATH/$x exp/make_fbank/$x $WSJDATA_SAVEPATH/fbank || exit 1;
      steps/compute_cmvn_stats.sh $WSJDATA_SAVEPATH/$x exp/make_fbank/$x $WSJDATA_SAVEPATH/fbank || exit 1;
      utils/fix_data_dir.sh $WSJDATA_SAVEPATH/$x || exit 1;
    done

  elif [ $TOOL = 'htk' ]; then
    # Make a config file to covert from wav to htk file
    python local/make_htk_config.py \
        --data_save_path $WSJDATA_SAVEPATH \
        --config_save_path ./conf \
        --channels $CHANNELS \
        --window $WINDOW \
        --slide $SLIDE \
        --energy $ENERGY \
        --delta $DELTA \
        --deltadelta $DELTADELTA || exit 1;

    # Convert from wav to htk files
    for data_type in test_eval92 test_eval93 test_dev93 train_si284; do
      mkdir -p $WSJDATA_SAVEPATH/$data_type/htk

      htk_paths=$(find $WSJDATA_SAVEPATH/$data_type/htk -iname '*.htk')
      htk_file_num=$(find $WSJDATA_SAVEPATH/$data_type/htk -iname '*.htk' | wc -l)

      if [ $htk_file_num -ne ${file_number[$data_type]} ]; then
        $HCOPY -T 1 -C ./conf/fbank.conf -S $WSJDATA_SAVEPATH/$data_type/wav2htk.scp || exit 1;
        touch $WSJDATA_SAVEPATH/$data_type/htk/.done_make_htk
      fi
    done

  else
    if ! which sox >&/dev/null; then
      echo "This script requires you to first install sox";
      exit 1;
    fi
  fi

  python local/feature_extraction.py \
    --data_save_path $WSJDATA_SAVEPATH \
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

exit 1


# utils/prepare_lang.sh data/local/dict \
#   "<SPOKEN_NOISE>" data/local/lang_tmp data/lang || exit 1;

# utils/prepare_lang.sh data/local/dict_larger \
#   "<SPOKEN_NOISE>" data/local/lang_tmp_larger data/lang_bd || exit 1;

# The following demonstrate how to re-segment long audios.
# local/run_segmentation_long_utts.sh

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
