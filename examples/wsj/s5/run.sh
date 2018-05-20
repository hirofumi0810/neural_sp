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
hierarchical_model=false
# hierarchical_model=true
run_background=true
restart=false

### Set path to original data
# wsj0="/n/rd21/corpora_1/WSJ/wsj0"
# wsj1="/n/rd21/corpora_1/WSJ/wsj1"

# Sometimes, we have seen WSJ distributions that do not have subdirectories
# like '11-13.1', but instead have 'doc', 'si_et_05', etc. directly under the
# wsj0 or wsj1 directories. In such cases, try the following:
#
# corpus=/exports/work/inf_hcrc_cstr_general/corpora/wsj
cstr_wsj="/n/rd21/corpora_1/WSJ"
# $corpus must contain a 'wsj0' and a 'wsj1' subdirectory for this to work.


### Set path to save dataset
export DATA_SAVEPATH="/n/sd8/inaguma/corpus/wsj/kaldi"

### Set path to save the model
MODEL_SAVEPATH="/n/sd8/inaguma/result/wsj"

### Select one tool to extract features (HTK is the fastest)
# TOOL=kaldi
TOOL=htk
# TOOL=python_speech_features
# TOOL=librosa

### Configuration of feature extranction
CHANNELS=80
WINDOW=0.025
SLIDE=0.01
ENERGY=1
DELTA=1
DELTADELTA=1
# NORMALIZE=global
NORMALIZE=speaker
# NORMALIZE=utterance


if [ ! -e $KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe ]; then
  echo ============================================================================
  echo "                           Install sph2pipe                               "
  echo ============================================================================
  SWBD_REPO=`pwd`
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

  if [ ! -e KALDI_ROOT/tools/sph2pipe_v2.5.tar.gz ]; then
    wget -T 3 -t 3 http://www.openslr.org/resources/3/sph2pipe_v2.5.tar.gz -P KALDI_ROOT/tools
  else
    echo "sph2pipe_v2.5.tar.gz is already downloaded."
  fi
  tar -xovzf KALDI_ROOT/tools/sph2pipe_v2.5.tar.gz -C $KALDI_ROOT/tools
  rm $KALDI_ROOT/tools/sph2pipe_v2.5.tar.gz
  echo "Enter into $KALDI_ROOT/tools/sph2pipe_v2.5 ..."
  cd $KALDI_ROOT/tools/sph2pipe_v2.5
  gcc -o sph2pipe *.c -lm
  echo "Get out of $KALDI_ROOT/tools/sph2pipe_v2.5 ..."
  cd $SWBD_REPO
fi


if [ $stage -le 0 ] && [ ! -e $DATA_SAVEPATH/.stage_0 ]; then
  echo ============================================================================
  echo "                           Data Preparation                               "
  echo ============================================================================

  # data preparation.
  # local/wsj_data_prep.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.?  || exit 1;

  local/cstr_wsj_data_prep.sh $cstr_wsj || exit 1;
  # rm $DATA_SAVEPATH/local/dict/lexiconp.txt

  # "nosp" refers to the dictionary before silence probabilities and pronunciation
  # probabilities are added.
  # local/wsj_prepare_dict.sh --dict-suffix "_nosp" || exit 1;

  # utils/prepare_lang.sh $DATA_SAVEPATH/local/dict_nosp \
  #                       "<SPOKEN_NOISE>" $DATA_SAVEPATH/local/lang_tmp_nosp data/lang_nosp || exit 1;

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
  #     utils/prepare_lang.sh $DATA_SAVEPATH/local/dict_nosp_larger \
  #                           "<SPOKEN_NOISE>" $DATA_SAVEPATH/local/lang_tmp_nosp_larger data/lang_nosp_bd && \
  #     local/wsj_train_lms.sh --dict-suffix "_nosp" &&
  #     local/wsj_format_local_lms.sh --lang-suffix "_nosp" # &&
  # ) &

  utils/subset_data_dir.sh --first $DATA_SAVEPATH/train_si284 7138 $DATA_SAVEPATH/train_si84 || exit 1

  touch $DATA_SAVEPATH/.stage_0
  echo "Finish data preparation (stage: 0)."
fi


if [ $stage -le 1 ] && [ ! -e $DATA_SAVEPATH/.stage_1 ]; then
  echo ============================================================================
  echo "                        Feature extranction                               "
  echo ============================================================================

  if [ $TOOL = "kaldi" ]; then
    for x in train_si84 train_si284 test_dev93 test_eval92; do
      steps/make_fbank.sh --nj 8 --cmd run.pl $DATA_SAVEPATH/$x exp/make_fbank/$x $DATA_SAVEPATH/fbank || exit 1;
      steps/compute_cmvn_stats.sh $DATA_SAVEPATH/$x exp/make_fbank/$x $DATA_SAVEPATH/fbank || exit 1;
      utils/fix_data_dir.sh $DATA_SAVEPATH/$x || exit 1;
    done

  elif [ $TOOL = "htk" ]; then
    # Make a config file to covert from wav to htk file
    python local/make_htk_config.py \
        --data_save_path $DATA_SAVEPATH \
        --config_save_path ./conf/fbank_htk.conf \
        --audio_file_type wav \
        --channels $CHANNELS \
        --sampling_rate 16000 \
        --window $WINDOW \
        --slide $SLIDE \
        --energy $ENERGY \
        --delta $DELTA \
        --deltadelta $DELTADELTA || exit 1;

    for data_type in train_si84 train_si284 test_dev93 test_eval92; do
      mkdir -p $DATA_SAVEPATH/wav/$data_type
      mkdir -p $DATA_SAVEPATH/htk/$data_type
      [ -e $DATA_SAVEPATH/$data_type/htk.scp ] && rm $DATA_SAVEPATH/$data_type/htk.scp
      touch $DATA_SAVEPATH/$data_type/htk.scp
      cat $DATA_SAVEPATH/$data_type/wav.scp | while read line
      do
        # Convert from sph to wav files
        sph_path=`echo $line | awk -F " " '{ print $(NF - 1) }'`
        speaker=`echo $line | awk -F "/" '{ print $(NF - 1) }'`
        mkdir -p $DATA_SAVEPATH/wav/$data_type/$speaker
        file_name=`basename $sph_path`
        base=${file_name%.*}
        # ext=${file_name##*.}
        wav_path=$DATA_SAVEPATH/wav/$data_type/$speaker/$base".wav"
        $KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe -f wav $sph_path $wav_path || exit 1;

        # Convert from wav to htk files
        mkdir -p $DATA_SAVEPATH/htk/$data_type/$speaker
        htk_path=$DATA_SAVEPATH/htk/$data_type/$speaker/$base".htk"
        if [ ! -e $htk_path ]; then
          echo $wav_path  $htk_path > ./tmp.scp
          $HCOPY -T 1 -C ./conf/fbank_htk.conf -S ./tmp.scp || exit 1;
          rm ./tmp.scp
        fi
        echo $htk_path >> $DATA_SAVEPATH/$data_type/htk.scp
      done
    done

  else
    if ! which sox >&/dev/null; then
      echo "This script requires you to first install sox";
      exit 1;
    fi
  fi

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

  touch $DATA_SAVEPATH/.stage_1
  echo "Finish feature extranction (stage: 1)."
fi


if [ $stage -le 2 ] && [ ! -e $DATA_SAVEPATH/.stage_2 ]; then
  echo ============================================================================
  echo "                            Create dataset                                "
  echo ============================================================================

  python local/make_dataset_csv.py \
    --data_save_path $DATA_SAVEPATH \
    --tool $TOOL || exit 1;

  touch $DATA_SAVEPATH/.stage_2
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


echo "Done."


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


# The following demonstrate how to re-segment long audios.
# local/run_segmentation_long_utts.sh

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
