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
echo "                              LibriSpeech                                 "
echo ============================================================================

stage=0
hierarchical_model=false
# hierarchical_model=true
run_background=true
restart=false

### Select data size
DATASIZE=100
# DATASIZE=460
# DATASIZE=960

### Set path to save dataset
DATA="/n/sd8/inaguma/corpus/librispeech/kaldi"

### Set path to save the model
MODEL="/n/sd8/inaguma/result/librispeech"

### Select one tool to extract features (HTK is the fastest)
# TOOL=kaldi
TOOL=htk
# TOOL=python_speech_features
# TOOL=librosa

### Configuration of feature extra./nction
CHANNELS=80
WINDOW=0.025
SLIDE=0.01
ENERGY=1
DELTA=1
DELTADELTA=1
# NORMALIZE=global
NORMALIZE=speaker
# NORMALIZE=utterance

if [ $DATASIZE = '100' ]; then
  train=train_clean_100
elif [ $DATASIZE = '460' ]; then
  train=train_clean_460
elif [ $DATASIZE = '960' ]; then
  train=train_other_960
fi
export DATA_DOWNLOADPATH=$DATA/data

if [ $stage -le 0 ] && [ ! -e $DATA/.stage_0_$DATASIZE ]; then
  echo ============================================================================
  echo "                           Data Preparation                               "
  echo ============================================================================

  # Base url for downloads.
  data_url=www.openslr.org/resources/12
  lm_url=www.openslr.org/resources/11

  mkdir -p $DATA
  mkdir -p $DATA_DOWNLOADPATH

  # download data
  for part in dev-clean test-clean dev-other test-other train-clean-100; do
    local/download_and_untar.sh $DATA_DOWNLOADPATH $data_url $part || exit 1;
  done

  # format the data as Kaldi data directories
  for part in dev-clean test-clean dev-other test-other train-clean-100; do
    # use underscore-separated names in data directories.
    local/data_prep.sh $DATA_DOWNLOADPATH/LibriSpeech/$part $DATA/$(echo $part | sed s/-/_/g) || exit 1;
  done
  cp -rf $DATA_DOWNLOADPATH/LibriSpeech/train-clean-100 $DATA_DOWNLOADPATH/LibriSpeech/train_100

  if [ $DATASIZE = "460" ] || [ $DATASIZE = '960' ]; then
    # download 360h clean data
    local/download_and_untar.sh $DATA_DOWNLOADPATH $data_url train-clean-360 || exit 1;
    # now add the "clean-360" subset to the mix ...
    local/data_prep.sh \
      $DATA_DOWNLOADPATH/LibriSpeech/train-clean-360 $DATA/train_clean_360 || exit 1;
    # ... and then combine the two sets into a 460 hour one
    utils/combine_data.sh \
      $DATA/train_460 $DATA/train_clean_100 $DATA/train_clean_360 || exit 1;

    if [ $DATASIZE = "960" ]; then
      # download 500h other data
      local/download_and_untar.sh $DATA_DOWNLOADPATH $data_url train-other-500 || exit 1;
      # prepare the 500 hour subset.
      local/data_prep.sh \
       $DATA_DOWNLOADPATH/LibriSpeech/train-other-500 $DATA/train_other_500 || exit 1;
      # combine all the data
      utils/combine_data.sh \
        $DATA/train_960 $DATA/train_460 $DATA/train_other_500 || exit 1;
    fi
  fi

  touch $DATA/.stage_0_$DATASIZE
  echo "Finish data preparation (stage: 0)."
fi


if [ $stage -le 1 ] && [ ! -e $DATA/.stage_1_$DATASIZE ]; then
  echo ============================================================================
  echo "                        Feature extranction                               "
  echo ============================================================================

  if [ $TOOL = "kaldi" ]; then
    for x in $train dev_clean dev_other test_clean test_other; do
      steps/make_fbank.sh --nj 8 --cmd run.pl $DATA/$x $DATA/make_fbank/$x $DATA/fbank || exit 1;
      steps/compute_cmvn_stats.sh $DATA/$x $DATA/make_fbank/$x $DATA/fbank || exit 1;
      utils/fix_data_dir.sh $DATA/$x || exit 1;
    done

  elif [ $TOOL = "htk" ]; then
    # Make a config file to covert from wav to htk file
    python local/make_htk_config.py \
        --data_save_path $DATA \
        --config_save_path ./conf/fbank_htk.conf \
        --audio_file_type wav \
        --channels $CHANNELS \
        --sampling_rate 16000 \
        --window $WINDOW \
        --slide $SLIDE \
        --energy $ENERGY \
        --delta $DELTA \
        --deltadelta $DELTADELTA || exit 1;

    for data_type in $train dev_clean dev_other test_clean test_other; do
      mkdir -p $DATA/wav/$data_type
      mkdir -p $DATA/htk/$data_type
      [ -e $DATA/$data_type/htk.scp ] && rm $DATA/$data_type/htk.scp
      touch $DATA/$data_type/htk.scp
      cat $DATA/$data_type/wav.scp | while read line
      do
        # Convert from flac to wav files
        flac_path=`echo $line | awk -F " " '{ print $(NF - 1) }'`
        speaker=`echo $line | awk -F "/" '{ print $(NF - 2) }'`
        chapter=`echo $line | awk -F "/" '{ print $(NF - 1) }'`
        mkdir -p $DATA/wav/$data_type/$speaker/$chapter
        file_name=`basename $flac_path`
        base=${file_name%.*}
        # ext=${file_name##*.}
        wav_path=$DATA/wav/$data_type/$speaker/$chapter/$base".wav"
        if [ ! -e $wav_path ]; then
          sox $flac_path -t wav $wav_path
        fi

        # Convert from wav to htk files
        mkdir -p $DATA/htk/$data_type/$speaker/$chapter
        htk_path=$DATA/htk/$data_type/$speaker/$chapter/$base".htk"
        if [ ! -e $htk_path ]; then
          echo $wav_path  $htk_path > ./tmp.scp
          $HCOPY -T 1 -C ./conf/fbank_htk.conf -S ./tmp.scp || exit 1;
          rm ./tmp.scp
        fi
        echo $htk_path >> $DATA/$data_type/htk.scp
      done
    done

  else
    if ! which sox >&/dev/null; then
      echo "This script requires you to first install sox";
      exit 1;
    fi
  fi

  python local/feature_extraction.py \
    --data_save_path $DATA \
    --tool $TOOL \
    --normalize $NORMALIZE \
    --channels $CHANNELS \
    --window $WINDOW \
    --slide $SLIDE \
    --energy $ENERGY \
    --delta $DELTA \
    --deltadelta $DELTADELTA || exit 1;

  touch $DATA/.stage_1_$DATASIZE
  echo "Finish feature extranction (stage: 1)."
fi


if [ $stage -le 2 ] && [ ! -e $DATA/.stage_2_$DATASIZE ]; then
  echo ============================================================================
  echo "                            Create dataset                                "
  echo ============================================================================

  python local/make_dataset_csv.py \
    --data_save_path $DATA \
    --tool $TOOL \
    --data_size $DATASIZE || exit 1;

  touch $DATA/.stage_2_$DATASIZE
  echo "Finish creating dataset (stage: 2)."
fi


exit 1


if [ $stage -le 3 ]; then
  echo ============================================================================
  echo "                             Training stage                               "
  echo ============================================================================

  config_path=$1
  gpu_index=$2
  filename=$(basename $config_path | awk -F. '{print $1}')

  mkdir -p log
  mkdir -p $MODEL

  echo "Start training..."

  if $hierarchical_model; then
    if $restart; then
      if $run_background; then
        CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
        nohup $PYTHON exp/training/train_hierarchical.py \
          --gpu $gpu_index \
          --saved_model_path $config_path \
          --data_save_path $DATA > log/$filename".log" &
      else
        CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
        nohup $PYTHON exp/training/train_hierarchical.py \
          --gpu $gpu_index \
          --saved_model_path $config_path \
          --data_save_path $DATA || exit 1;
      fi
    else
      if $run_background; then
        CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
        nohup $PYTHON exp/training/train_hierarchical.py \
          --gpu $gpu_index \
          --config_path $config_path \
          --model_save_path $MODEL \
          --data_save_path $DATA > log/$filename".log" &
      else
        CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
        $PYTHON exp/training/train_hierarchical.py \
          --gpu $gpu_index \
          --config_path $config_path \
          --model_save_path $MODEL \
          --data_save_path $DATA || exit 1;
      fi
    fi
  else
    if $restart; then
      if $run_background; then
        CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
        nohup $PYTHON exp/training/train.py \
          --gpu $gpu_index \
          --saved_model_path $config_path \
          --data_save_path $DATA > log/$filename".log" &
      else
        CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
        $PYTHON exp/training/train.py \
          --gpu $gpu_index \
          --saved_model_path $config_path \
          --data_save_path $DATA || exit 1;
      fi
    else
      if $run_background; then
        CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
        nohup $PYTHON exp/training/train.py \
          --gpu $gpu_index \
          --config_path $config_path \
          --model_save_path $MODEL \
          --data_save_path $DATA > log/$filename".log" &
      else
        CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
        $PYTHON exp/training/train.py \
          --gpu $gpu_index \
          --config_path $config_path \
          --model_save_path $MODEL \
          --data_save_path $DATA || exit 1;
      fi
    fi
  fi

  echo "Finish model training (stage: 3)."
fi


if [ $stage -le 4 ]; then
  echo ============================================================================
  echo "                             LM training                                 "
  echo ============================================================================

  # download the LM resources
  local/download_lm.sh $lm_url data/local/lm

  ## Optional text corpus normalization and LM training
  ## These scripts are here primarily as a documentation of the process that has been
  ## used to build the LM. Most users of this recipe will NOT need/want to run
  ## this step. The pre-built language models and the pronunciation lexicon, as
  ## well as some intermediate data(e.g. the normalized text used for LM training),
  ## are available for download at http://www.openslr.org/11/
  #local/lm/train_lm.sh $LM_CORPUS_ROOT \
  #  data/local/lm/norm/tmp data/local/lm/norm/norm_texts data/local/lm

  ## Optional G2P training scripts.
  ## As the LM training scripts above, this script is intended primarily to
  ## document our G2P model creation process
  #local/g2p/train_g2p.sh data/local/dict/cmudict data/local/lm

  echo "Finish LM training (stage: 4)."
fi


if [ $stage -le 5 ]; then
  echo ============================================================================
  echo "                              Rescoring                                   "
  echo ============================================================================

  echo "Finish rescoring (stage: 5)."
fi


echo "Done."


# when "--stage 3" option is used below we skip the G2P steps, and use the
# lexicon we have already downloaded from openslr.org/11/
# local/prepare_dict.sh --stage 3 --nj 30 --cmd "$train_cmd" \
#    data/local/lm data/local/lm data/local/dict_nosp

# utils/prepare_lang.sh data/local/dict_nosp \
#   "<UNK>" data/local/lang_tmp_nosp data/lang_nosp

# local/format_lms.sh --src-dir data/lang_nosp data/local/lm

# Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
# utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
#   data/lang_nosp data/lang_nosp_test_tglarge
# utils/build_const_arpa_lm.sh data/local/lm/lm_fglarge.arpa.gz \
#   data/lang_nosp data/lang_nosp_test_fglarge

# utils/prepare_lang.sh data/local/dict \
#   "<UNK>" data/local/lang_tmp data/lang
# local/format_lms.sh --src-dir data/lang data/local/lm

# utils/build_const_arpa_lm.sh \
#   data/local/lm/lm_tglarge.arpa.gz data/lang data/lang_test_tglarge
# utils/build_const_arpa_lm.sh \
#   data/local/lm/lm_fglarge.arpa.gz data/lang data/lang_test_fglarge


# this does some data-cleaning. The cleaned data should be useful when we add
# the neural net and chain systems.
# local/run_cleanup_segmentation.sh


# steps/cleanup/debug_lexicon.sh --remove-stress true  --nj 200 --cmd "$train_cmd" data/train_clean_100 \
#    data/lang exp/tri6b data/local/dict/lexicon.txt exp/debug_lexicon_100h

# #Perform rescoring of tri6b be means of faster-rnnlm
# #Attention: with default settings requires 4 GB of memory per rescoring job, so commenting this out by default
# wait && local/run_rnnlm.sh \
#     --rnnlm-ver "faster-rnnlm" \
#     --rnnlm-options "-hidden 150 -direct 1000 -direct-order 5" \
#     --rnnlm-tag "h150-me5-1000" $data data/local/lm

# #Perform rescoring of tri6b be means of faster-rnnlm using Noise contrastive estimation
# #Note, that could be extremely slow without CUDA
# #We use smaller direct layer size so that it could be stored in GPU memory (~2Gb)
# #Suprisingly, bottleneck here is validation rather then learning
# #Therefore you can use smaller validation dataset to speed up training
# wait && local/run_rnnlm.sh \
#     --rnnlm-ver "faster-rnnlm" \
#     --rnnlm-options "-hidden 150 -direct 400 -direct-order 3 --nce 20" \
#     --rnnlm-tag "h150-me3-400-nce20" $data data/local/lm


# # train models on cleaned-up data
# # we've found that this isn't helpful-- see the comments in local/run_data_cleaning.sh
# local/run_data_cleaning.sh
