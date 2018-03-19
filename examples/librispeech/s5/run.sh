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

### Set true when hiding progress bar
run_background=true

### Set true when restarting training
restart=false

### Select data size
DATASIZE=100h
# DATASIZE=460h
# DATASIZE=960h

### Set path to save dataset
DATA_SAVEPATH="/n/sd8/inaguma/corpus/librispeech/kaldi"

### Set path to save the model
MODEL_SAVEPATH="/n/sd8/inaguma/result/librispeech"

### Select one tool to extract features (HTK is the fastest)
# TOOL=kaldi
TOOL=htk
# TOOL=python_speech_features
# TOOL=librosa
# # TOOL=wav

### Configuration of feature extra./nction
CHANNELS=4
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


export DATA_DOWNLOADPATH=$DATA_SAVEPATH/data
export DATA_SAVEPATH=$DATA_SAVEPATH/$DATASIZE
if [ $stage -le 0 ]; then
  echo ============================================================================
  echo "                           Data Preparation                               "
  echo ============================================================================

  # Base url for downloads.
  data_url=www.openslr.org/resources/12
  lm_url=www.openslr.org/resources/11

  mkdir -p $DATA_SAVEPATH
  mkdir -p $DATA_DOWNLOADPATH

  # download data
  for part in dev-clean test-clean dev-other test-other train-clean-100; do
    local/download_and_untar.sh $DATA_DOWNLOADPATH $data_url $part || exit 1;
  done

  # format the data as Kaldi data directories
  # for part in dev-clean test-clean dev-other test-other train-clean-100; do
  #   # use underscore-separated names in data directories.
  #   local/data_prep.sh $DATA_DOWNLOADPATH/LibriSpeech/$part $DATA_SAVEPATH/$(echo $part | sed s/-/_/g) || exit 1;
  # done

  if [ $DATASIZE != "100h" ]; then
    # download 360h clean data
    local/download_and_untar.sh $DATA_DOWNLOADPATH $data_url train-clean-360 || exit 1;

    # now add the "clean-360" subset to the mix ...
    local/data_prep.sh \
      $DATA_DOWNLOADPATH/LibriSpeech/train-clean-360 $DATA_SAVEPATH/train_clean_360 || exit 1;

    # ... and then combine the two sets into a 460 hour one
    utils/combine_data.sh \
      $DATA_SAVEPATH/train_clean_460 $DATA_SAVEPATH/train_clean_100 $DATA_SAVEPATH/train_clean_360 || exit 1;

    if [ $DATASIZE = "960h" ]; then
      # download 500h other data
      local/download_and_untar.sh $DATA_DOWNLOADPATH $data_url train-other-500 || exit 1;

      # prepare the 500 hour subset.
      local/data_prep.sh \
       $DATA_DOWNLOADPATH/LibriSpeech/train-other-500 $DATA_SAVEPATH/train_other_500 || exit 1;

      # combine all the data
      utils/combine_data.sh \
        $DATA_SAVEPATH/train_960 $DATA_SAVEPATH/train_clean_460 $DATA_SAVEPATH/train_other_500 || exit 1;
    fi
  fi

  echo ============================================================================
  echo "                        Convert from flac to wav                          "
  echo ============================================================================

  flac_paths=$(find $DATA_DOWNLOADPATH/. -iname '*.flac')
  for flac_path in $flac_paths ; do
    dir_path=$(dirname $flac_path)
    file_name=$(basename $flac_path)
    base=${file_name%.*}
    ext=${file_name##*.}
    wav_path=$dir_path/$base".wav"
    if [ $ext = "flac" ]; then
      echo "Converting from "$flac_path" to "$wav_path
      sox $flac_path -t wav $wav_path
      # rm -f $flac_path
    else
      echo "Already converted: "$wav_path
    fi
  done

  echo "Finish data preparation (stage: 0)."
fi


if [ $stage -le 1 ]; then
  echo ============================================================================
  echo "                        Feature extranction                               "
  echo ============================================================================

  if [ $TOOL = "kaldi" ]; then
    for x in train dev-clean dev-other test-clean test-other; do
      steps/make_fbank.sh --nj 8 --cmd run.pl $DATA_SAVEPATH/$x $DATA_SAVEPATH/make_fbank/$x $DATA_SAVEPATH/fbank || exit 1;
      steps/compute_cmvn_stats.sh $DATA_SAVEPATH/$x $DATA_SAVEPATH/make_fbank/$x $DATA_SAVEPATH/fbank || exit 1;
      utils/fix_data_dir.sh $DATA_SAVEPATH/$x || exit 1;
    done

  elif [ $TOOL = "htk" ]; then
    # Make a config file to covert from wav to htk file
    python local/make_htk_config.py \
        --data_save_path $DATA_SAVEPATH \
        --config_save_path ./conf/fbank_htk.conf \
        --data_size $DATASIZE \
        --channels $CHANNELS \
        --window $WINDOW \
        --slide $SLIDE \
        --energy $ENERGY \
        --delta $DELTA \
        --deltadelta $DELTADELTA || exit 1;

    # Convert from wav to htk files
    for data_type in train dev-clean dev-other test-clean test-other; do
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


exit 1


if [ $stage -le 2 ]; then
  echo ============================================================================
  echo "                            Create dataset                                "
  echo ============================================================================

  if [ ! -e $DATA_SAVEPATH/dataset/$TOOL/.done_dataset ]; then
    python local/make_dataset_csv.py \
      --data_save_path $DATA_SAVEPATH \
      --tool $TOOL \
      --data_size $DATASIZE || exit 1;
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
