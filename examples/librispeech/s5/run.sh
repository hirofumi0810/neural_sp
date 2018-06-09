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
run_background=true

### Select data size
datasize=100
# datasize=460
# datasize=960

### Set path to save dataset
libri="/n/sd8/inaguma/corpus/librispeech/kaldi"

### Set path to save the model
model="/n/sd8/inaguma/result/librispeech"

### Select one tool to extract features (HTK is the fastest)
tool=kaldi
# tool=htk
# tool=python_speech_features
# tool=librosa

### Configuration of feature extra./nction
channels=80
window=0.025
slide=0.01
energy=1
delta=1
deltadelta=1
do_delta=false
# normalize=global
normalize=speaker
# normalize=utterance

# Base url for downloads.
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11

train=train_${datasize}
if [ $stage -le 0 ] && [ ! -e ${libri}/.stage_0_${datasize} ]; then
  echo ============================================================================
  echo "                           Data Preparation                               "
  echo ============================================================================

  mkdir -p ${libri}/data

  # download data
  for part in dev-clean test-clean dev-other test-other train-clean-100; do
    local/download_and_untar.sh ${libri}/data $data_url $part || exit 1;
  done
  if [ ${datasize} = "460" ] || [ ${datasize} = '960' ]; then
    local/download_and_untar.sh ${libri}/data $data_url train-clean-360 || exit 1;
    if [ ${datasize} = "960" ]; then
      local/download_and_untar.sh ${libri}/data $data_url train-other-500 || exit 1;
    fi
  fi

  # format the data as Kaldi data directories
  for part in dev-clean test-clean dev-other test-other train-clean-100; do
    # use underscore-separated names in data directories.
    local/data_prep.sh ${libri}/data/LibriSpeech/$part ${libri}/$(echo $part | sed s/-/_/g) || exit 1;
  done
  if [ ${datasize} = "100" ]; then
    cp -rf ${libri}/train_clean_100 ${libri}/train_100
  else
    if [ ${datasize} = "460" ] || [ ${datasize} = '960' ]; then
      # now add the "clean-360" subset to the mix ...
      local/data_prep.sh \
        ${libri}/data/LibriSpeech/train-clean-360 ${libri}/train_clean_360 || exit 1;
      # ... and then combine the two sets into a 460 hour one
      utils/combine_data.sh \
        ${libri}/train_460 ${libri}/train_clean_100 ${libri}/train_clean_360 || exit 1;

      if [ ${datasize} = "960" ]; then
        # prepare the 500 hour subset.
        local/data_prep.sh \
         ${libri}/data/LibriSpeech/train-other-500 ${libri}/train_other_500 || exit 1;
        # combine all the data
        utils/combine_data.sh \
          ${libri}/train_960 ${libri}/train_460 ${libri}/train_other_500 || exit 1;
      fi
    fi
  fi

  # download the LM resources
  local/download_lm.sh $lm_url ${libri}/local/lm || exit 1;

  ## Optional text corpus normalization and LM training
  ## These scripts are here primarily as a documentation of the process that has been
  ## used to build the LM. Most users of this recipe will NOT need/want to run
  ## this step. The pre-built language models and the pronunciation lexicon, as
  ## well as some intermediate data(e.g. the normalized text used for LM training),
  ## are available for download at http://www.openslr.org/11/
  # local/lm/train_lm.sh $LM_CORPUS_ROOT \
  #  ${libri}/local/lm/norm/tmp ${libri}/local/lm/norm/norm_texts ${libri}/local/lm || exit 1;

  ## Optional G2P training scripts.
  ## As the LM training scripts above, this script is intended primarily to
  ## document our G2P model creation process
  # local/g2p/train_g2p.sh ${libri}/local/dict/cmudict ${libri}/local/lm || exit 1;

  touch ${libri}/.stage_0_${datasize}
  echo "Finish data preparation (stage: 0)."
fi


if ! which sox >&/dev/null; then
  echo "This script requires you to first install sox";
  exit 1;
fi


if [ $stage -le 1 ] && [ ! -e ${libri}/.stage_1_${datasize} ]; then
  echo ============================================================================
  echo "                        Feature extranction                               "
  echo ============================================================================

  # TODO: remove
  # local/lm/train_lm.sh $LM_CORPUS_ROOT \
  #  ${libri}/local/lm/norm/tmp ${libri}/local/lm/norm/norm_texts ${libri}/local/lm
  # local/g2p/train_g2p.sh ${libri}/local/dict/cmudict ${libri}/local/lm

  if [ ${tool} = "kaldi" ]; then
    # make filterbank features
    for x in ${train} dev_clean dev_other test_clean test_other; do
      steps/make_fbank.sh --nj 8 --cmd run.pl ${libri}/$x ${libri}/ark/log/make_fbank/$x ${libri}/ark || exit 1;
      mkdir -p ${libri}/feature/kaldi/${datasize}/$x
      feat-to-len scp:${libri}/feature/kaldi/${datasize}/$x/feats.scp ark,t:${libri}/feature/kaldi/${datasize}/$x/frame_num.scp || exit 1;
      exit 1

      if [ `echo $x | grep 'train'` ]; then
        # compute global CMVN
        compute-cmvn-stats scp:${libri}/${train}/feats.scp ${libri}/ark/cmvn_${train}.ark || exit 1;
        # steps/compute_cmvn_stats.sh ${libri}/$x ${libri}/ark/log/make_fbank/$x ${libri}/ark || exit 1;

        # dump features for training
        local/dump.sh --cmd "$train_cmd" --nj 80 --do_delta $do_delta \
            ${libri}/$x/feats.scp ${libri}/ark/cmvn_${train}.ark ${libri}/ark/log/dump_feats/$x ${libri}/feature/kaldi/${datasize}/$x || exit 1;
      else
        local/dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
            ${libri}/$x/feats.scp ${libri}/ark/cmvn_${train}.ark ${libri}/ark/log/dump_feats/$x ${libri}/feature/kaldi/${datasize}/$x || exit 1;
      fi
    done

  elif [ ${tool} = "htk" ]; then
    # Make a config file to covert from wav to htk file
    ${PYTHON} local/make_htk_config.py \
        --data_save_path ${libri} \
        --config_save_path ./conf/fbank_htk.conf \
        --audio_file_type wav \
        --channels ${channels} \
        --sampling_rate 16000 \
        --window ${window} \
        --slide ${slide} \
        --energy ${energy} \
        --delta ${delta} \
        --deltadelta ${deltadelta} || exit 1;

    for data_type in ${train} dev_clean dev_other test_clean test_other; do
      if [ `echo $data_type | grep 'train'` ]; then
        mkdir -p ${libri}/htk/train
        mkdir -p ${libri}/wav/train
      else
        mkdir -p ${libri}/wav/$data_type
        mkdir -p ${libri}/htk/$data_type
      fi
      [ -e ${libri}/$data_type/htk.scp ] && rm ${libri}/$data_type/htk.scp
      touch ${libri}/$data_type/htk.scp
      cat ${libri}/$data_type/wav.scp | while read line
      do
        # Convert from flac to wav files
        flac_path=`echo $line | awk -F " " '{ print $(NF - 1) }'`
        speaker=`echo $line | awk -F "/" '{ print $(NF - 2) }'`
        chapter=`echo $line | awk -F "/" '{ print $(NF - 1) }'`
        file_name=`basename $flac_path`
        base=${file_name%.*}
        # ext=${file_name##*.}
        if [ `echo $data_type | grep 'train'` ]; then
          mkdir -p ${libri}/wav/train/$speaker/$chapter
          wav_path=${libri}/wav/train/$speaker/$chapter/$base".wav"
        else
          mkdir -p ${libri}/wav/$data_type/$speaker/$chapter
          wav_path=${libri}/wav/$data_type/$speaker/$chapter/$base".wav"
        fi
        if [ ! -e $wav_path ]; then
          sox $flac_path -t wav $wav_path
        fi

        # Convert from wav to htk files
        if [ `echo $data_type | grep 'train'` ]; then
          mkdir -p ${libri}/htk/train/$speaker/$chapter
          htk_path=${libri}/htk/train/$speaker/$chapter/$base".htk"
        else
          mkdir -p ${libri}/htk/$data_type/$speaker/$chapter
          htk_path=${libri}/htk/$data_type/$speaker/$chapter/$base".htk"
        fi
        if [ ! -e $htk_path ]; then
          echo $wav_path  $htk_path > ./tmp.scp
          $HCOPY -T 1 -C ./conf/fbank_htk.conf -S ./tmp.scp || exit 1;
          rm ./tmp.scp
        fi
        echo $htk_path >> ${libri}/$data_type/htk.scp
      done
    done

  else
    if ! which sox >&/dev/null; then
      echo "This script requires you to first install sox";
      exit 1;
    fi
  fi

  ${PYTHON} local/feature_extraction.py \
    --data_save_path ${libri} \
    --data_size ${datasize} \
    --tool ${tool} \
    --normalize ${normalize} \
    --channels ${channels} \
    --window ${window} \
    --slide ${slide} \
    --energy ${energy} \
    --delta ${delta} \
    --deltadelta ${deltadelta} || exit 1;

  touch ${libri}/.stage_1_${datasize}
  echo "Finish feature extranction (stage: 1)."
fi


if [ $stage -le 2 ] && [ ! -e ${libri}/.stage_2_${datasize} ]; then
  echo ============================================================================
  echo "                            Create dataset                                "
  echo ============================================================================

  ${PYTHON} local/make_dataset_csv.py \
    --data_save_path ${libri} \
    --tool ${tool} \
    --data_size ${datasize} || exit 1;

  touch ${libri}/.stage_2_${datasize}
  echo "Finish creating dataset (stage: 2)."
fi


if [ $stage -le 3 ]; then
  echo ============================================================================
  echo "                             Training stage                               "
  echo ============================================================================

  config_path=$1
  gpu_index=$2
  filename=$(basename ${config_path} | awk -F. '{print $1}')

  mkdir -p log
  mkdir -p ${model}

  echo "Start training..."

  if [ `echo ${config_path} | grep 'hierarchical'` ]; then
    if [ `echo ${config_path} | grep 'result'` ]; then
      if $run_background; then
        CUDA_VISIBLE_DEVICES=${gpu_index} \
        nohup ${PYTHON} exp/training/train_hierarchical.py \
          --gpu ${gpu_index} \
          --saved_model_path ${config_path} \
          --data_save_path ${libri} > log/$filename".log" &
      else
        CUDA_VISIBLE_DEVICES=${gpu_index} \
        nohup ${PYTHON} exp/training/train_hierarchical.py \
          --gpu ${gpu_index} \
          --saved_model_path ${config_path} \
          --data_save_path ${libri} || exit 1;
      fi
    else
      if $run_background; then
        CUDA_VISIBLE_DEVICES=${gpu_index} \
        nohup ${PYTHON} exp/training/train_hierarchical.py \
          --gpu ${gpu_index} \
          --config_path ${config_path} \
          --model_save_path ${model} \
          --data_save_path ${libri} > log/$filename".log" &
      else
        CUDA_VISIBLE_DEVICES=${gpu_index} \
        ${PYTHON} exp/training/train_hierarchical.py \
          --gpu ${gpu_index} \
          --config_path ${config_path} \
          --model_save_path ${model} \
          --data_save_path ${libri} || exit 1;
      fi
    fi
  else
    if [ `echo ${config_path} | grep 'result'` ]; then
      if $run_background; then
        CUDA_VISIBLE_DEVICES=${gpu_index} \
        nohup ${PYTHON} exp/training/train.py \
          --gpu ${gpu_index} \
          --saved_model_path ${config_path} \
          --data_save_path ${libri} > log/$filename".log" &
      else
        CUDA_VISIBLE_DEVICES=${gpu_index} \
        ${PYTHON} exp/training/train.py \
          --gpu ${gpu_index} \
          --saved_model_path ${config_path} \
          --data_save_path ${libri} || exit 1;
      fi
    else
      if $run_background; then
        CUDA_VISIBLE_DEVICES=${gpu_index} \
        nohup ${PYTHON} exp/training/train.py \
          --gpu ${gpu_index} \
          --config_path ${config_path} \
          --model_save_path ${model} \
          --data_save_path ${libri} > log/$filename".log" &
      else
        CUDA_VISIBLE_DEVICES=${gpu_index} \
        ${PYTHON} exp/training/train.py \
          --gpu ${gpu_index} \
          --config_path ${config_path} \
          --model_save_path ${model} \
          --data_save_path ${libri} || exit 1;
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
