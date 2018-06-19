#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

. utils/parse_options.sh  # e.g. this parses the --{stage} option if supplied.

if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./run.sh path_to_config_file {gpu_id}" 1>&2
  exit 1
fi


echo ============================================================================
echo "                                   WSJ                                     "
echo ============================================================================

stage=0
run_background=true
# run_background=false

### Set path to original data
wsj0="/n/rd21/corpora_1/WSJ/wsj0"
wsj1="/n/rd21/corpora_1/WSJ/wsj1"

# Sometimes, we have seen WSJ distributions that do not have subdirectories
# like '11-13.1', but instead have 'doc', 'si_et_05', etc. directly under the
# wsj0 or wsj1 directories. In such cases, try the following:
CSTR_WSJTATATOP="/n/rd21/corpora_1/WSJ"
# $CSTR_WSJTATATOP must contain a 'wsj0' and a 'wsj1' subdirectory for this to work.

directory_type=cstr # or original

### Set path to save dataset
export data="/n/sd8/inaguma/corpus/wsj/kaldi"

### Set path to save the model
model="/n/sd8/inaguma/result/wsj"

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


if [ ${stage} -le 0 ] && [ ! -e ${data}/.stage_0 ]; then
  echo ============================================================================
  echo "                           Data Preparation                               "
  echo ============================================================================

  # data preparation.
  if [ $directory_type = "original" ]; then
    local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?  || exit 1;
  elif [ $directory_type = "cstr" ]; then
    local/cstr_wsj_data_prep.sh $CSTR_WSJTATATOP || exit 1;
    # rm ${data}/local/dict/lexiconp.txt
  else
    echo "directory_type is original or cstr.";
    exit 1;
  fi

  # "nosp" refers to the dictionary before silence probabilities and pronunciation
  # probabilities are added.
  # local/wsj_prepare_dict.sh --dict-suffix "_nosp" || exit 1;

  # utils/prepare_lang.sh ${data}/local/dict_nosp \
  #                       "<SPOKEN_NOISE>" ${data}/local/lang_tmp_nosp data/lang_nosp || exit 1;

  local/wsj_format_data.sh --lang-suffix "_nosp" || exit 1;

  # TODO:
  # local/cstr_wsj_extend_dict.sh --dict-suffix "_nosp" $CSTR_WSJTATATOP/wsj1/doc/

  # We suggest to run the next three commands in the background,
  # as they are not a precondition for the system building and
  # most of the tests: these commands build a dictionary
  # containing many of the OOVs in the WSJ LM training data,
  # and an LM trained directly on that data (i.e. not just
  # copying the arpa files from the disks from LDC).
  # Caution: the commands below will only work if $decode_cmd
  # is setup to use qsub.  Else, just remove the --cmd option.
  # NOTE: If you have a setup corresponding to the older cstr_wsj_data_prep.sh style,
  # use local/cstr_wsj_extend_dict.sh --dict-suffix "_nosp" $CSTR_WSJTATATOP/wsj1/doc/ instead.
  # (
  #   local/wsj_extend_dict.sh --dict-suffix "_nosp" ${wsj1}/13-32.1  && \
  #     utils/prepare_lang.sh ${data}/local/dict_nosp_larger \
  #                           "<SPOKEN_NOISE>" ${data}/local/lang_tmp_nosp_larger data/lang_nosp_bd && \
  #     local/wsj_train_lms.sh --dict-suffix "_nosp" &&
  #     local/wsj_format_local_lms.sh --lang-suffix "_nosp" # &&
  # ) &

  utils/subset_data_dir.sh --first ${data}/train_si284 7138 ${data}/train_si84 || exit 1

  touch ${data}/.stage_0
  echo "Finish data preparation ({stage}: 0)."
fi


if [ ${stage} -le 1 ] && [ ! -e ${data}/.stage_1 ]; then
  echo ============================================================================
  echo "                        Feature extranction                               "
  echo ============================================================================

  if [ ${tool} = "kaldi" ]; then
    for x in train_si84 train_si284 test_dev93 test_eval92; do
      steps/make_fbank.sh --nj 8 --cmd run.pl ${data}/$x exp/make_fbank/$x ${data}/fbank || exit 1;
      steps/compute_cmvn_stats.sh ${data}/$x exp/make_fbank/$x ${data}/fbank || exit 1;
      utils/fix_data_dir.sh ${data}/$x || exit 1;
    done

  elif [ ${tool} = "htk" ]; then
    # Make a config file to covert from wav to htk file
    ${PYTHON} local/make_htk_config.py \
        --data_save_path ${data} \
        --config_save_path ./conf/fbank_htk.conf \
        --audio_file_type wav \
        --channels ${channels} \
        --sampling_rate 16000 \
        --window ${window} \
        --slide ${slide} \
        --energy ${energy} \
        --delta ${delta} \
        --deltadelta ${deltadelta} || exit 1;

    for data_type in train_si84 train_si284 test_dev93 test_eval92; do
      mkdir -p ${data}/wav/$data_type
      mkdir -p ${data}/htk/$data_type
      [ -e ${data}/$data_type/htk.scp ] && rm ${data}/$data_type/htk.scp
      touch ${data}/$data_type/htk.scp
      cat ${data}/$data_type/wav.scp | while read line
      do
        # Convert from sph to wav files
        sph_path=`echo $line | awk -F " " '{ print $(NF - 1) }'`
        speaker=`echo $line | awk -F "/" '{ print $(NF - 1) }'`
        mkdir -p ${data}/wav/$data_type/$speaker
        file_name=`basename $sph_path`
        base=${file_name%.*}
        # ext=${file_name##*.}
        wav_path=${data}/wav/$data_type/$speaker/$base".wav"
        if [ ! -e $wav_path ]; then
          ${KALDI_ROOT}/tools/sph2pipe_v2.5/sph2pipe -f wav $sph_path $wav_path || exit 1;
        fi

        # Convert from wav to htk files
        mkdir -p ${data}/htk/$data_type/$speaker
        htk_path=${data}/htk/$data_type/$speaker/$base".htk"
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
    --tool ${tool} \
    --normalize ${normalize} \
    --channels ${channels} \
    --window ${window} \
    --slide ${slide} \
    --energy ${energy} \
    --delta ${delta} \
    --deltadelta ${deltadelta} || exit 1;

  touch ${data}/.stage_1
  echo "Finish feature extranction ({stage}: 1)."
fi


if [ ${stage} -le 2 ] && [ ! -e ${data}/.stage_2 ]; then
  echo ============================================================================
  echo "                            Create dataset                                "
  echo ============================================================================

  ${PYTHON} local/make_dataset_csv.py \
    --data_save_path ${data} \
    --tool ${tool} || exit 1;

  touch ${data}/.stage_2
  echo "Finish creating dataset ({stage}: 2)."
fi


if [ ${stage} -le 3 ]; then
  echo ============================================================================
  echo "                             Training stage                               "
  echo ============================================================================

  config_path=$1
  gpu_id=$2
  filename=$(basename $config_path | awk -F. '{print $1}')

  mkdir -p log
  mkdir -p ${model}

  echo "Start training..."

  if [ `echo $config_path | grep 'hierarchical'` ]; then
    if [ `echo $config_path | grep 'result'` ]; then
      if ${run_background}; then
        CUDA_VISIBLE_DEVICES=${gpu_id} \
        nohup ${PYTHON} exp/training/train_hierarchical.py \
          --gpu ${gpu_id} \
          --saved_model_path $config_path \
          --data_save_path ${data} > log/$filename".log" &
      else
        CUDA_VISIBLE_DEVICES=${gpu_id} \
        nohup ${PYTHON} exp/training/train_hierarchical.py \
          --gpu ${gpu_id} \
          --saved_model_path $config_path \
          --data_save_path ${data} || exit 1;
      fi
    else
      if ${run_background}; then
        CUDA_VISIBLE_DEVICES=${gpu_id} \
        nohup ${PYTHON} exp/training/train_hierarchical.py \
          --gpu ${gpu_id} \
          --config_path $config_path \
          --model_save_path ${model} \
          --data_save_path ${data} > log/$filename".log" &
      else
        CUDA_VISIBLE_DEVICES=${gpu_id} \
        ${PYTHON} exp/training/train_hierarchical.py \
          --gpu ${gpu_id} \
          --config_path $config_path \
          --model_save_path ${model} \
          --data_save_path ${data} || exit 1;
      fi
    fi
  else
    if [ `echo $config_path | grep 'result'` ]; then
      if ${run_background}; then
        CUDA_VISIBLE_DEVICES=${gpu_id} \
        nohup ${PYTHON} exp/training/train.py \
          --gpu ${gpu_id} \
          --saved_model_path $config_path \
          --data_save_path ${data} > log/$filename".log" &
      else
        CUDA_VISIBLE_DEVICES=${gpu_id} \
        ${PYTHON} exp/training/train.py \
          --gpu ${gpu_id} \
          --saved_model_path $config_path \
          --data_save_path ${data} || exit 1;
      fi
    else
      if ${run_background}; then
        CUDA_VISIBLE_DEVICES=${gpu_id} \
        nohup ${PYTHON} exp/training/train.py \
          --gpu ${gpu_id} \
          --config_path $config_path \
          --model_save_path ${model} \
          --data_save_path ${data} > log/$filename".log" &
      else
        CUDA_VISIBLE_DEVICES=${gpu_id} \
        ${PYTHON} exp/training/train.py \
          --gpu ${gpu_id} \
          --config_path $config_path \
          --model_save_path ${model} \
          --data_save_path ${data}|| exit 1;
      fi
    fi
  fi

  echo "Finish model training ({stage}: 3)."
fi


echo "Done."


# echo ============================================================================
# echo "                             LM training                                 "
# echo ============================================================================
# if [ ${stage} -le 4 ]; then
#
#   echo "Finish LM training ({stage}: 4)."
# fi


# echo ============================================================================
# echo "                              Rescoring                                   "
# echo ============================================================================
# if [ ${stage} -le 5 ]; then
#
#   echo "Finish rescoring ({stage}: 5)."
# fi


# The following demonstrate how to re-segment long audios.
# local/run_segmentation_long_utts.sh

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
