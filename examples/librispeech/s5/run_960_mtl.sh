#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                              LibriSpeech                                 "
echo ============================================================================

stage=0
gpu=

### path to save dataset
export data=/n/sd8/inaguma/corpus/librispeech

### vocabulary
unit=word
vocab_size=30000
# unit=wordpiece
# vocab_size=5000
unit_sub=char

# for wordpiece
wp_model_type=unigram  # or bpe

### path to save the model
model_dir=/n/sd8/inaguma/result/librispeech

### path to the model directory to restart training
rnnlm_resume_model=
resume_model=

# path to download data
data_download_path=/n/rd21/corpora_7/librispeech/

### data size
data_size=960

### configuration
rnnlm_config=conf/${unit}_lstm_rnnlm.yml
config=conf/attention/${unit}_blstm_att_${data_size}_${unit_sub}_ctc.yml
# config=conf/ctc/${unit}_blstm_ctc_${data_size}_${unit_sub}_ctc.yml

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

if [ -z $gpu ]; then
  echo "Error: set GPU number." 1>&2
  echo "Usage: ./run.sh --gpu 0" 1>&2
  exit 1
fi
ngpus=`echo $gpu | tr "," "\n" | wc -l`
rnnlm_gpu=`echo $gpu | cut -d "," -f 1`

# Base url for downloads.
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11

train_set=train_${data_size}
dev_set=dev_${data_size}
test_set="dev_clean dev_other test_clean test_other"

if [ ${unit} != wordpiece ]; then
  wp_model_type=
fi


if [ ${stage} -le 0 ] && [ ! -e .done_stage_0_${data_size} ]; then
  echo ============================================================================
  echo "                       Data Preparation (stage:0)                          "
  echo ============================================================================

  # download data
  # mkdir -p ${data}
  # for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
  #   local/download_and_untar.sh ${data_download_path} ${data_url} ${part} || exit 1;
  # done

  # download the LM resources
  # local/download_lm.sh ${lm_url} ${data}/local/lm || exit 1;

  # format the data as Kaldi data directories
  for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    # use underscore-separated names in data directories.
    local/data_prep.sh ${data_download_path}/LibriSpeech/${part} ${data}/$(echo ${part} | sed s/-/_/g) || exit 1;
  done

  # lowercasing
  for x in dev_clean test_clean dev_other test_other train_clean_100 train_clean_360 train_other_500; do
      cp ${data}/${x}/text ${data}/${x}/text.tmp
      paste -d "" <(cut -f 1 -d" " ${data}/${x}/text.tmp) \
                  <(awk '{$1=""; print tolower($0)}' ${data}/${x}/text.tmp) > ${data}/${x}/text
      rm ${data}/${x}/text.tmp
  done

  touch .done_stage_0_${data_size} && echo "Finish data preparation (stage: 0)."
fi


if [ ${stage} -le 1 ] && [ ! -e .done_stage_1_${data_size} ]; then
  echo ============================================================================
  echo "                    Feature extranction (stage:1)                          "
  echo ============================================================================

  for x in dev_clean test_clean dev_other test_other train_clean_100 train_clean_360 train_other_500; do
    steps/make_fbank.sh --nj 16 --cmd "$train_cmd" --write_utt2num_frames true \
      ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1;
  done

  utils/combine_data.sh --extra_files "utt2num_frames" ${data}/${train_set} ${data}/train_clean_100 ${data}/train_clean_360 ${data}/train_other_500 || exit 1;
  utils/combine_data.sh --extra_files "utt2num_frames" ${data}/dev ${data}/dev_clean ${data}/dev_other || exit 1;

  # Compute global CMVN
  compute-cmvn-stats scp:${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark || exit 1;

  # Apply global CMVN & dump features
  for x in ${train_set}; do
    dump_dir=${data}/dump/${x}; mkdir -p ${dump_dir}
    dump_feat.sh --cmd "$train_cmd" --nj 16 --add_deltadelta false \
      ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x} ${dump_dir} || exit 1;
  done
  for x in dev ${test_set}; do
    dump_dir=${data}/dump/${x}_${data_size}; mkdir -p ${dump_dir}
    dump_feat.sh --cmd "$train_cmd" --nj 16 --add_deltadelta false \
      ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x}_${data_size} ${dump_dir} || exit 1;
  done

  touch .done_stage_1_${data_size} && echo "Finish feature extranction (stage: 1)."
fi


dict=${data}/dict/${train_set}_${unit}${wp_model_type}${vocab_size}.txt; mkdir -p ${data}/dict/
dict_sub=${data}/dict/${train_set}_${unit_sub}.txt
wp_model=${data}/dict/${train_set}_${wp_model_type}${vocab_size}

if [ ! -f ${dict} ]; then
  echo "There is no file such as "${dict}
  exit 1
fi

if [ ! -f ${dict_sub} ]; then
  echo "There is no file such as "${dict_sub}
  exit 1
fi


mkdir -p ${model_dir}
if [ ${stage} -le 3 ]; then
  echo ============================================================================
  echo "                      RNNLM Training stage (stage:3)                       "
  echo ============================================================================

  echo "Start RNNLM training..."

  # NOTE: support only a single GPU for RNNLM training
  CUDA_VISIBLE_DEVICES=${rnnlm_gpu} ../../../neural_sp/bin/lm/train.py \
    --ngpus 1 \
    --train_set ${data}/dataset/${train_set}_${unit}${wp_model_type}${vocab_size}.csv \
    --dev_set ${data}/dataset/${dev_set}_${unit}${wp_model_type}${vocab_size}.csv \
    --dict ${dict} \
    --wp_model ${wp_model}.model \
    --config ${rnnlm_config} \
    --model ${model_dir}/rnnlm \
    --label_type ${unit} || exit 1;
    # --resume_model ${resume_model} || exit 1;

   echo "Finish RNNLM training (stage: 3)."
fi


if [ ${stage} -le 4 ]; then
  echo ============================================================================
  echo "                       ASR Training stage (stage:4)                        "
  echo ============================================================================

  echo "Start ASR training..."

  CUDA_VISIBLE_DEVICES=${gpu} ../../../neural_sp/bin/asr/train.py \
    --ngpus ${ngpus} \
    --train_set ${data}/dataset/${train_set}_${unit}${wp_model_type}${vocab_size}.csv \
    --train_set_sub ${data}/dataset/${train_set}_${unit_sub}.csv \
    --dev_set ${data}/dataset/${dev_set}_${unit}${wp_model_type}${vocab_size}.csv \
    --dev_set_sub ${data}/dataset/${dev_set}_${unit_sub}.csv \
    --dict ${dict} \
    --dict_sub ${dict_sub} \
    --wp_model ${wp_model}.model \
    --config ${config} \
    --model ${model_dir}/asr \
    --label_type ${unit} \
    --label_type_sub ${unit_sub} || exit 1;
    # --resume_model ${resume_model} || exit 1;

  touch ${model}/.done_training && echo "Finish model training (stage: 4)."
fi
