#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                                   CSJ                                     "
echo ============================================================================

if [ $# -lt 1 ]; then
  echo "Error: set GPU number." 1>&2
  echo "Usage: ./run.sh gpu_id1 gpu_id2... (arbitrary number)" 1>&2
  exit 1
fi

ngpus=`expr $#`
gpu_ids=$1

if [ $# -gt 2 ]; then
  rest_ngpus=`expr $ngpus - 1`
  for i in `seq 1 $rest_ngpus`
  do
    gpu_ids=$gpu_ids","${3}
    shift
  done
fi

stage=0

### path to save dataset
export data=/n/sd8/inaguma/corpus/csj

### vocabulary
unit=word
vocab_size=30000
# unit=wordpiece
# vocab_size=5000
unit_sub=char

# for wordpiece
wp_model_type=unigram  # or bpe

### feature extranction
pitch=0  ## or 1

### path to save the model
model_dir=/n/sd8/inaguma/result/csj

### path to the model directory to restart training
rnnlm_resume_model=
asr_resume_model=

### path to original data
CSJDATATOP=/n/rd25/mimura/corpus/CSJ  ## CSJ database top directory.
CSJVER=dvd  ## Set your CSJ format (dvd or usb).
            ## Usage    :
            ## Case DVD : We assume CSJ DVDs are copied in this directory with the names dvd1, dvd2,...,dvd17.
            ##            Neccesary directory is dvd3 - dvd17.
            ##            e.g. $ ls ${CSJDATATOP}(DVD) => 00README.txt dvd1 dvd2 ... dvd17
            ##
            ## Case USB : Neccesary directory is MORPH/SDB and WAV
            ##            e.g. $ ls ${CSJDATATOP}(USB) => 00README.txt DOC MORPH ... WAV fileList.csv
            ## Case merl :MERL setup. Neccesary directory is WAV and sdb

### data size
# export data_size=aps_other
# export data_size=aps
# export data_size=sps
# export data_size=all_except_dialog
export data_size=all
# NOTE: aps_other=default using "Academic lecture" and "other" data,
#       aps=using "Academic lecture" data,
#       sps=using "Academic lecture" data,
#       all_except_dialog=using All data except for "dialog" data,
#       all=using All data

### configuration
rnnlm_config=conf/rnnlm/${unit}_lstm_rnnlm_all.yml
asr_config=conf/attention/${unit}_blstm_att_${data_size}_${unit_sub}_ctc.yml
# asr_config=conf/ctc/${unit}_blstm_ctc_${data_size}_${unit_sub}_ctc.yml

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

train_set=train_${data_size}
dev_set=dev_${data_size}
test_set="eval1 eval2 eval3"

if [ ${unit} != wordpiece ]; then
  wp_model_type=
fi


if [ ${stage} -le 0 ] && [ ! -e .done_stage_0_${data_size} ]; then
  echo ============================================================================
  echo "                       Data Preparation (stage:0)                          "
  echo ============================================================================

  mkdir -p ${data}
  local/csj_make_trans/csj_autorun.sh ${CSJDATATOP} ${data}/csj-data ${CSJVER} || exit 1;
  local/csj_data_prep.sh ${data}/csj-data ${data_size} || exit 1;
  for x in eval1 eval2 eval3; do
    local/csj_eval_data_prep.sh ${data}/csj-data/eval ${x} || exit 1;
  done

  touch .done_stage_0_${data_size} && echo "Finish data preparation (stage: 0)."
fi

if [ ${stage} -le 1 ] && [ ! -e .done_stage_1_${data_size} ]; then
  echo ============================================================================
  echo "                    Feature extranction (stage:1)                          "
  echo ============================================================================

  for x in ${train_set} ${test_set}; do
    case $pitch in
      1)  steps/make_fbank_pitch.sh --nj 16 --cmd "$train_cmd" --write_utt2num_frames true \
            ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1; ;;
      0)  steps/make_fbank.sh --nj 16 --cmd "$train_cmd" --write_utt2num_frames true \
            ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1; ;;
      *) echo "Set pitch to 0 or 1." && exit 1; ;;
    esac
  done

  # Use the first 4k sentences from training data as dev set. (39 speakers.)
  utils/subset_data_dir.sh --first ${data}/${train_set} 4000 ${data}/${dev_set} || exit 1; # 6hr 31min
  n=$[`cat ${data}/${train_set}/segments | wc -l` - 4000]
  utils/subset_data_dir.sh --last ${data}/${train_set} ${n} ${data}/${train_set}.tmp || exit 1;

  # Finally, the full training set:
  utils/data/remove_dup_utts.sh 300 ${data}/${train_set}.tmp ${data}/${train_set} || exit 1;  # 233hr 36min
  rm -rf ${data}/*.tmp

  # Compute global CMVN
  compute-cmvn-stats scp:${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark || exit 1;

  # Apply global CMVN & dump features
  for x in ${train_set} ${dev_set}; do
    dump_dir=${data}/dump/${x}; mkdir -p ${dump_dir}
    dump_feat.sh --cmd "$train_cmd" --nj 16 --add_deltadelta false \
      ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x} ${dump_dir} || exit 1;
  done
  for x in ${test_set}; do
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
  # CUDA_VISIBLE_DEVICES=${gpu_ids} ../../../src/bin/lm/train.py \
  #   --ngpus 1 \
  #   --train_set ${data}/dataset/${train_set}.csv \
  #   --dev_set ${data}/dataset/${dev_set}.csv \
  #   --eval_sets ${data}/dataset/eval1_${data_size}_${unit}${wp_model_type}${vocab_size}.csv \
  #   --config ${rnn_config} \
  #   --model ${model_dir} \
  #   --resume_model ${rnnlm_resume_model} || exit 1;
fi


if [ ${stage} -le 4 ]; then
  echo ============================================================================
  echo "                       ASR Training stage (stage:4)                        "
  echo ============================================================================

  echo "Start ASR training..."

  # export CUDA_LAUNCH_BLOCKING=1
  CUDA_VISIBLE_DEVICES=${gpu_ids} ../../../neural_sp/bin/asr/train.py \
    --ngpus ${ngpus} \
    --train_set ${data}/dataset/${train_set}_${unit}${wp_model_type}${vocab_size}.csv \
    --train_set_sub ${data}/dataset/${train_set}_${unit_sub}.csv \
    --dev_set ${data}/dataset/${dev_set}_${unit}${wp_model_type}${vocab_size}.csv \
    --dev_set_sub ${data}/dataset/${dev_set}_${unit_sub}.csv \
    --eval_sets ${data}/dataset/eval1_${data_size}_${unit_sub}.csv \
    --dict ${dict} \
    --dict_sub ${dict_sub} \
    --wp_model ${wp_model}.model \
    --config ${asr_config} \
    --model ${model_dir}/asr \
    --label_type ${unit} \
    --label_type_sub ${unit_sub} || exit 1;
    # --resume_model ${asr_resume_model} || exit 1;
    # TODO(hirofumi): send a e-mail

  touch ${model}/.done_training && echo "Finish model training (stage: 4)."
fi
