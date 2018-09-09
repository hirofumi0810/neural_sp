#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                                   CSJ                                     "
echo ============================================================================

if [ $# -lt 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./run.sh path_to_config_file (or path_to_saved_model) gpu_id1 gpu_id2... (arbitrary number)" 1>&2
  exit 1
fi

ngpus=`expr $# - 1`
config=$1
gpu_ids=$2

if [ $# -gt 2 ]; then
  rest_ngpus=`expr $ngpus - 1`
  for i in `seq 1 $rest_ngpus`
  do
    gpu_ids=$gpu_ids","${3}
    shift
  done
fi

stage=0

### vocabulary
vocab_size=15000

### feature extranction
pitch=0  ## or 1

### Set path to original data
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

### Select data size
export datasize=aps_other
# export datasize=aps
# export datasize=sps  # TODO
# export datasize=all_except_dialog
# export datasize=all
# NOTE: aps_other=default using "Academic lecture" and "other" data,
#       aps=using "Academic lecture" data,
#       sps=using "Academic lecture" data,
#       all_except_dialog=using All data except for "dialog" data,
#       all=using All data

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

train_set=train_${datasize}
dev_set=dev_${datasize}
test_set="eval1 eval2 eval3"


if [ $stage -le 0 ] && [ ! -e .done_stage_0_${datasize} ]; then
  echo ============================================================================
  echo "                       Data Preparation (stage:0)                          "
  echo ============================================================================

  local/csj_make_trans/csj_autorun.sh ${CSJDATATOP} ${data}/csj-data ${CSJVER} || exit 1;
  local/csj_data_prep.sh ${data}/csj-data ${datasize} || exit 1;

  # Data preparation and formatting for evaluation set.
  # CSJ has 3 types of evaluation data
  for x in eval1 eval2 eval3; do
    local/csj_eval_data_prep.sh ${data}/csj-data/eval ${x} || exit 1;
  done

  touch .done_stage_0_${datasize} && echo "Finish data preparation (stage: 0)."\n
fi


if [ $stage -le 1 ] && [ ! -e .done_stage_1_${datasize} ]; then
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
  compute-cmvn-stats scp:${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark

  # Apply global CMVN & dump features
  for x in ${train_set} ${dev_set}; do
    dump_dir=${data}/feat/${x}; mkdir -p ${dump_dir}
    dump_feat.sh --cmd "$train_cmd" --nj 16 --do_delta false \
      ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x} ${dump_dir}
  done
  for x in ${test_set}; do
    dump_dir=${data}/feat/${x}_${datasize}; mkdir -p ${dump_dir}
    dump_feat.sh --cmd "$train_cmd" --nj 16 --do_delta false \
      ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x}_${datasize} ${dump_dir}
  done

  touch .done_stage_1_${datasize} && echo "Finish feature extranction (stage: 1)."\n
fi

# word-level dictionary
dict=${data}/dict/${train_set}_word_${vocab_size}.txt; mkdir -p ${data}/dict/
if [ $stage -le 2 ] && [ ! -e .done_stage_2_${datasize} ]; then
  echo ============================================================================
  echo "                      Dataset preparation (stage:2)                        "
  echo ============================================================================

  # make a dictionary
  echo "<blank> 0" > ${dict}
  echo "<unk> 1" >> ${dict}
  offset=`cat ${dict} | wc -l`
  remove_tokens="<sp>"
  echo "Making a dictionary..."
  text2dict.py ${data}/${train_set}/text --remove_tokens ${remove_tokens} --unit word --vocab_size ${vocab_size} --word_boundary | \
    sort | uniq | grep -v -e '^\s*$' | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}
  echo "vocab size:" `cat ${dict} | wc -l`

  exit 1

  # make csv datset
  # cs: utt_id, feat_path, xlen, text, tokenid, ylen
  for x in ${train_set} ${dev_set}; do
    echo "Making a csv file for ${x}..."
    dump_dir=${data}/feat/${x}
    make_dataset_csv.sh --feat ${dump_dir}/feats.scp \
      ${data}/${x} ${dict} > ${data}/dataset/${x}/dataset.csv
  done
  for x in ${test_set}; do
    dump_dir=${data}/feat/${x}_${datasize}
    make_dataset_csv.sh --feat ${dump_dir}/feats.scp \
      ${data}/${x} ${dict} > ${data}/dataset/${x}_${datasize}/dataset.csv
  done

  touch .done_stage_2_${datasize} && echo "Finish creating dataset (stage: 2)."\n
fi


if [ $stage -le 3 ]; then
  echo ============================================================================
  echo "                      RNNLM Training stage (stage:4)                       "
  echo ============================================================================
  echo "Start RNNLM training..."

fi


if [ $stage -le 4 ]; then
  echo ============================================================================
  echo "                       ASR Training stage (stage:4)                        "
  echo ============================================================================

  filename=$(basename ${config} | awk -F. '{print $1}')
  echo "Start ASR training..."

  if [ `echo ${config} | grep 'result'` ]; then
    CUDA_VISIBLE_DEVICES=${gpu_ids} ${PYTHON} neural_sp/bin/training/train.py \
      --corpus ${corpus} \
      --ngpus ${ngpus} \
      --train_set train \
      --dev_set dev \
      --eval_sets eval1 \
      --saved_model_path ${config} \
      --data_save_path ${data} || exit 1;
  else
    CUDA_VISIBLE_DEVICES=${gpu_ids} ${PYTHON} neural_sp/bin/training/train.py \
      --corpus ${corpus} \
      --ngpus ${ngpus} \
      --train_set train \
      --dev_set dev \
      --eval_sets eval1 \
      --config ${config} \
      --model_save_path ${model} \
      --data_save_path ${data} || exit 1;
  fi

  echo "Finish model training (stage: 4)."\n
fi
