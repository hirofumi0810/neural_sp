#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                                  TIMIT                                    "
echo ============================================================================

stage=0
gpu=

### Set path to save dataset
export data=/n/sd8/inaguma/corpus/timit

### configuration
config=conf/attention/bgru_att_phone61.yml
# config=conf/ctc/blstm_ctc_phone61.yml

### Set path to save the model
model_dir=/n/sd8/inaguma/result/timit

### Restart training (path to the saved model directory)
resume_model=

### Set path to original data
TIMITDATATOP=/n/rd21/corpora_1/TIMIT

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

train_set=train
dev_set=dev
test_set=test


if [ ${stage} -le 0 ] && [ ! -e .done_stage_0 ]; then
  echo ============================================================================
  echo "                       Data Preparation (stage:0)                          "
  echo ============================================================================

  mkdir -p ${data}
  local/timit_data_prep.sh ${TIMITDATATOP} || exit 1;
  local/timit_format_data.sh || exit 1;

  touch .done_stage_0 && echo "Finish data preparation (stage: 0)."
fi


if [ ${stage} -le 1 ] && [ ! -e .done_stage_1 ]; then
  echo ============================================================================
  echo "                    Feature extranction (stage:1)                          "
  echo ============================================================================

  for x in train dev test; do
    steps/make_fbank.sh --nj 16 --cmd "$train_cmd" --write_utt2num_frames true \
      ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1;
  done

  # Compute global CMVN
  compute-cmvn-stats scp:${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark || exit 1;

  # Apply global CMVN & dump features
  for x in ${train_set} ${dev_set} ${test_set}; do
    dump_dir=${data}/feat/${x}; mkdir -p ${dump_dir}
    dump_feat.sh --cmd "$train_cmd" --nj 16 --add_deltadelta true \
      ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x} ${dump_dir} || exit 1;
  done

  touch .done_stage_1 && echo "Finish feature extranction (stage: 1)."
fi


dict=${data}/dict/${train_set}.txt; mkdir -p ${data}/dict/
if [ ${stage} -le 2 ] && [ ! -e .done_stage_2 ]; then
  echo ============================================================================
  echo "                      Dataset preparation (stage:2)                        "
  echo ============================================================================

  # Make a dictionary
  echo "<blank> 0" > ${dict}
  echo "<unk> 1" >> ${dict}
  echo "<sos> 2" >> ${dict}
  echo "<eos> 3" >> ${dict}
  echo "<pad> 4" >> ${dict}
  offset=`cat ${dict} | wc -l`
  echo "Making a dictionary..."
  text2dict.py ${data}/${train_set}/text --unit phone | \
    sort | uniq | grep -v -e '^\s*$' | awk -v offset=${offset} '{print $0 " " NR+offset-1}' >> ${dict} || exit 1;
  echo "vocab size:" `cat ${dict} | wc -l`

  # Make datset csv files
  mkdir -p ${data}/dataset/
  for x in ${train_set} ${dev_set}; do
    echo "Making a csv file for ${x}..."
    dump_dir=${data}/feat/${x}
    make_dataset_csv.sh --feat ${dump_dir}/feats.scp --unit phone \
      ${data}/${x} ${dict} > ${data}/dataset/${x}.csv || exit 1;
  done
  for x in ${test_set}; do
    dump_dir=${data}/feat/${x}
    make_dataset_csv.sh --is_test true --feat ${dump_dir}/feats.scp --unit phone \
      ${data}/${x} ${dict} > ${data}/dataset/${x}.csv || exit 1;
  done

  touch .done_stage_2 && echo "Finish creating dataset (stage: 2)."
fi

# NOTE: skip RNNLM training (stage:3)

mkdir -p ${model_dir}
if [ ${stage} -le 4 ]; then
  echo ============================================================================
  echo "                       ASR Training stage (stage:4)                        "
  echo ============================================================================

  echo "Start ASR training..."

  CUDA_VISIBLE_DEVICES=${gpu} ../../../neural_sp/bin/asr/train.py \
    --ngpus ${ngpus} \
    --train_set ${data}/dataset/${train_set}.csv \
    --dev_set ${data}/dataset/${dev_set}.csv \
    --eval_sets ${data}/dataset/${test_set}.csv \
    --dict ${dict} \
    --config ${config} \
    --model ${model_dir}/asr \
    --label_type phone || exit 1;
    # --resume_model ${resume_model} || exit 1;

  touch ${model}/.done_training && echo "Finish model training (stage: 4)."
fi


echo "Done."
