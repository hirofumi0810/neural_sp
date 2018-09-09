#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                                   WSJ                                     "
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

### Set path to save dataset
export data="/n/sd8/inaguma/corpus/wsj"

### configuration
rnnlm_config="conf/word_lstm_rnnlm_all.yml"
asr_config="conf/single_task/word/word_blstm_att_aps_other.yml"

### Set path to save the model
model_dir="/n/sd8/inaguma/result/wsj"

### Restart training (path to the saved model directory)
rnnlm_saved_model=
asr_saved_model=


### Set path to original data
wsj0="/n/rd21/corpora_1/WSJ/wsj0"
wsj1="/n/rd21/corpora_1/WSJ/wsj1"

# Sometimes, we have seen WSJ distributions that do not have subdirectories
# like '11-13.1', but instead have 'doc', 'si_et_05', etc. directly under the
# wsj0 or wsj1 directories. In such cases, try the following:
CSTR_WSJTATATOP="/n/rd21/corpora_1/WSJ"
# $CSTR_WSJTATATOP must contain a 'wsj0' and a 'wsj1' subdirectory for this to work.

directory_type=cstr # or original

### Select data size
export data_size=si284
# export data_size=si84

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

train_set=train_${data_size}
dev_set=test_dev93
test_set=test_eval92


if [ ${stage} -le 0 ] && [ ! -e .done_stage_0 ]; then
  echo ============================================================================
  echo "                           Data Preparation                               "
  echo ============================================================================

  case ${directory_type} in
    original) local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.? || exit 1; ;;
    cstr) local/cstr_wsj_data_prep.sh $CSTR_WSJTATATOP || exit 1; ;;
  esac
  # rm ${data}/local/dict/lexiconp.txt

  # "nosp" refers to the dictionary before silence probabilities and pronunciation
  # probabilities are added.
  local/wsj_prepare_dict.sh --dict-suffix "_nosp" || exit 1;
  local/wsj_format_data.sh --lang-suffix "_nosp" || exit 1;

  # NOTE: If you have a setup corresponding to the older cstr_wsj_data_prep.sh style,
  # use local/cstr_wsj_extend_dict.sh --dict-suffix "_nosp" $CSTR_WSJTATATOP/wsj1/doc/ instead.
  case ${directory_type} in
    original) local/wsj_extend_dict.sh --dict-suffix "_nosp" ${wsj1}/13-32.1 || exit 1; ;;
    cstr) local/cstr_wsj_extend_dict.sh --dict-suffix "_nosp" $CSTR_WSJTATATOP/wsj1/doc/ || exit 1; ;;
  esac

  # Make a subset
  utils/subset_data_dir.sh --first ${data}/train_si284 7138 ${data}/train_si84 || exit 1;

  touch .done_stage_0 && echo "Finish data preparation (stage: 0)."
fi


if [ ${stage} -le 1 ] && [ ! -e .done_stage_1 ]; then
  echo ============================================================================
  echo "                    Feature extranction (stage:1)                          "
  echo ============================================================================

  for x in train_si84 train_si284 test_dev93 test_eval92; do
    steps/make_fbank.sh --nj 16 --cmd "$train_cmd" --write_utt2num_frames true \
      ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1;
  done

  # Compute global CMVN
  compute-cmvn-stats scp:${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark || exit 1;

  # Apply global CMVN & dump features
  for x in ${train_set} ${dev_set}; do
    dump_dir=${data}/feat/${x}; mkdir -p ${dump_dir}
    dump_feat.sh --cmd "$train_cmd" --nj 16 --add_deltadelta false \
      ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x} ${dump_dir} || exit 1;
  done
  for x in ${test_set}; do
    dump_dir=${data}/feat/${x}_${data_size}; mkdir -p ${dump_dir}
    dump_feat.sh --cmd "$train_cmd" --nj 16 --add_deltadelta false \
      ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x}_${data_size} ${dump_dir} || exit 1;
  done

  touch .done_stage_1 && echo "Finish feature extranction (stage: 1)."
fi

dict=${data}/dict/${train_set}_char.txt; mkdir -p ${data}/dict/
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
  text2dict.py ${data}/${train_set}/text --unit word --vocab_size ${vocab_size} --word_boundary | \
    sort | uniq | grep -v -e '^\s*$' | awk -v offset=${offset} '{print $0 " " NR+offset-1}' >> ${dict} || exit 1;
  echo "vocab size:" `cat ${dict} | wc -l`

  # Extend dictionary for the external text data
  # ${data}/local/dict_nosp_larger/cleaned.gz

  # Make datset csv files
  # csv: utt_id, feat_path, x_len, x_dim, text, tokenid, y_len, y_dim
  for x in ${train_set} ${dev_set}; do
    echo "Making a csv file for ${x}..."
    dump_dir=${data}/feat/${x}
    mkdir -p ${data}/dataset/
    make_dataset_csv.sh --feat ${dump_dir}/feats.scp --unit word --word_boundary true \
      ${data}/dataset/${x}.csv ${data}/${x} ${dict} || exit 1;
  done
  for x in ${test_set}; do
    dump_dir=${data}/feat/${x}_${data_size}
    make_dataset_csv.sh --is_test true --feat ${dump_dir}/feats.scp --unit word --word_boundary true \
      ${data}/dataset/${x}_${data_size}.csv ${data}/${x} ${dict} || exit 1;
  done

  touch .done_stage_2 && echo "Finish creating dataset (stage: 2)."
fi


# if [ ${stage} -le 3 ]; then
#   echo ============================================================================
#   echo "                      RNNLM Training stage (stage:3)                       "
#   echo ============================================================================
#   echo "Start RNNLM training..."
#
#   # NOTE: support only a single GPU for RNNLM training
#   CUDA_VISIBLE_DEVICES=${gpu_ids} ../../../src/bin/lm/train.py \
#     --corpus wsj \
#     --ngpus 1 \
#     --train_set ${data}/dataset/${train_set}.csv \
#     --dev_set ${data}/dataset/${dev_set}.csv \
#     --eval_sets ${data}/dataset/eval1_${data_size}.csv \
#     --config ${rnn_config} \
#     --model ${model_dir} \
#     --saved_model ${rnnlm_saved_model} || exit 1;
# fi


if [ ${stage} -le 4 ]; then
  echo ============================================================================
  echo "                       ASR Training stage (stage:4)                        "
  echo ============================================================================

  mkdir -p ${model_dir}
  echo "Start ASR training..."

  CUDA_VISIBLE_DEVICES=${gpu_ids} ../../../neural_sp/bin/asr/train.py \
    --corpus wsj \
    --ngpus ${ngpus} \
    --train_set ${data}/dataset/${train_set}.csv \
    --dev_set ${data}/dataset/${dev_set}.csv \
    --eval_sets ${data}/dataset/${test_set}.csv \
    --dict ${dict} \
    --config ${asr_config} \
    --model ${model_dir} \
    --label_type word || exit 1;
    # --saved_model ${asr_saved_model} || exit 1;
    # TODO(hirofumi): send a e-mail

    touch ${model}/.done_training && echo "Finish model training (stage: 4)."
  fi
