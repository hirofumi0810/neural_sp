#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                                   WSJ                                     "
echo ============================================================================

stage=0
gpu=

### path to save dataset
export data=/n/sd8/inaguma/corpus/wsj

### vocabulary
# unit=wordpiece
# vocab_size=300
unit=char

# for wordpiece
wp_model_type=unigram  # or bpe

### path to save the model
model_dir=/n/sd8/inaguma/result/wsj

### path to the model directory to restart training
rnnlm_resume_model=
resume_model=

### path to original data
wsj0=/n/rd21/corpora_1/WSJ/wsj0
wsj1=/n/rd21/corpora_1/WSJ/wsj1

# Sometimes, we have seen WSJ distributions that do not have subdirectories
# like '11-13.1', but instead have 'doc', 'si_et_05', etc. directly under the
# wsj0 or wsj1 directories. In such cases, try the following:
CSTR_WSJTATATOP=/n/rd21/corpora_1/WSJ
# $CSTR_WSJTATATOP must contain a 'wsj0' and a 'wsj1' subdirectory for this to work.

directory_type=cstr # or original

### Select data size
data_size=si284
# data_size=si84

### configuration
rnnlm_config=conf/${unit}_lstm_rnnlm.yml
config=conf/attention/${unit}_blstm_att_${data_size}.yml

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

train_set=train_${data_size}
dev_set=test_dev93
test_set=test_eval92

if [ ${unit} = char ]; then
  vocab_size=
fi
if [ ${unit} != wordpiece ]; then
  wp_model_type=
fi


if [ ${stage} -le 0 ] && [ ! -e .done_stage_0 ]; then
  echo ============================================================================
  echo "                           Data Preparation                               "
  echo ============================================================================

  case ${directory_type} in
    original) local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.? || exit 1; ;;
    cstr) local/cstr_wsj_data_prep.sh $CSTR_WSJTATATOP || exit 1; ;;
  esac

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


if [ ${stage} -le 1 ] && [ ! -e .done_stage_1_${data_size} ]; then
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
  for x in ${train_set}; do
    dump_dir=${data}/dump/${x}; mkdir -p ${dump_dir}
    dump_feat.sh --cmd "$train_cmd" --nj 16 --add_deltadelta false \
      ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x} ${dump_dir} || exit 1;
  done
  for x in ${dev_set} ${test_set}; do
    dump_dir=${data}/dump/${x}_${data_size}; mkdir -p ${dump_dir}
    dump_feat.sh --cmd "$train_cmd" --nj 16 --add_deltadelta false \
      ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x}_${data_size} ${dump_dir} || exit 1;
  done

  touch .done_stage_1_${data_size} && echo "Finish feature extranction (stage: 1)."
fi


dict=${data}/dict/${train_set}_${unit}${wp_model_type}${vocab_size}.txt; mkdir -p ${data}/dict/
wp_model=${data}/dict/${train_set}_${wp_model_type}${vocab_size}
if [ ${stage} -le 2 ] && [ ! -e .done_stage_2_${unit}${wp_model_type}${vocab_size} ]; then
  echo ============================================================================
  echo "                      Dataset preparation (stage:2)                        "
  echo ============================================================================

  # Make a dictionary
  echo "<blank> 0" > ${dict}
  echo "<unk> 1" >> ${dict}
  echo "<sos> 2" >> ${dict}
  echo "<eos> 3" >> ${dict}
  echo "<pad> 4" >> ${dict}
  if [ ${unit} = char ]; then
    echo "<space> 5" >> ${dict}
  fi
  offset=`cat ${dict} | wc -l`
  echo "Making a dictionary..."
  if [ ${unit} = wordpiece ]; then
    cut -f 2- -d " " ${data}/${train_set}/text > ${data}/dict/input.txt
    spm_train --input=${data}/dict/input.txt --vocab_size=${vocab_size} --model_type=${wp_model_type} --model_prefix=${wp_model} --input_sentence_size=100000000
    spm_encode --model=${wp_model}.model --output_format=piece < ${data}/dict/input.txt | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset-1}' >> ${dict}
  else
    text2dict.py ${data}/${train_set}/text --unit ${unit} --vocab_size ${vocab_size} \
      --wp_model_type ${wp_model_type} --wp_model ${wp_model} | \
      sort | uniq | grep -v -e '^\s*$' | awk -v offset=${offset} '{print $0 " " NR+offset-1}' >> ${dict} || exit 1;
  fi
  echo "vocab size:" `cat ${dict} | wc -l`

  # Make datset csv files
  mkdir -p ${data}/dataset/
  for x in ${train_set}; do
    echo "Making a csv file for ${x}..."
    dump_dir=${data}/dump/${x}
    make_dataset_csv.sh --feat ${dump_dir}/feats.scp --unit ${unit} --wp_model ${wp_model} \
      ${data}/${x} ${dict} > ${data}/dataset/${x}_${unit}${wp_model_type}${vocab_size}.csv || exit 1;
  done
  for x in ${dev_set}; do
    dump_dir=${data}/dump/${x}_${data_size}
    make_dataset_csv.sh --feat ${dump_dir}/feats.scp --unit ${unit} \
      ${data}/${x} ${dict} > ${data}/dataset/${x}_${data_size}_${unit}${wp_model_type}${vocab_size}.csv || exit 1;
  done
  for x in ${test_set}; do
    dump_dir=${data}/dump/${x}_${data_size}
    make_dataset_csv.sh --is_test true --feat ${dump_dir}/feats.scp --unit ${unit} \
      ${data}/${x} ${dict} > ${data}/dataset/${x}_${data_size}_${unit}${wp_model_type}${vocab_size}.csv || exit 1;
  done

  touch .done_stage_2_${unit}${wp_model_type}${vocab_size} && echo "Finish creating dataset (stage: 2)."
fi


if [ ${stage} -le 3 ]; then
  echo ============================================================================
  echo "                      RNNLM Training stage (stage:3)                       "
  echo ============================================================================

  echo "Start RNNLM training..."

  # Extend dictionary for the external text data
  # ${data}/local/dict_nosp_larger/cleaned.gz

  # NOTE: support only a single GPU for RNNLM training
  # CUDA_VISIBLE_DEVICES=${rnnlm_gpu} ../../../src/bin/lm/train.py \
  #   --ngpus 1 \
  #   --train_set ${data}/dataset/${train_set}.csv \
  #   --dev_set ${data}/dataset/${dev_set}.csv \
  #   --eval_sets ${data}/dataset/eval1_${data_size}.csv \
  #   --config ${rnn_config} \
  #   --model ${model_dir} \
  #   --resume_model ${rnnlm_resume_model} || exit 1;
fi


if [ ${stage} -le 4 ]; then
  echo ============================================================================
  echo "                       ASR Training stage (stage:4)                        "
  echo ============================================================================

  echo "Start ASR training..."

  CUDA_VISIBLE_DEVICES=${gpu} ../../../neural_sp/bin/asr/train.py \
    --ngpus ${ngpus} \
    --train_set ${data}/dataset/${train_set}_${unit}${wp_model_type}${vocab_size}.csv \
    --dev_set ${data}/dataset/${dev_set}_${data_size}_${unit}${wp_model_type}${vocab_size}.csv \
    --eval_sets ${data}/dataset/${test_set}_${data_size}_${unit}${wp_model_type}${vocab_size}.csv \
    --dict ${dict} \
    --wp_model ${wp_model}.model \
    --config ${config} \
    --model ${model_dir}/asr \
    --label_type ${unit} || exit 1;
    # --resume_model ${resume_model} || exit 1;

  touch ${model}/.done_training && echo "Finish model training (stage: 4)."
fi
