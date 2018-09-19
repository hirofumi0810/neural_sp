#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                                IWSLT18                                   "
echo ============================================================================

stage=0
gpu=

### path to save dataset
export data=/n/sd8/inaguma/corpus/iwslt18

### vocabulary
unit=wordpiece
vocab_size=5000
# unit=char

# for wordpiece
wp_model_type=unigram  # or bpe

### path to save the model
model_dir=/n/sd8/inaguma/result/iwslt18

### path to the model directory to restart training
rnnlm_resume_model=
resume_model=

### path to download data
datadir=/n/sd8/inaguma/corpus/iwslt18/data

### configuration
rnnlm_config=conf/${unit}_lstm_rnnlm.yml
config=conf/st/${unit}_blstm_att_asr.yml
# config=conf/st/${unit}_blstm_att_mt.yml
# config=conf/st/${unit}_blstm_att_asr_mt.yml

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

train_set=train.de
dev_set=dev.de
recog_set="dev2010.de tst2010.de tst2013.de tst2014.de tst2015.de"

if [ ${unit} = char ]; then
  vocab_size=
fi
if [ ${unit} != wordpiece ]; then
  wp_model_type=
fi


if [ ${stage} -le 0 ] && [ ! -e .done_stage_0 ]; then
  echo ============================================================================
  echo "                       Data Preparation (stage:0)                          "
  echo ============================================================================

  # download data
  # for part in train dev2010 tst2010 tst2013 tst2014 tst2015 tst2018; do
  #     local/download_and_untar.sh ${datadir} ${part}
  # done

  local/data_prep_train.sh ${datadir}
  for part in dev2010 tst2010 tst2013 tst2014 tst2015 tst2018; do
    local/data_prep_eval.sh ${datadir} ${part}
  done

  touch .done_stage_0 && echo "Finish data preparation (stage: 0)."
fi


if [ ${stage} -le 1 ] && [ ! -e .done_stage_1.de ]; then
  echo ============================================================================
  echo "                    Feature extranction (stage:1)                          "
  echo ============================================================================

  # for x in train_org dev2010 tst2010 tst2013 tst2014 tst2015 tst2018; do
  for x in train_org; do
    steps/make_fbank.sh --nj 16 --cmd "$train_cmd" --write_utt2num_frames true \
      ${data}/${x}.de ${data}/log/make_fbank/${x} ${data}/fbank || exit 1;
  done

  # make a dev set
  for lang in de en; do
    utils/subset_data_dir.sh --first ${data}/train_org.${lang} 4000 ${data}/dev.${lang}
    n=$[`cat ${data}/train_org.${lang}/segments | wc -l` - 4000]
    utils/subset_data_dir.sh --last ${data}/train_org.${lang} ${n} ${data}/train.${lang}
  done

  # compute global CMVN
  compute-cmvn-stats scp:${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark

  # Apply global CMVN & dump features
  for x in ${train_set} ${dev_set}; do
    dump_dir=${data}/dump/${x}; mkdir -p ${dump_dir}
    dump_feat.sh --cmd "$train_cmd" --nj 16 --add_deltadelta false \
      ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x} ${dump_dir} || exit 1;
  done
  # for x in ${test_set}; do
  #   dump_dir=${data}/dump/${x}; mkdir -p ${dump_dir}
  #   dump_feat.sh --cmd "$train_cmd" --nj 16 --add_deltadelta false \
  #     ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x} ${dump_dir} || exit 1;
  # done

  touch .done_stage_1.de && echo "Finish feature extranction (stage: 1)."
fi


dict=${data}/dict/train_${unit}${wp_model_type}${vocab_size}.txt; mkdir -p ${data}/dict/
nlsyms=${data}/dict/non_linguistic_symbols.txt
wp_model=${data}/dict/train_${wp_model_type}${vocab_size}

if [ ! -f ${dict} ]; then
  echo "There is no file such as "${dict}
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
  echo "                       ST Training stage (stage:4)                        "
  echo ============================================================================

  echo "Start ST training..."

  CUDA_VISIBLE_DEVICES=${gpu} ../../../neural_sp/bin/asr/train.py \
    --ngpus ${ngpus} \
    --train_set ${data}/dataset/${train_set}_${unit}${wp_model_type}${vocab_size}.csv \
    --train_set_sub ${data}/dataset/train.en_${unit}${wp_model_type}${vocab_size}.csv \
    --dev_set ${data}/dataset/${dev_set}_${unit}${wp_model_type}${vocab_size}.csv \
    --dev_set_sub ${data}/dataset/dev.en_${unit}${wp_model_type}${vocab_size}.csv \
    --dict ${dict} \
    --dict_sub ${dict} \
    --wp_model ${wp_model}.model \
    --config ${config} \
    --model ${model_dir}/st \
    --label_type ${unit} \
    --metric loss || exit 1;
    # --resume_model ${resume_model} || exit 1;

  touch ${model}/.done_training && echo "Finish ST training (stage: 4)."
fi
