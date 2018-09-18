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
# unit=char

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
config=conf/attention/${unit}_blstm_att_${data_size}.yml
# config=conf/ctc/${unit}_blstm_ctc_${data_size}.yml

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

if [ ${unit} = char ]; then
  vocab_size=
fi
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
wp_model=${data}/dict/${train_set}_${wp_model_type}${vocab_size}
if [ ${stage} -le 2 ] && [ ! -e .done_stage_2_${data_size}_${unit}${wp_model_type}${vocab_size} ]; then
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

  # Compute OOV rate
  if [ ${unit} = word ]; then
    mkdir -p ${data}/dict/word_count ${data}/dict/oov_rate
    echo "OOV rate:" > ${data}/dict/oov_rate/word_${vocab_size}.txt
    for x in ${train_set} dev ${test_set}; do
      cut -f 2- -d " " ${data}/${x}/text | tr " " "\n" | sort | uniq -c | sort -n -k1 -r \
        > ${data}/dict/word_count/${x}.txt || exit 1;
      compute_oov_rate.py ${data}/dict/word_count/${x}.txt ${dict} ${x} \
        >> ${data}/dict/oov_rate/word_${vocab_size}.txt || exit 1;
    done
    cat ${data}/dict/oov_rate/word_${vocab_size}.txt
  fi

  # Make datset csv files
  mkdir -p ${data}/dataset/
  for x in ${train_set}; do
    echo "Making a csv file for ${x}..."
    dump_dir=${data}/dump/${x}
    make_dataset_csv.sh --feat ${dump_dir}/feats.scp --unit ${unit} --wp_model ${wp_model} \
      ${data}/${x} ${dict} > ${data}/dataset/${x}_${unit}${wp_model_type}${vocab_size}.csv || exit 1;
  done
  for x in dev; do
    dump_dir=${data}/dump/${x}_${data_size}
    make_dataset_csv.sh --feat ${dump_dir}/feats.scp --unit ${unit} --wp_model ${wp_model} \
      ${data}/${x} ${dict} > ${data}/dataset/${x}_${data_size}_${unit}${wp_model_type}${vocab_size}.csv || exit 1;
  done
  for x in ${test_set}; do
    dump_dir=${data}/dump/${x}_${data_size}
    make_dataset_csv.sh --is_test true --feat ${dump_dir}/feats.scp --unit ${unit} \
      ${data}/${x} ${dict} > ${data}/dataset/${x}_${data_size}_${unit}${wp_model_type}${vocab_size}.csv || exit 1;
  done

  touch .done_stage_2_${data_size}_${unit}${wp_model_type}${vocab_size} && echo "Finish creating dataset (stage: 2)."
fi


mkdir -p ${model_dir}
if [ ${stage} -le 3 ]; then
  echo ============================================================================
  echo "                      RNNLM Training stage (stage:3)                       "
  echo ============================================================================

  echo "Start RNNLM training..."

  # NOTE: support only a single GPU for RNNLM training
  # CUDA_VISIBLE_DEVICES=${rnnlm_gpu} ../../../src/bin/lm/train.py \
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

  CUDA_VISIBLE_DEVICES=${gpu} ../../../neural_sp/bin/asr/train.py \
    --ngpus ${ngpus} \
    --train_set ${data}/dataset/${train_set}_${unit}${wp_model_type}${vocab_size}.csv \
    --dev_set ${data}/dataset/${dev_set}_${unit}${wp_model_type}${vocab_size}.csv \
    --dict ${dict} \
    --wp_model ${wp_model}.model \
    --config ${config} \
    --model ${model_dir}/asr \
    --label_type ${unit} || exit 1;
    # --resume_model ${resume_model} || exit 1;

  touch ${model}/.done_training && echo "Finish model training (stage: 4)."
fi



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
