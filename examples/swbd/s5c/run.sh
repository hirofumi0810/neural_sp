#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                              Switchboard                                 "
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
export data=/n/sd8/inaguma/corpus/swbd

### vocabulary
unit=word
vocab_size=15000
# unit=wordpiece
# vocab_size=1000
# unit=char

# for wordpiece
wp_model_type=unigram  # or bpe

### path to save the model
model_dir=/n/sd8/inaguma/result/swbd

### path to the model directory to restart training
rnnlm_saved_model=
asr_saved_model=

### path to original data
SWBD_AUDIOPATH=/n/rd21/corpora_7/swb
EVAL2000_AUDIOPATH=/n/rd21/corpora_7/hub5_english/LDC2002S09
EVAL2000_TRANSPATH=/n/rd21/corpora_7/hub5_english/LDC2002T43
RT03_PATH=
# FISHER_PATH=/n/rd7/fisher_english
FISHER_PATH=

### configuration
rnnlm_config=conf/${unit}_lstm_rnnlm.yml
asr_config=conf/attention/${unit}_blstm_att.yml
# asr_config=conf/attention/${unit}_vggblstm_att.yml
# asr_config=conf/attention/${unit}_vggblstm_att_large.yml
# asr_config=conf/ctc/${unit}_blstm_ctc.yml

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

train_set=train
dev_set=dev
test_set=eval2000

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

  local/swbd1_data_download.sh ${SWBD_AUDIOPATH} || exit 1;
  local/swbd1_prepare_dict.sh || exit 1;
  local/swbd1_data_prep.sh ${SWBD_AUDIOPATH} || exit 1;
  local/eval2000_data_prep.sh ${EVAL2000_AUDIOPATH} ${EVAL2000_TRANSPATH} || exit 1;

  # if [ -d ${RT03_PATH} ]; then
  #   local/rt03_data_prep.sh ${RT03_PATH}
  # fi

  # prepare fisher data for language models (optional)
  # if [ -d ${FISHER_PATH} ]; then
  #   # prepare fisher data and put it under data/train_fisher
  #   local/fisher_data_prep.sh ${FISHER_PATH}
  #   local/fisher_swbd_prepare_dict.sh
  #
  #   # merge two datasets into one
  #   mkdir -p ${data}/train_swbd_fisher
  #   for f in spk2utt utt2spk wav.scp text segments; do
  #     cat ${data}/train_fisher/$f ${data}/train_swbd/$f > ${data}/train_swbd_fisher/$f
  #   done
  # fi

  touch .done_stage_0 && echo "Finish data preparation (stage: 0)."
fi


if [ ${stage} -le 1 ] && [ ! -e .done_stage_1 ]; then
  echo ============================================================================
  echo "                    Feature extranction (stage:1)                          "
  echo ============================================================================

  for x in train eval2000; do
      steps/make_fbank.sh --nj 16 --cmd "$train_cmd" --write_utt2num_frames true \
        ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1;
  done

  # Use the first 4k sentences as dev set.
  utils/subset_data_dir.sh --first ${data}/${train_set} 4000 ${data}/${dev_set} || exit 1; # 5hr 6min
  n=$[`cat ${data}/${train_set}/segments | wc -l` - 4000]
  utils/subset_data_dir.sh --last ${data}/${train_set} ${n} ${data}/${train_set}.tmp || exit 1;

  # Finally, the full training set:
  utils/data/remove_dup_utts.sh 300 ${data}/${train_set}.tmp ${data}/${train_set} || exit 1;  # 286hr
  rm -rf ${data}/*.tmp

  # Compute global CMVN
  compute-cmvn-stats scp:${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark || exit 1;

  # Apply global CMVN & dump features
  for x in ${train_set} ${dev_set}; do
    dump_dir=${data}/feat/${x}; mkdir -p ${dump_dir}
    dump_feat.sh --cmd "$train_cmd" --nj 16 --add_deltadelta false \
      ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x} ${dump_dir} || exit 1;
  done
  for x in ${test_set}; do
    dump_dir=${data}/feat/${x}; mkdir -p ${dump_dir}
    dump_feat.sh --cmd "$train_cmd" --nj 16 --add_deltadelta false \
      ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x} ${dump_dir} || exit 1;
  done

  touch .done_stage_1 && echo "Finish feature extranction (stage: 1)."
fi


dict=${data}/dict/${train_set}_${unit}${wp_model_type}${vocab_size}.txt; mkdir -p ${data}/dict/
nlsyms=${data}/dict/non_linguistic_symbols.txt
wp_model=${data}/dict/${train_set}_${wp_model_type}${vocab_size}
if [ ${stage} -le 2 ] && [ ! -e .done_stage_2_${unit}${wp_model_type}${vocab_size} ]; then
  echo ============================================================================
  echo "                      Dataset preparation (stage:2)                        "
  echo ============================================================================

  echo "make a non-linguistic symbol list"
  cut -f 2- ${data}/${train_set}/text | tr " " "\n" | sort | uniq | grep "\[" > ${nlsyms}
  cat ${nlsyms}

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
    spm_train --user_defined_symbols=`cat ${nlsyms} | tr "\n" ","` --input=${data}/dict/input.txt --vocab_size=${vocab_size} \
      --model_type=${wp_model_type} --model_prefix=${wp_model} --input_sentence_size=100000000
    spm_encode --model=${wp_model}.model --output_format=piece < ${data}/dict/input.txt | tr ' ' '\n' | \
      sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset-1}' >> ${dict}
  else
    text2dict.py ${data}/${train_set}/text --unit ${unit} --vocab_size ${vocab_size} --nlsyms ${nlsyms} \
      --wp_model_type ${wp_model_type} --wp_model ${wp_model} | \
      sort | uniq | grep -v -e '^\s*$' | awk -v offset=${offset} '{print $0 " " NR+offset-1}' >> ${dict} || exit 1;
  fi
  echo "vocab size:" `cat ${dict} | wc -l`

  # Compute OOV rate
  if [ ${unit} = word ]; then
    mkdir -p ${data}/dict/word_count ${data}/dict/oov_rate
    echo "OOV rate:" > ${data}/dict/oov_rate/word_${vocab_size}.txt
    for x in ${train_set} ${dev_set}; do
      cut -f 2- -d " " ${data}/${x}/text | tr " " "\n" | sort | uniq -c | sort -n -k1 -r \
        > ${data}/dict/word_count/${x}.txt || exit 1;
      compute_oov_rate.py ${data}/dict/word_count/${x}.txt ${dict} ${x} \
        >> ${data}/dict/oov_rate/word_${vocab_size}.txt || exit 1;
    done

    # 1) convert upper to lower
    # 2) remove tags (%AH) (%HESITATION) (%UH)
    # 3) remove <B_ASIDE> <E_ASIDE>
    # 4) remove "(" or ")"
    # swichboard
    grep -v en ${data}/${test_set}/text | cut -f 2- -d " " | awk '{ print tolower($0) }' | \
      perl -pe 's| \(\%.*\)||g' | perl -pe 's| \<.*\>||g' | sed -e "s/(//g" -e "s/)//g" | sed -e 's/\s\+/ /g' | \
      tr " " "\n" | sort | uniq -c | sort -n -k1 -r \
      > ${data}/dict/word_count/${test_set}_sw.txt || exit 1;
    compute_oov_rate.py ${data}/dict/word_count/${test_set}_sw.txt ${dict} ${test_set}_sw \
      >> ${data}/dict/oov_rate/word_${vocab_size}.txt || exit 1;
    # callhome
    grep -v sw ${data}/${test_set}/text | cut -f 2- -d " " | awk '{ print tolower($0) }' | \
      perl -pe 's| \(\%.*\)||g' | perl -pe 's| \<.*\>||g' | sed -e "s/(//g" -e "s/)//g" | sed -e 's/\s\+/ /g' | \
      tr " " "\n" | sort | uniq -c | sort -n -k1 -r \
      > ${data}/dict/word_count/${test_set}_ch.txt || exit 1;
    compute_oov_rate.py ${data}/dict/word_count/${test_set}_ch.txt ${dict} ${test_set}_ch \
      >> ${data}/dict/oov_rate/word_${vocab_size}.txt || exit 1;
    cat ${data}/dict/oov_rate/word_${vocab_size}.txt
  fi

  # Make datset csv files
  mkdir -p ${data}/dataset/
  for x in ${train_set} ${dev_set}; do
    echo "Making a csv file for ${x}..."
    dump_dir=${data}/feat/${x}
    make_dataset_csv.sh --feat ${dump_dir}/feats.scp --unit ${unit} --nlsyms ${nlsyms} --wp_model ${wp_model} \
      ${data}/${x} ${dict} > ${data}/dataset/${x}_${unit}${wp_model_type}${vocab_size}.csv || exit 1;
  done
  for x in ${test_set}; do
    dump_dir=${data}/feat/${x}
    make_dataset_csv.sh --is_test true --feat ${dump_dir}/feats.scp --unit ${unit} --nlsyms ${nlsyms} \
      ${data}/${x} ${dict} > ${data}/dataset/${x}_${unit}${wp_model_type}${vocab_size}.csv || exit 1;
  done

  touch .done_stage_2_${unit}${wp_model_type}${vocab_size} && echo "Finish creating dataset (stage: 2)."
fi


mkdir -p ${model_dir}
if [ ${stage} -le 3 ]; then
  echo ============================================================================
  echo "                      RNNLM Training stage (stage:3)                       "
  echo ============================================================================

  echo "Start RNNLM training..."

  # NOTE: support only a single GPU for RNNLM training
  # CUDA_VISIBLE_DEVICES=${gpu_ids} ../../../src/bin/lm/train.py \
  #   --corpus librispeech \
  #   --ngpus 1 \
  #   --train_set ${data}/dataset/${train_set}.csv \
  #   --dev_set ${data}/dataset/${dev_set}.csv \
  #   --eval_sets ${data}/dataset/eval1_${datasize}_${unit}${wp_model_type}${vocab_size}.csv \
  #   --config ${rnn_config} \
  #   --model ${model_dir} \
  #   --saved_model ${rnnlm_saved_model} || exit 1;
fi


if [ ${stage} -le 4 ]; then
  echo ============================================================================
  echo "                       ASR Training stage (stage:4)                        "
  echo ============================================================================

  echo "Start ASR training..."

  # export CUDA_LAUNCH_BLOCKING=1
  CUDA_VISIBLE_DEVICES=${gpu_ids} ../../../neural_sp/bin/asr/train.py \
    --corpus swbd \
    --ngpus ${ngpus} \
    --train_set ${data}/dataset/${train_set}_${unit}${wp_model_type}${vocab_size}.csv \
    --dev_set ${data}/dataset/${dev_set}_${unit}${wp_model_type}${vocab_size}.csv \
    --dict ${dict} \
    --wp_model ${wp_model} \
    --config ${asr_config} \
    --model ${model_dir} \
    --label_type ${unit} || exit 1;
    # --saved_model ${asr_saved_model} || exit 1;
    # TODO(hirofumi): send a e-mail
    # NOTE: ${train_set} is excluded in the training stage for swbd

  touch ${model}/.done_training && echo "Finish model training (stage: 4)."
fi
