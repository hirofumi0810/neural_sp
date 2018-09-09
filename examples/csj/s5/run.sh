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
# unit=bpe
# vocab_size=15000
vocab_size=30000

### feature extranction
pitch=0  ## or 1

### path to save the model
model_dir=/n/sd8/inaguma/result/csj

### path to the model directory to restart training
rnnlm_saved_model=
asr_saved_model=


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
# export datasize=aps_other
# export datasize=aps
# export datasize=sps  # TODO
# export datasize=all_except_dialog
export datasize=all
# NOTE: aps_other=default using "Academic lecture" and "other" data,
#       aps=using "Academic lecture" data,
#       sps=using "Academic lecture" data,
#       all_except_dialog=using All data except for "dialog" data,
#       all=using All data

### configuration
rnnlm_config=conf/rnnlm/${unit}_lstm_rnnlm_all.yml
asr_config=conf/attention/${unit}_blstm_att_${datasize}.yml
# asr_config=conf/ctc/${unit}_blstm_ctc_${datasize}.yml


. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

train_set=train_${datasize}
dev_set=dev_${datasize}
test_set="eval1 eval2 eval3"


if [ ${stage} -le 0 ] && [ ! -e .done_stage_0_${datasize} ]; then
  echo ============================================================================
  echo "                       Data Preparation (stage:0)                          "
  echo ============================================================================

  mkdir -p ${data}
  local/csj_make_trans/csj_autorun.sh ${CSJDATATOP} ${data}/csj-data ${CSJVER} || exit 1;
  local/csj_data_prep.sh ${data}/csj-data ${datasize} || exit 1;
  for x in eval1 eval2 eval3; do
    local/csj_eval_data_prep.sh ${data}/csj-data/eval ${x} || exit 1;
  done

  touch .done_stage_0_${datasize} && echo "Finish data preparation (stage: 0)."
fi

if [ ${stage} -le 1 ] && [ ! -e .done_stage_1_${datasize} ]; then
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
    dump_dir=${data}/feat/${x}; mkdir -p ${dump_dir}
    dump_feat.sh --cmd "$train_cmd" --nj 16 --add_deltadelta false \
      ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x} ${dump_dir} || exit 1;
  done
  for x in ${test_set}; do
    dump_dir=${data}/feat/${x}_${datasize}; mkdir -p ${dump_dir}
    dump_feat.sh --cmd "$train_cmd" --nj 16 --add_deltadelta false \
      ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x}_${datasize} ${dump_dir} || exit 1;
  done

  touch .done_stage_1_${datasize} && echo "Finish feature extranction (stage: 1)."
fi


dict=${data}/dict/${train_set}_${unit}_${vocab_size}.txt; mkdir -p ${data}/dict/
if [ ${stage} -le 2 ] && [ ! -e .done_stage_2_${datasize}_${unit}_${vocab_size} ]; then
  echo ============================================================================
  echo "                      Dataset preparation (stage:2)                        "
  echo ============================================================================

  # Remove <sp> (short pause)
  for x in ${train_set} ${dev_set} ${test_set}; do
    cat ${data}/${x}/text | sed -e 's/ <sp>//g' | sed -e 's/<sp> //g' > ${data}/${x}/text.tmp
    mv ${data}/${x}/text.tmp ${data}/${x}/text
  done

  # Make a dictionary
  echo "<blank> 0" > ${dict}
  echo "<unk> 1" >> ${dict}
  echo "<sos> 2" >> ${dict}
  echo "<eos> 3" >> ${dict}
  echo "<pad> 4" >> ${dict}
  offset=`cat ${dict} | wc -l`
  echo "Making a dictionary..."
  text2dict.py ${data}/${train_set}/text --unit ${unit} --vocab_size ${vocab_size} | \
    sort | uniq | grep -v -e '^\s*$' | awk -v offset=${offset} '{print $0 " " NR+offset-1}' >> ${dict} || exit 1;
  echo "vocab size:" `cat ${dict} | wc -l`

  # Compute OOV rate
  if [ ${unit} = word ]; then
    mkdir -p ${data}/dict/word_count ${data}/dict/oov_rate
    echo "OOV rate:" > ${data}/dict/oov_rate/word_${vocab_size}.txt
    for x in ${train_set} ${dev_set} ${test_set}; do
      cut -f 2- -d " " ${data}/${x}/text | tr " " "\n" | sort | uniq -c | sort -n -k1 -r \
        > ${data}/dict/word_count/${x}.txt || exit 1;
      compute_oov_rate.py ${data}/dict/word_count/${x}.txt ${dict} ${x} \
        >> ${data}/dict/oov_rate/word_${vocab_size}.txt || exit 1;
    done
    cat ${data}/dict/oov_rate/word_${vocab_size}.txt
  fi

  # Make datset csv files
  mkdir -p ${data}/dataset/
  for x in ${train_set} ${dev_set}; do
    echo "Making a csv file for ${x}..."
    dump_dir=${data}/feat/${x}
    make_dataset_csv.sh --feat ${dump_dir}/feats.scp --unit ${unit} \
      ${data}/${x} ${dict} > ${data}/dataset/${x}_${unit}_${vocab_size}.csv || exit 1;
  done
  for x in ${test_set}; do
    dump_dir=${data}/feat/${x}_${datasize}
    make_dataset_csv.sh --is_test true --feat ${dump_dir}/feats.scp --unit ${unit} \
      ${data}/${x} ${dict} > ${data}/dataset/${x}_${datasize}_${unit}_${vocab_size}.csv || exit 1;
  done

  touch .done_stage_2_${datasize}_${unit}_${vocab_size} && echo "Finish creating dataset (stage: 2)."
fi


mkdir -p ${model_dir}
if [ ${stage} -le 3 ]; then
  echo ============================================================================
  echo "                      RNNLM Training stage (stage:3)                       "
  echo ============================================================================

  echo "Start RNNLM training..."

  # NOTE: support only a single GPU for RNNLM training
  # CUDA_VISIBLE_DEVICES=${gpu_ids} ../../../src/bin/lm/train.py \
  #   --corpus csj \
  #   --ngpus 1 \
  #   --train_set ${data}/dataset/${train_set}.csv \
  #   --dev_set ${data}/dataset/${dev_set}.csv \
  #   --eval_sets ${data}/dataset/eval1_${datasize}.csv \
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
    --corpus csj \
    --ngpus ${ngpus} \
    --train_set ${data}/dataset/${train_set}_${unit}_${vocab_size}.csv \
    --dev_set ${data}/dataset/${dev_set}_${unit}_${vocab_size}.csv \
    --eval_sets ${data}/dataset/eval1_${datasize}_${unit}_${vocab_size}.csv \
    --dict ${dict} \
    --config ${asr_config} \
    --model ${model_dir} \
    --label_type ${unit} || exit 1;
    # --saved_model ${asr_saved_model} || exit 1;
    # TODO(hirofumi): send a e-mail

  touch ${model}/.done_training && echo "Finish model training (stage: 4)."
fi
