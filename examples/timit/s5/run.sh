#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                                  TIMIT                                    "
echo ============================================================================

stage=0
stop_stage=5
gpu=
benchmark=true
stdout=false

#########################
# ASR configuration
#########################
conf=conf/blstm_ctc.yaml

### path to save the model
model=/n/work2/inaguma/results/timit

### path to the model directory to resume training
resume=

### path to save preproecssed data
export data=/n/work2/inaguma/corpus/timit

### path to original data
TIMITDATATOP=/n/rd21/corpora_1/TIMIT

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

if [ -z ${gpu} ]; then
    echo "Error: set GPU number." 1>&2
    echo "Usage: ./run.sh --gpu 0" 1>&2
    exit 1
fi
n_gpus=$(echo ${gpu} | tr "," "\n" | wc -l)

train_set=train
dev_set=dev
test_set="test"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ] && [ ! -e ${data}/.done_stage_0 ]; then
    echo ============================================================================
    echo "                       Data Preparation (stage:0)                          "
    echo ============================================================================

    mkdir -p ${data}
    local/timit_data_prep.sh ${TIMITDATATOP} || exit 1;
    local/timit_format_data.sh || exit 1;

    touch ${data}/.done_stage_0 && echo "Finish data preparation (stage: 0)."
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ ! -e ${data}/.done_stage_1 ]; then
    echo ============================================================================
    echo "                    Feature extranction (stage:1)                          "
    echo ============================================================================

    for x in train dev test; do
        steps/make_fbank.sh --nj 32 --cmd "$train_cmd" --write_utt2num_frames true \
            ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1;
    done

    # Compute global CMVN
    compute-cmvn-stats scp:${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark || exit 1;

    # Apply global CMVN & dump features
    dump_feat.sh --cmd "$train_cmd" --nj 80 --add_deltas true \
        ${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${train_set} ${data}/dump/${train_set} || exit 1;
    for x in ${dev_set} ${test_set}; do
        dump_dir=${data}/dump/${x}
        dump_feat.sh --cmd "$train_cmd" --nj 32 --add_deltas true \
            ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x} ${dump_dir} || exit 1;
    done

    touch ${data}/.done_stage_1 && echo "Finish feature extranction (stage: 1)."
fi

dict=${data}/dict/${train_set}.txt; mkdir -p ${data}/dict
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ ! -e ${data}/.done_stage_2 ]; then
    echo ============================================================================
    echo "                      Dataset preparation (stage:2)                        "
    echo ============================================================================

    echo "Making a dictionary..."
    echo "<unk> 1" > ${dict}  # <unk> must be 1, 0 will be used for "blank" in CTC
    echo "<eos> 2" >> ${dict}  # <sos> and <eos> share the same index
    echo "<pad> 3" >> ${dict}
    offset=$(cat ${dict} | wc -l)
    text2dict.py ${data}/${train_set}/text --unit phone | \
        awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict} || exit 1;
    echo "vocab size:" $(cat ${dict} | wc -l)

    echo "Making dataset tsv files for ASR ..."
    mkdir -p ${data}/dataset
    for x in ${train_set} ${dev_set} ${test_set}; do
        dump_dir=${data}/dump/${x}
        make_dataset.sh --feat ${dump_dir}/feats.scp --unit phone \
            ${data}/${x} ${dict} > ${data}/dataset/${x}.tsv || exit 1;
    done

    touch ${data}/.done_stage_2 && echo "Finish creating dataset for ASR (stage: 2)."
fi

# NOTE: skip LM training (stage:3)

mkdir -p ${model}
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo ============================================================================
    echo "                       ASR Training stage (stage:4)                        "
    echo ============================================================================

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/asr/train.py \
        --corpus timit \
        --config ${conf} \
        --n_gpus ${n_gpus} \
        --cudnn_benchmark ${benchmark} \
        --train_set ${data}/dataset/${train_set}.tsv \
        --dev_set ${data}/dataset/${dev_set}.tsv \
        --eval_sets ${data}/dataset/${test_set}.tsv \
        --unit phone \
        --dict ${dict} \
        --model_save_dir ${model}/asr \
        --model ${model}/asr \
        --stdout ${stdout} \
        --resume ${resume} || exit 1;

    echo "Finish ASR model training (stage: 4)."
fi
