#!/bin/bash

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                                AISHELL-1                                 "
echo ============================================================================

stage=-1
stop_stage=5
gpu=
benchmark=true
stdout=false

### vocabulary
unit=char
unit_sub1=phone

#########################
# ASR configuration
#########################
conf=conf/asr/blstm_las.yaml
conf2=
asr_init=

### path to save the model
model=/n/work2/inaguma/results/aishell1

### path to the model directory to resume training
resume=
lm_resume=

### path to save preproecssed data
export data=/n/work2/inaguma/corpus/aishell1

# Base url for downloads.
data_url=www.openslr.org/resources/33

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

train_set=train_sp
dev_set=dev_sp
test_set="test_sp"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    mkdir -p ${data}
    local/download_and_untar.sh ${data} ${data_url} data_aishell
    local/download_and_untar.sh ${data} ${data_url} resource_aishell
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ] && [ ! -e ${data}/.done_stage_0 ]; then
    echo ============================================================================
    echo "                       Data Preparation (stage:0)                          "
    echo ============================================================================

    local/aishell_data_prep.sh ${data}/data_aishell/wav ${data}/data_aishell/transcript
    # remove space in text
    for x in train dev test; do
        cp ${data}/${x}/text ${data}/${x}/text.org
        paste -d " " <(cut -f 1 -d" " ${data}/${x}/text.org) <(cut -f 2- -d" " ${data}/${x}/text.org | tr -d " ") \
            > ${data}/${x}/text
        rm ${data}/${x}/text.org
    done

    touch ${data}/.done_stage_0 && echo "Finish data preparation (stage: 0)."
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ ! -e ${data}/.done_stage_1 ]; then
    echo ============================================================================
    echo "                    Feature extranction (stage:1)                          "
    echo ============================================================================

    for x in train dev test; do
        steps/make_fbank.sh --nj 32 --cmd "$train_cmd" --write_utt2num_frames true \
            ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1;
        utils/fix_data_dir.sh ${data}/${x}
    done

    speed_perturb_3way.sh ${data} train ${train_set}
    cp -rf ${data}/dev ${data}/${dev_set}
    cp -rf ${data}/test ${data}/${test_set}

    # Compute global CMVN
    compute-cmvn-stats scp:${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark || exit 1;

    # Apply global CMVN & dump features
    dump_feat.sh --cmd "$train_cmd" --nj 80 \
        ${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${train_set} ${data}/dump/${train_set} || exit 1;
    for x in ${dev_set} ${test_set}; do
        dump_dir=${data}/dump/${x}
        dump_feat.sh --cmd "$train_cmd" --nj 32 \
            ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x} ${dump_dir} || exit 1;
    done

    touch ${data}/.done_stage_1
fi

dict=${data}/dict/${train_set}.txt; mkdir -p ${data}/dict
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ ! -e ${data}/.done_stage_2 ]; then
    echo ============================================================================
    echo "                      Dataset preparation (stage:2)                        "
    echo ============================================================================

    make_vocab.sh --unit ${unit} --speed_perturb true \
        ${data} ${dict} ${data}/${train_set}/text || exit 1;

    echo "Making dataset tsv files for ASR ..."
    mkdir -p ${data}/dataset
    for x in ${train_set} ${dev_set} ${test_set}; do
        dump_dir=${data}/dump/${x}
        make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit} \
            ${data}/${x} ${dict} > ${data}/dataset/${x}.tsv || exit 1;
    done

    touch ${data}/.done_stage_2 && echo "Finish creating dataset for ASR (stage: 2)."
fi

dict_sub1=${data}/dict/${train_set}_${unit_sub1}.txt; mkdir -p ${data}/dict
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ ! -e ${data}/.done_stage_2_${unit_sub1} ]; then
    echo ============================================================================
    echo "                      Dataset preparation (stage:2, sub1)                        "
    echo ============================================================================

    lexicon=${data}/resource_aishell/lexicon.txt
    unk=sil
    for x in ${train_set} ${dev_set} ${test_set}; do
        map2phone.py --text ${data}/${x}/text --lexicon ${lexicon} --unk ${unk} --word_segmentation false \
            > ${data}/${x}/text.phone
    done
    make_vocab.sh --unit ${unit_sub1} --speed_perturb true \
        ${data} ${dict_sub1} ${data}/${train_set}/text.phone || exit 1;

    echo "Making dataset tsv files for ASR ..."
    for x in ${train_set} ${dev_set} ${test_set}; do
        dump_dir=${data}/dump/${x}
        make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit_sub1} --text ${data}/${x}/text.phone \
            ${data}/${x} ${dict_sub1} > ${data}/dataset/${x}_${unit_sub1}.tsv || exit 1;
    done

    touch ${data}/.done_stage_2_${unit_sub1} && echo "Finish creating dataset for ASR (stage: 2)."
fi


mkdir -p ${model}
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo ============================================================================
    echo "                       ASR Training stage (stage:4)                        "
    echo ============================================================================

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/asr/train.py \
        --corpus aishell1 \
        --config ${conf} \
        --config2 ${conf2} \
        --n_gpus ${n_gpus} \
        --cudnn_benchmark ${benchmark} \
        --train_set ${data}/dataset/${train_set}.tsv \
        --train_set_sub1 ${data}/dataset/${train_set}_${unit_sub1}.tsv \
        --dev_set ${data}/dataset/${dev_set}.tsv \
        --dev_set_sub1 ${data}/dataset/${dev_set}_${unit_sub1}.tsv \
        --unit ${unit} \
        --unit_sub1 ${unit_sub1} \
        --dict ${dict} \
        --dict_sub1 ${dict_sub1} \
        --model_save_dir ${model}/asr \
        --asr_init ${asr_init} \
        --stdout ${stdout} \
        --resume ${resume} || exit 1;

    echo "Finish ASR model training (stage: 4)."
fi
