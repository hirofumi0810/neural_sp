#!/bin/bash

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                                  CI Test                                 "
echo ============================================================================

stage=0
stop_stage=5
gpu=
benchmark=true
speed_perturb=false
stdout=false

### vocabulary
unit=char      # word/wp/char/word_char
vocab=100
wp_type=bpe  # bpe/unigram (for wordpiece)

#########################
# ASR configuration
#########################
conf=conf/asr/blstm_las.yaml
conf2=
asr_init=
external_lm=

#########################
# LM configuration
#########################
lm_conf=conf/lm/rnnlm.yaml

### path to save the model
model=results

### path to the model directory to resume training
resume=
lm_resume=

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

if [ -z ${gpu} ]; then
    n_gpus=0
else
    n_gpus=$(echo ${gpu} | tr "," "\n" | wc -l)
fi

train_set=train
dev_set=train

if [ ${unit} = char ]; then
    vocab=
fi
if [ ${unit} != wp ]; then
    wp_type=
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ] && [ ! -e data/.done_stage_0 ]; then
    echo ============================================================================
    echo "                       Data Preparation (stage:0)                          "
    echo ============================================================================

    # download data
    mkdir -p data
    local/download_sample.sh || exit 1;

    touch data/.done_stage_0 && echo "Finish data preparation (stage: 0)."
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ ! -e data/.done_stage_1 ]; then
    echo ============================================================================
    echo "                    Feature extranction (stage:1)                          "
    echo ============================================================================

    for x in train; do
        steps/make_fbank.sh --nj 1 --cmd "$train_cmd" --write_utt2num_frames true \
            data/${x} data/log/make_fbank/${x} data/fbank
        data/log/make_fbank/train/make_fbank_train.1.log
    done

    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark || exit 1;

    # Apply global CMVN & dump features
    dump_feat.sh --cmd "$train_cmd" --nj 1 \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark data/log/dump_feat/${train_set} data/dump/${train_set} || exit 1;

    touch data/.done_stage_1 && echo "Finish feature extranction (stage: 1)."
fi

dict=data/dict/${train_set}_${unit}${wp_type}${vocab}.txt; mkdir -p data/dict
wp_model=data/dict/${train_set}_${wp_type}${vocab}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ ! -e data/.done_stage_2_${unit}${wp_type}${vocab} ]; then
    echo ============================================================================
    echo "                      Dataset preparation (stage:2)                        "
    echo ============================================================================

    if [ ${unit} = wp ]; then
        make_vocab.sh --unit ${unit} \
            --vocab ${vocab} --wp_type ${wp_type} --wp_model ${wp_model} \
            data ${dict} data/${train_set}/text || exit 1;
    else
        make_vocab.sh --unit ${unit} \
            data ${dict} data/${train_set}/text || exit 1;
    fi

    # Compute OOV rate
    if [ ${unit} = word ]; then
        mkdir -p data/dict/word_count data/dict/oov_rate
        echo "OOV rate:" > data/dict/oov_rate/word${vocab}.txt
        for x in ${train_set}; do
            cut -f 2- -d " " data/${x}/text | tr " " "\n" | sort | uniq -c | sort -n -k1 -r \
                > data/dict/word_count/${x}.txt || exit 1;
            compute_oov_rate.py data/dict/word_count/${x}.txt ${dict} ${x} \
                >> data/dict/oov_rate/word${vocab}.txt || exit 1;
            # NOTE: speed perturbation is not considered
        done
        cat data/dict/oov_rate/word${vocab}.txt
    fi

    echo "Making dataset tsv files for ASR ..."
    mkdir -p data/dataset
    make_dataset.sh --feat data/dump/${train_set}/feats.scp --unit ${unit} --wp_model ${wp_model} \
        data/${train_set} ${dict} > data/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv || exit 1;

    touch data/.done_stage_2_${unit}${wp_type}${vocab} && echo "Finish creating dataset for ASR (stage: 2)."
fi

mkdir -p ${model}
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo ============================================================================
    echo "                        LM Training stage (stage:3)                       "
    echo ============================================================================

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/lm/train.py \
        --corpus ci_test \
        --config ${lm_conf} \
        --n_gpus ${n_gpus} \
        --cudnn_benchmark ${benchmark} \
        --train_set data/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv \
        --dev_set data/dataset/${dev_set}_${unit}${wp_type}${vocab}.tsv \
        --unit ${unit} \
        --dict ${dict} \
        --wp_model ${wp_model}.model \
        --model_save_dir ${model}/lm \
        --stdout ${stdout} \
        --resume ${lm_resume} || exit 1;

    echo "Finish LM training (stage: 3)."
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo ============================================================================
    echo "                       ASR Training stage (stage:4)                        "
    echo ============================================================================

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/asr/train.py \
        --corpus ci_test \
        --config ${conf} \
        --config2 ${conf2} \
        --n_gpus ${n_gpus} \
        --cudnn_benchmark ${benchmark} \
        --train_set data/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv \
        --dev_set data/dataset/${dev_set}_${unit}${wp_type}${vocab}.tsv \
        --unit ${unit} \
        --dict ${dict} \
        --wp_model ${wp_model}.model \
        --model_save_dir ${model}/asr \
        --asr_init ${asr_init} \
        --external_lm ${external_lm} \
        --stdout ${stdout} \
        --resume ${resume} || exit 1;

    echo "Finish ASR model training (stage: 4)."
fi
