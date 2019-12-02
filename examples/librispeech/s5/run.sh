#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                              LibriSpeech                                  "
echo ============================================================================

stage=0
gpu=
speed_perturb=false
spec_augment=false
stdout=false

### vocabulary
unit=wp      # word/wp/char/word_char
vocab=30000
wp_type=bpe  # bpe/unigram (for wordpiece)

#########################
# ASR configuration
#########################
conf=conf/asr/blstm_las.yaml
conf2=
asr_init=
lm_init=

# TDS
# enc_type=tds
# conv_channels="10_10_14_14_14_18_18_18_18_18_18"
# conv_kernel_sizes="(21,1)_(21,1)_(21,1)_(21,1)_(21,1)_(21,1)_(21,1)_(21,1)_(21,1)_(21,1)_(21,1)"
# subsample="1_1_1_1_1"
# clip_grad_norm=15.0
# dropout_enc=0.2
# lsm_prob=0.05

# GatedConv
# enc_type=gated_conv
# conv_channels="200_220_242_266_292_321_353_388_426_468_514_565_621_683_751_826_908"
# conv_kernel_sizes="(13,1)_(14,1)_(15,1)_(16,1)_(17,1)_(18,1)_(19,1)_(20,1)_(21,1)_(22,1)_(23,1)_(24,1)_(25,1)_(26,1)_(27,1)_(28,1)_(29,1)"
# subsample="1_1_1_1_1"

#########################
# LM configuration
#########################
lm_conf=conf/lm/rnnlm.yaml

### path to save the model
model=/n/work1/inaguma/results/librispeech

### path to the model directory to resume training
resume=
lm_resume=

### path to save preproecssed data
export data=/n/work1/inaguma/corpus/librispeech

### path to download data
data_download_path=/n/rd21/corpora_7/librispeech/

### data size
datasize=960     # 100/460/960
lm_datasize=960  # 100/460/960
use_external_text=true

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

if [ ${speed_perturb} = true ]; then
    conf2=conf/asr/speed_perturb.yaml
elif [ ${spec_augment} = true ]; then
    conf2=conf/asr/spec_augment.yaml
fi

if [ -z ${gpu} ]; then
    echo "Error: set GPU number." 1>&2
    echo "Usage: ./run.sh --gpu 0" 1>&2
    exit 1
fi
n_gpus=$(echo ${gpu} | tr "," "\n" | wc -l)

# Base url for downloads.
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11

train_set=train_${datasize}
dev_set=dev_clean
test_set="dev_clean dev_other test_clean test_other"
if [ ${speed_perturb} = true ]; then
    train_set=train_sp_${datasize}
    dev_set=dev_clean_sp
    test_set="dev_clean_sp dev_other_sp test_clean_sp test_other_sp"
fi

if [ ${unit} = char ]; then
    vocab=
fi
if [ ${unit} != wp ]; then
    wp_type=
fi

if [ ${stage} -le 0 ] && [ ! -e ${data}/.done_stage_0 ]; then
    echo ============================================================================
    echo "                       Data Preparation (stage:0)                          "
    echo ============================================================================

    # download data
    mkdir -p ${data}
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        local/download_and_untar.sh ${data_download_path} ${data_url} ${part} || exit 1;
    done

    # download the LM resources
    local/download_lm.sh ${lm_url} ${data}/local/lm || exit 1;

    # format the data as Kaldi data directories
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        # use underscore-separated names in data directories.
        local/data_prep.sh ${data_download_path}/LibriSpeech/${part} ${data}/$(echo ${part} | sed s/-/_/g) || exit 1;
    done

    # lowercasing
    for x in dev_clean test_clean dev_other test_other train_clean_100 train_clean_360 train_other_500; do
        cp ${data}/${x}/text ${data}/${x}/text.org
        paste -d " " <(cut -f 1 -d " " ${data}/${x}/text.org) \
            <(cut -f 2- -d " " ${data}/${x}/text.org | awk '{print tolower($0)}') > ${data}/${x}/text
    done

    touch ${data}/.done_stage_0 && echo "Finish data preparation (stage: 0)."
fi

if [ ${stage} -le 1 ] && [ ! -e ${data}/.done_stage_1_${datasize}_sp${speed_perturb} ]; then
    echo ============================================================================
    echo "                    Feature extranction (stage:1)                          "
    echo ============================================================================

    for x in dev_clean test_clean dev_other test_other train_clean_100; do
        steps/make_fbank.sh --nj 32 --cmd "$train_cmd" --write_utt2num_frames true \
            ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1;
    done

    if [ ${datasize} == '100' ]; then
        utils/combine_data.sh --extra_files "utt2num_frames" ${data}/${train_set} \
            ${data}/train_clean_100 || exit 1;
    elif [ ${datasize} == '460' ]; then
        steps/make_fbank.sh --nj 32 --cmd "$train_cmd" --write_utt2num_frames true \
            ${data}/train_clean_360 ${data}/log/make_fbank/train_clean_360 ${data}/fbank || exit 1;
        utils/combine_data.sh --extra_files "utt2num_frames" ${data}/${train_set} \
            ${data}/train_clean_100 ${data}/train_clean_360 || exit 1;
    elif [ ${datasize} == '960' ]; then
        steps/make_fbank.sh --nj 32 --cmd "$train_cmd" --write_utt2num_frames true \
            ${data}/train_clean_360 ${data}/log/make_fbank/train_clean_360 ${data}/fbank || exit 1;
        steps/make_fbank.sh --nj 32 --cmd "$train_cmd" --write_utt2num_frames true \
            ${data}/train_other_500 ${data}/log/make_fbank/train_other_500 ${data}/fbank || exit 1;
        utils/combine_data.sh --extra_files "utt2num_frames" ${data}/${train_set} \
            ${data}/train_clean_100 ${data}/train_clean_360 ${data}/train_other_500 || exit 1;
    else
        echo "datasize is 100 or 460 or 960." && exit 1;
    fi

    if [ ${speed_perturb} = true ]; then
        # speed-perturbed
        speed_perturb_3way.sh ${data} train_${datasize} ${train_set}

        cp -rf ${data}/dev_clean ${data}/dev_clean_sp
        cp -rf ${data}/dev_other ${data}/dev_other_sp
        cp -rf ${data}/test_clean ${data}/test_clean_sp
        cp -rf ${data}/test_other ${data}/test_other_sp
    fi

    # Compute global CMVN
    compute-cmvn-stats scp:${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark || exit 1;

    # Apply global CMVN & dump features
    dump_feat.sh --cmd "$train_cmd" --nj 80 \
        ${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${train_set} ${data}/dump/${train_set} || exit 1;
    for x in ${test_set}; do
        dump_dir=${data}/dump/${x}_${datasize}
        dump_feat.sh --cmd "$train_cmd" --nj 32 \
            ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x}_${datasize} ${dump_dir} || exit 1;
    done

    touch ${data}/.done_stage_1_${datasize}_sp${speed_perturb} && echo "Finish feature extranction (stage: 1)."
fi

dict=${data}/dict/${train_set}_${unit}${wp_type}${vocab}.txt; mkdir -p ${data}/dict
wp_model=${data}/dict/${train_set}_${wp_type}${vocab}
if [ ${stage} -le 2 ] && [ ! -e ${data}/.done_stage_2_${datasize}_${unit}${wp_type}${vocab}_sp${speed_perturb} ]; then
    echo ============================================================================
    echo "                      Dataset preparation (stage:2)                        "
    echo ============================================================================

    echo "Making a dictionary..."
    echo "<unk> 1" > ${dict}  # <unk> must be 1, 0 will be used for "blank" in CTC
    echo "<eos> 2" >> ${dict}  # <sos> and <eos> share the same index
    echo "<pad> 3" >> ${dict}
    [ ${unit} = char ] && echo "<space> 4" >> ${dict}
    offset=$(cat ${dict} | wc -l)
    if [ ${unit} = wp ]; then
        if [ ${speed_perturb} = true ]; then
            grep sp1.0 ${data}/${train_set}/text > ${data}/${train_set}/text.org
            cp ${data}/${dev_set}/text ${data}/${dev_set}/text.org
            cut -f 2- -d " " ${data}/${train_set}/text.org > ${data}/dict/input.txt
        else
            cut -f 2- -d " " ${data}/${train_set}/text > ${data}/dict/input.txt
        fi
        spm_train --input=${data}/dict/input.txt --vocab_size=${vocab} \
            --model_type=${wp_type} --model_prefix=${wp_model} --input_sentence_size=100000000 --character_coverage=1.0
        spm_encode --model=${wp_model}.model --output_format=piece < ${data}/dict/input.txt | tr ' ' '\n' | \
            sort | uniq -c | sort -n -k1 -r | sed -e 's/^[ ]*//g' | cut -d " " -f 2 | grep -v '^\s*$' | awk -v offset=${offset} '{print $1 " " NR+offset}' >> ${dict}
    else
        text2dict.py ${data}/${train_set}/text --unit ${unit} --vocab ${vocab} --speed_perturb ${speed_perturb} | \
            awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict} || exit 1;
    fi
    echo "vocab size:" $(cat ${dict} | wc -l)

    # Compute OOV rate
    if [ ${unit} = word ]; then
        mkdir -p ${data}/dict/word_count ${data}/dict/oov_rate
        echo "OOV rate:" > ${data}/dict/oov_rate/word${vocab}_${datasize}.txt
        for x in ${train_set} ${test_set}; do
            if [ ${speed_perturb} = true ]; then
                cut -f 2- -d " " ${data}/${x}/text.org | tr " " "\n" | sort | uniq -c | sort -n -k1 -r \
                    > ${data}/dict/word_count/${x}_${datasize}.txt || exit 1;
            else
                cut -f 2- -d " " ${data}/${x}/text | tr " " "\n" | sort | uniq -c | sort -n -k1 -r \
                    > ${data}/dict/word_count/${x}_${datasize}.txt || exit 1;
            fi
            compute_oov_rate.py ${data}/dict/word_count/${x}_${datasize}.txt ${dict} ${x} \
                >> ${data}/dict/oov_rate/word${vocab}_${datasize}.txt || exit 1;
        done
        cat ${data}/dict/oov_rate/word${vocab}_${datasize}.txt
    fi

    echo "Making dataset tsv files for ASR ..."
    mkdir -p ${data}/dataset
    make_dataset.sh --feat ${data}/dump/${train_set}/feats.scp --unit ${unit} --wp_model ${wp_model} \
        ${data}/${train_set} ${dict} > ${data}/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv || exit 1;
    for x in ${test_set}; do
        dump_dir=${data}/dump/${x}_${datasize}
        make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit} --wp_model ${wp_model} \
            ${data}/${x} ${dict} > ${data}/dataset/${x}_${datasize}_${unit}${wp_type}${vocab}.tsv || exit 1;
    done

    touch ${data}/.done_stage_2_${datasize}_${unit}${wp_type}${vocab}_sp${speed_perturb} && echo "Finish creating dataset for ASR (stage: 2)."
fi

exit 1

mkdir -p ${model}
if [ ${stage} -le 3 ] && [ ${speed_perturb} = false ]; then
    echo ============================================================================
    echo "                        LM Training stage (stage:3)                       "
    echo ============================================================================

    if [ ! -e ${data}/.done_stage_3_${datasize}${lm_datasize}_${unit}${wp_type}${vocab}_${use_external_text} ]; then
        [ ! -e ${data}/.done_stage_1_${datasize}_sp${speed_perturb} ] && echo "run ./run.sh --datasize ${lm_datasize} first" && exit 1

        echo "Making dataset tsv files for LM ..."
        mkdir -p ${data}/dataset_lm
        for x in train dev_clean dev_other test_clean test_other; do
            if [ ${lm_datasize} = ${datasize} ]; then
                cp ${data}/dataset/${x}_${datasize}_${unit}${wp_type}${vocab}.tsv \
                    ${data}/dataset_lm/${x}_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv || exit 1;
            else
                make_dataset.sh --unit ${unit} --wp_model ${wp_model} \
                    ${data}/${x} ${dict} > ${data}/dataset_lm/${x}_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv || exit 1;
            fi
        done

        # use external data
        if ${use_external_text}; then
            if [ ! -e ${data}/local/lm_train/librispeech-lm-norm.txt.gz ]; then
                wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P ${data}/local/lm_train/
            fi
            zcat ${data}/local/lm_train/librispeech-lm-norm.txt.gz | shuf | awk '{print "unpaired-text-"NR, tolower($0)}' > ${data}/dataset_lm/text
            update_dataset.sh --unit ${unit} --wp_model ${wp_model} \
                ${data}/dataset_lm/text ${dict} ${data}/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv \
                > ${data}/dataset_lm/train_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}_external.tsv || exit 1;
        fi

        touch ${data}/.done_stage_3_${datasize}${lm_datasize}_${unit}${wp_type}${vocab}_${use_external_text} && echo "Finish creating dataset for LM (stage: 3)."
    fi

    if ${use_external_text}; then
        lm_train_set="${data}/dataset_lm/train_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}_external.tsv"
    else
        lm_train_set="${data}/dataset_lm/train_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv"
    fi

    lm_test_set="${data}/dataset_lm/dev_other_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv \
                 ${data}/dataset_lm/test_clean_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv \
                 ${data}/dataset_lm/test_other_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv"

    # NOTE: support only a single GPU for LM training
    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/lm/train.py \
        --corpus librispeech \
        --config ${lm_conf} \
        --n_gpus ${n_gpus} \
        --train_set ${lm_train_set} \
        --dev_set ${data}/dataset_lm/dev_clean_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv \
        --eval_sets ${lm_test_set} \
        --unit ${unit} \
        --dict ${dict} \
        --wp_model ${wp_model}.model \
        --model_save_dir ${model}/lm \
        --stdout ${stdout} \
        --resume ${lm_resume} || exit 1;

    echo "Finish LM training (stage: 3)." && exit 1;
fi

if [ ${stage} -le 4 ]; then
    echo ============================================================================
    echo "                       ASR Training stage (stage:4)                        "
    echo ============================================================================

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/asr/train.py \
        --corpus librispeech \
        --config ${conf} \
        --config2 ${conf2} \
        --n_gpus ${n_gpus} \
        --train_set ${data}/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv \
        --dev_set ${data}/dataset/${dev_set}_${datasize}_${unit}${wp_type}${vocab}.tsv \
        --unit ${unit} \
        --dict ${dict} \
        --wp_model ${wp_model}.model \
        --model_save_dir ${model}/asr \
        --asr_init ${asr_init} \
        --lm_init ${lm_init} \
        --stdout ${stdout} \
        --resume ${resume} || exit 1;

    echo "Finish ASR model training (stage: 4)."
fi
