#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                                   WSJ                                     "
echo ============================================================================

stage=0
stop_stage=5
gpu=
benchmark=true
speed_perturb=false
stdout=false

### vocabulary
unit=wp      # word/wp/char/word_char
vocab=500
wp_type=bpe  # bpe/unigram (for wordpiece)

#########################
# ASR configuration
#########################
conf=conf/asr/blstm_las.yaml
conf2=
asr_init=
external_lm=
teacher=
teacher_lm=

#########################
# LM configuration
#########################
lm_conf=conf/lm/rnnlm.yaml

### path to save the model
model=/n/work2/inaguma/results/wsj

### path to the model directory to resume training
resume=
lm_resume=

### path to save preproecssed data
export data=/n/work2/inaguma/corpus/wsj

### path to original data
wsj0=/n/work2/inaguma/corpus/csr_1_senn
wsj1=/n/work2/inaguma/corpus/csr_senn

# Sometimes, we have seen WSJ distributions that do not have subdirectories
# like '11-13.1', but instead have 'doc', 'si_et_05', etc. directly under the
# wsj0 or wsj1 directories. In such cases, try the following:
CSTR_WSJTATATOP=/n/rd21/corpora_1/WSJ
# $CSTR_WSJTATATOP must contain a 'wsj0' and a 'wsj1' subdirectory for this to work.

directory_type=original # original/cstr

### data size
datasize=si284  # si284 or si84

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

if [ ${speed_perturb} = true ]; then
  if [ -z ${conf2} ]; then
    echo "Error: Set --conf2." 1>&2
    exit 1
  fi
fi

if [ -z ${gpu} ]; then
    echo "Error: set GPU number." 1>&2
    echo "Usage: ./run.sh --gpu 0" 1>&2
    exit 1
fi
n_gpus=$(echo ${gpu} | tr "," "\n" | wc -l)

train_set=train_${datasize}
dev_set=test_dev93
test_set="test_eval92"
if [ ${speed_perturb} = true ]; then
    train_set=train_sp_${datasize}
    dev_set=test_dev93_sp
    test_set="test_eval92_sp"
fi

if [ ${unit} = char ]; then
    vocab=
fi
if [ ${unit} != wp ]; then
    wp_type=
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ] && [ ! -e ${data}/.done_stage_0 ]; then
    echo ============================================================================
    echo "                       Data Preparation (stage:0)                          "
    echo ============================================================================

    case ${directory_type} in
        original) local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.? || exit 1; ;;
        cstr) local/cstr_wsj_data_prep.sh $CSTR_WSJTATATOP || exit 1; ;;
    esac

    # "nosp" refers to the dictionary before silence probabilities and pronunciation
    # probabilities are added.
    local/wsj_prepare_dict.sh --dict-suffix "_nosp" || exit 1;
    local/wsj_format_data.sh --lang-suffix "_nosp" || exit 1;
    case ${directory_type} in
        original) local/wsj_extend_dict.sh --dict-suffix "_nosp" ${wsj1}/13-32.1 || exit 1; ;;
        cstr) local/cstr_wsj_extend_dict.sh --dict-suffix "_nosp" $CSTR_WSJTATATOP/wsj1/doc/ || exit 1; ;;
    esac

    # lowercasing
    for x in train_si84 train_si284 test_dev93 test_eval92; do
        cp ${data}/${x}/text ${data}/${x}/text.org
        paste -d " " <(cut -f 1 -d " " ${data}/${x}/text.org) \
            <(cut -f 2- -d " " ${data}/${x}/text.org | awk '{print tolower($0)}') > ${data}/${x}/text
    done

    touch ${data}/.done_stage_0 && echo "Finish data preparation (stage: 0)."
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ ! -e ${data}/.done_stage_1_${datasize}_sp${speed_perturb} ]; then
    echo ============================================================================
    echo "                    Feature extranction (stage:1)                          "
    echo ============================================================================

    if [ ! -e ${data}/.done_stage_1_${datasize}_spfalse ]; then
        for x in train_${datasize} test_dev93 test_eval92; do
            steps/make_fbank.sh --nj 32 --cmd "$train_cmd" --write_utt2num_frames true \
                ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1;
        done
    fi

    if [ ${speed_perturb} = true ]; then
        speed_perturb_3way.sh ${data} train_${datasize} ${train_set}
        cp -rf ${data}/test_dev93 ${data}/test_dev93_sp
        cp -rf ${data}/test_eval92 ${data}/test_eval92_sp
    fi

    # Compute global CMVN
    compute-cmvn-stats scp:${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark || exit 1;

    # Apply global CMVN & dump features
    dump_feat.sh --cmd "$train_cmd" --nj 80 \
        ${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${train_set} ${data}/dump/${train_set} || exit 1;
    for x in ${dev_set} ${test_set}; do
        dump_dir=${data}/dump/${x}_${datasize}
        dump_feat.sh --cmd "$train_cmd" --nj 32 \
            ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x}_${datasize} ${dump_dir} || exit 1;
    done

    touch ${data}/.done_stage_1_${datasize}_sp${speed_perturb} && echo "Finish feature extranction (stage: 1)."
fi

dict=${data}/dict/${train_set}_${unit}${wp_type}${vocab}.txt; mkdir -p ${data}/dict
nlsyms=${data}/dict/nlsyms_${train_set}.txt
wp_model=${data}/dict/${train_set}_${wp_type}${vocab}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ ! -e ${data}/.done_stage_2_${datasize}_${unit}${wp_type}${vocab}_sp${speed_perturb} ]; then
    echo ============================================================================
    echo "                      Dataset preparation (stage:2)                        "
    echo ============================================================================

    echo "make a non-linguistic symbol list"
    if [ ${speed_perturb} = true ]; then
        grep sp1.0 ${data}/${train_set}/text | cut -f 2- -d " " | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    else
        cut -f 2- -d " " ${data}/${train_set}/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    fi
    cat ${nlsyms}

    if [ ${unit} = wp ]; then
        make_vocab.sh --unit ${unit} --nlsyms ${nlsyms} --speed_perturb ${speed_perturb} \
            --vocab ${vocab} --wp_type ${wp_type} --wp_model ${wp_model} \
            ${data} ${dict} ${data}/${train_set}/text || exit 1;
    else
        make_vocab.sh --unit ${unit} --nlsyms ${nlsyms} --speed_perturb ${speed_perturb} \
            ${data} ${dict} ${data}/${train_set}/text || exit 1;
    fi

    # Compute OOV rate
    if [ ${unit} = word ]; then
        mkdir -p ${data}/dict/word_count ${data}/dict/oov_rate
        echo "OOV rate:" > ${data}/dict/oov_rate/word${vocab}_${datasize}.txt
        for x in ${train_set} ${dev_set} ${test_set}; do
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
    make_dataset.sh --feat ${data}/dump/${train_set}/feats.scp --unit ${unit} --nlsyms ${nlsyms} --wp_model ${wp_model} \
        ${data}/${train_set} ${dict} > ${data}/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv || exit 1;
    for x in ${dev_set} ${test_set}; do
        dump_dir=${data}/dump/${x}_${datasize}
        make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit} --nlsyms ${nlsyms} --wp_model ${wp_model} \
            ${data}/${x} ${dict} > ${data}/dataset/${x}_${datasize}_${unit}${wp_type}${vocab}.tsv || exit 1;
    done

    touch ${data}/.done_stage_2_${datasize}_${unit}${wp_type}${vocab}_sp${speed_perturb} && echo "Finish creating dataset for ASR (stage: 2)."
fi

mkdir -p ${model}
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && [ ${speed_perturb} = false ]; then
    echo ============================================================================
    echo "                        LM Training stage (stage:3)                       "
    echo ============================================================================

    # Extend dictionary for the external text data
    if [ ! -e ${data}/.done_stage_3_${datasize}_${unit}${wp_type}${vocab} ]; then
        echo "Making dataset tsv files for LM ..."
        mkdir -p ${data}/dataset_lm
        cat ${data}/local/dict_nosp_larger/cleaned | tr "[:upper:]" "[:lower:]" > ${data}/dataset_lm/cleaned
        awk '{print "unpaired-text-"NR, $0}' ${data}/dataset_lm/cleaned > ${data}/dataset_lm/text

        update_dataset.sh --unit ${unit} --nlsyms ${nlsyms} --wp_model ${wp_model} \
            ${data}/dataset_lm/text ${dict} ${data}/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv \
            > ${data}/dataset_lm/${train_set}_${unit}${wp_type}${vocab}.tsv || exit 1;
        cp ${data}/dataset/${dev_set}_${datasize}_${unit}${wp_type}${vocab}.tsv \
            ${data}/dataset_lm/${dev_set}_${datasize}_${unit}${wp_type}${vocab}.tsv || exit 1;

        touch ${data}/.done_stage_3_${datasize}_${unit}${wp_type}${vocab} && echo "Finish creating dataset for LM (stage: 3)."
    fi

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/lm/train.py \
        --corpus wsj \
        --config ${lm_conf} \
        --n_gpus ${n_gpus} \
        --cudnn_benchmark ${benchmark} \
        --train_set ${data}/dataset_lm/${train_set}_${unit}${wp_type}${vocab}.tsv \
        --dev_set ${data}/dataset_lm/${dev_set}_${datasize}_${unit}${wp_type}${vocab}.tsv \
        --unit ${unit} \
        --nlsyms ${nlsyms} \
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
        --corpus wsj \
        --config ${conf} \
        --config2 ${conf2} \
        --n_gpus ${n_gpus} \
        --cudnn_benchmark ${benchmark} \
        --train_set ${data}/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv \
        --dev_set ${data}/dataset/${dev_set}_${datasize}_${unit}${wp_type}${vocab}.tsv \
        --eval_sets ${data}/dataset/${test_set}_${datasize}_${unit}${wp_type}${vocab}.tsv \
        --unit ${unit} \
        --nlsyms ${nlsyms} \
        --dict ${dict} \
        --wp_model ${wp_model}.model \
        --model_save_dir ${model}/asr \
        --asr_init ${asr_init} \
        --external_lm ${external_lm} \
        --teacher ${teacher} \
        --teacher_lm ${teacher_lm} \
        --stdout ${stdout} \
        --resume ${resume} || exit 1;

    echo "Finish ASR model training (stage: 4)."
fi
