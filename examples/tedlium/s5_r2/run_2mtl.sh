#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                                TEDLIUM2                                  "
echo ============================================================================

# NOTE: speed perturbation is adopted by default

stage=0
stop_stage=5
gpu=
benchmark=true
stdout=false

### vocabulary
unit=wp      # word/wp/char/word_char
vocab=10000
wp_type=bpe  # bpe/unigram (for wordpiece)
unit_sub1=phone
wp_type_sub1=bpe  # bpe/unigram (for wordpiece)
vocab_sub1=

#########################
# ASR configuration
#########################
conf=conf/asr/blstm_las_2mtl.yaml
conf2=
asr_init=

### path to save the model
model=/n/work2/inaguma/results/tedlium2

### path to the model directory to resume training
resume=
lm_resume=

### path to save preproecssed data
export data=/n/work2/inaguma/corpus/tedlium2

### path to original data
export db=/n/rd21/corpora_7/tedlium

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

# main
if [ ${unit} = char ]; then
    vocab=
fi
if [ ${unit} != wp ]; then
    wp_type=
fi
# sub1
if [ ${unit_sub1} = char ]; then
    vocab_sub1=
fi
if [ ${unit_sub1} != wp ]; then
    wp_type_sub1=
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ] && [ ! -e ${data}/.done_stage_0 ]; then
    echo ============================================================================
    echo "                       Data Preparation (stage:0)                          "
    echo ============================================================================

    local/download_data.sh
    local/prepare_data.sh
    for dset in dev test train; do
        utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 ${data}/${dset}.orig ${data}/${dset}
    done
    local/prepare_dict.sh
    # utils/prepare_lang.sh ${data}/local/dict_nosp "<unk>" ${data}/local/lang_nosp ${data}/lang_nosp
    # local/ted_download_lm.sh
    # local/ted_train_lm.sh
    # local/format_lms.sh

    touch ${data}/.done_stage_0 && echo "Finish data preparation (stage: 0)."
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ ! -e ${data}/.done_stage_1_sptrue ]; then
    echo ============================================================================
    echo "                    Feature extranction (stage:1)                          "
    echo ============================================================================

    if [ ! -e ${data}/.done_stage_1_spfalse ]; then
        for x in train dev test; do
            steps/make_fbank.sh --nj 32 --cmd "$train_cmd" --write_utt2num_frames true \
                ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1;
        done
    fi

    # speed perturbation
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

    touch ${data}/.done_stage_1_sptrue && echo "Finish feature extranction (stage: 1)."
fi

# main
dict=${data}/dict/${train_set}_${unit}${wp_type}${vocab}.txt; mkdir -p ${data}/dict
wp_model=${data}/dict/${train_set}_${wp_type}${vocab}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ ! -e ${data}/.done_stage_2_${unit}${wp_type}${vocab}_sptrue ]; then
    echo ============================================================================
    echo "                      Dataset preparation (stage:2, main)                  "
    echo ============================================================================

    if [ ${unit} = wp ]; then
        make_vocab.sh --unit ${unit} --speed_perturb true \
            --vocab ${vocab} --wp_type ${wp_type} --wp_model ${wp_model} \
            ${data} ${dict} ${data}/${train_set}/text || exit 1;
    else
        # character
        make_vocab.sh --unit ${unit} --speed_perturb true \
            ${data} ${dict} ${data}/${train_set}/text || exit 1;
    fi

    # Compute OOV rate
    if [ ${unit} = word ]; then
        mkdir -p ${data}/dict/word_count ${data}/dict/oov_rate
        echo "OOV rate:" > ${data}/dict/oov_rate/word${vocab}.txt
        for x in ${train_set} ${dev_set} ${test_set}; do
            cut -f 2- -d " " ${data}/${x}/text | tr " " "\n" | sort | uniq -c | sort -n -k1 -r \
                > ${data}/dict/word_count/${x}.txt || exit 1;
            compute_oov_rate.py ${data}/dict/word_count/${x}.txt ${dict} ${x} \
                >> ${data}/dict/oov_rate/word${vocab}.txt || exit 1;
            # NOTE: speed perturbation is not considered
        done
        cat ${data}/dict/oov_rate/word${vocab}.txt
    fi

    echo "Making dataset tsv files for ASR ..."
    mkdir -p ${data}/dataset
    for x in ${train_set} ${dev_set} ${test_set}; do
        dump_dir=${data}/dump/${x}
        make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit} --wp_model ${wp_model} \
            ${data}/${x} ${dict} > ${data}/dataset/${x}_${unit}${wp_type}${vocab}.tsv || exit 1;
    done

    touch ${data}/.done_stage_2_${unit}${wp_type}${vocab}_sptrue && echo "Finish creating dataset for ASR (stage: 2)."
fi

# sub1
dict_sub1=${data}/dict/${train_set}_${unit_sub1}${wp_type_sub1}${vocab_sub1}.txt
wp_model_sub1=${data}/dict/${train_set}_${wp_type_sub1}${vocab_sub1}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ ! -e ${data}/.done_stage_2_${unit_sub1}${wp_type_sub1}${vocab_sub1}_sptrue ]; then
    echo ============================================================================
    echo "                      Dataset preparation (stage:2, sub1)                  "
    echo ============================================================================

    if [ ${unit_sub1} = phone ]; then
        echo "Making a dictionary..."
        echo "<unk> 1" > ${dict_sub1}  # <unk> must be 1, 0 will be used for "blank" in CTC
        echo "<eos> 2" >> ${dict_sub1}  # <sos> and <eos> share the same index
        echo "<pad> 3" >> ${dict_sub1}
        offset=$(cat ${dict_sub1} | wc -l)
        lexicon=${data}/local/dict_nosp/lexicon.txt
        map2phone.py --text ${data}/${train_set}/text --lexicon ${lexicon} --noise NSN > ${data}/${train_set}/text.phone
        map2phone.py --text ${data}/${dev_set}/text --lexicon ${lexicon} --noise NSN > ${data}/${dev_set}/text.phone
        map2phone.py --text ${data}/${test_set}/text --lexicon ${lexicon} --noise NSN > ${data}/${test_set}/text.phone
        text2dict.py ${data}/${train_set}/text.phone --unit ${unit_sub1} --speed_perturb true | \
            awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict_sub1} || exit 1;
    else
        make_vocab.sh --unit ${unit_sub1} --speed_perturb true \
            ${data} ${dict_sub1} ${data}/${train_set}/text || exit 1;
        # NOTE: bpe is not supported here
    fi

    echo "Making dataset tsv files for ASR ..."
    for x in ${train_set} ${dev_set} ${test_set}; do
        dump_dir=${data}/dump/${x}
        if [ ${unit_sub1} = phone ]; then
            make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit_sub1} --text ${data}/${x}/text.phone \
                ${data}/${x} ${dict_sub1} > ${data}/dataset/${x}_${unit_sub1}${wp_type_sub1}${vocab_sub1}.tsv || exit 1;
        else
            make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit_sub1} --wp_model ${wp_model_sub1} \
                ${data}/${x} ${dict_sub1} > ${data}/dataset/${x}_${unit_sub1}${wp_type_sub1}${vocab_sub1}.tsv || exit 1;
        fi
    done

    touch ${data}/.done_stage_2_${unit_sub1}${wp_type_sub1}${vocab_sub1}_sptrue && echo "Finish creating dataset for ASR (stage: 2)."
fi

mkdir -p ${model}
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo ============================================================================
    echo "                       ASR Training stage (stage:4)                        "
    echo ============================================================================

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/asr/train.py \
        --corpus tedlium2 \
        --config ${conf} \
        --config2 ${conf2} \
        --n_gpus ${n_gpus} \
        --cudnn_benchmark ${benchmark} \
        --train_set ${data}/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv \
        --train_set_sub1 ${data}/dataset/${train_set}_${unit_sub1}${wp_type_sub1}${vocab_sub1}.tsv \
        --dev_set ${data}/dataset/${dev_set}_${unit}${wp_type}${vocab}.tsv \
        --dev_set_sub1 ${data}/dataset/${dev_set}_${unit_sub1}${wp_type_sub1}${vocab_sub1}.tsv \
        --eval_sets ${data}/dataset/${test_set}_${unit}${wp_type}${vocab}.tsv \
        --unit ${unit} \
        --unit_sub1 ${unit_sub1} \
        --dict ${dict} \
        --dict_sub1 ${dict_sub1} \
        --wp_model ${wp_model}.model \
        --wp_model_sub1 ${wp_model_sub1}.model \
        --model_save_dir ${model}/asr \
        --asr_init ${asr_init} \
        --stdout ${stdout} \
        --resume ${resume} || exit 1;

    echo "Finish ASR model training (stage: 4)."
fi
