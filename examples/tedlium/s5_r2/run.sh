#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                                TEDLIUM2                                  "
echo ============================================================================

stage=0
gpu=
speed_perturb=false
spec_augment=false
stdout=false

### vocabulary
unit=wp      # word/wp/char/word_char
vocab=10000
wp_type=bpe  # bpe/unigram (for wordpiece)

#########################
# ASR configuration
#########################
asr_conf=conf/asr/rnn_seq2seq.yaml
asr_conf2=
pretrained_model=

#########################
# LM configuration
#########################
lm_conf=conf/lm/rnnlm.yaml
# lm_conf=conf/lm/gated_convlm.yaml
# lm_conf=conf/lm/transformerlm.yaml
lm_pretrained_model=

### path to save the model
model=/n/sd3/inaguma/result/tedlium2

### path to the model directory to resume training
resume=
lm_resume=

### path to save preproecssed data
export data=/n/sd3/inaguma/corpus/tedlium2

### path to original data
export db=/n/rd21/corpora_7/tedlium

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

if [ ${speed_perturb} = true ]; then
    asr_conf2=conf/asr/speed_perturb.yaml
elif [ ${spec_augment} = true ]; then
    asr_conf2=conf/asr/spec_augment.yaml
fi

if [ -z ${gpu} ]; then
    echo "Error: set GPU number." 1>&2
    echo "Usage: ./run.sh --gpu 0" 1>&2
    exit 1
fi
n_gpus=$(echo ${gpu} | tr "," "\n" | wc -l)
lm_gpu=$(echo ${gpu} | cut -d "," -f 1)

train_set=train
dev_set=dev
test_set="test"
if [ ${speed_perturb} = true ]; then
    train_set=train_sp
    dev_set=dev_sp
    test_set="test_sp"
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

if [ ${stage} -le 1 ] && [ ! -e ${data}/.done_stage_1_sp${speed_perturb} ]; then
    echo ============================================================================
    echo "                    Feature extranction (stage:1)                          "
    echo ============================================================================

    for x in train dev test; do
        steps/make_fbank.sh --nj 32 --cmd "$train_cmd" --write_utt2num_frames true \
            ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1;
    done

    if [ ${speed_perturb} = true ]; then
        # speed-perturbed
        speed_perturb_3way.sh ${data} train ${train_set}

        cp -rf ${data}/dev ${data}/${dev_set}
        cp -rf ${data}/test ${data}/${test_set}
    fi

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

    touch ${data}/.done_stage_1_sp${speed_perturb} && echo "Finish feature extranction (stage: 1)."
fi

dict=${data}/dict/${train_set}_${unit}${wp_type}${vocab}.txt; mkdir -p ${data}/dict
wp_model=${data}/dict/${train_set}_${wp_type}${vocab}
if [ ${stage} -le 2 ] && [ ! -e ${data}/.done_stage_2_${unit}${wp_type}${vocab}_sp${speed_perturb} ]; then
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
        echo "OOV rate:" > ${data}/dict/oov_rate/word${vocab}.txt
        for x in ${train_set} ${dev_set} ${test_set}; do
            if [ ${speed_perturb} = true ]; then
                cut -f 2- -d " " ${data}/${x}/text.org | tr " " "\n" | sort | uniq -c | sort -n -k1 -r \
                    > ${data}/dict/word_count/${x}.txt || exit 1;
            else
                cut -f 2- -d " " ${data}/${x}/text | tr " " "\n" | sort | uniq -c | sort -n -k1 -r \
                    > ${data}/dict/word_count/${x}.txt || exit 1;
            fi
            compute_oov_rate.py ${data}/dict/word_count/${x}.txt ${dict} ${x} \
                >> ${data}/dict/oov_rate/word${vocab}.txt || exit 1;
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

    touch ${data}/.done_stage_2_${unit}${wp_type}${vocab}_sp${speed_perturb} && echo "Finish creating dataset for ASR (stage: 2)."
fi

mkdir -p ${model}
if [ ${stage} -le 3 ] && [ ${speed_perturb} = false ]; then
    echo ============================================================================
    echo "                        LM Training stage (stage:3)                       "
    echo ============================================================================

    # Extend dictionary for the external text data
    if [ ! -e ${data}/.done_stage_3_${unit}${wp_type}${vocab} ]; then
        echo "Making dataset tsv files for LM ..."
        mkdir -p ${data}/dataset_lm

        gunzip -c ${db}/TEDLIUM_release2/LM/*.en.gz | sed 's/ <\/s>//g' | local/join_suffix.py | uniq | awk '{print "unpaired-text-"NR, $0}' > ${data}/dataset_lm/text
        # NOTE: remove exactly the same lines
        update_dataset.sh --unit ${unit} --wp_model ${wp_model} \
            ${data}/dataset_lm/text ${dict} ${data}/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv \
            > ${data}/dataset_lm/${train_set}_${unit}${wp_type}${vocab}.tsv || exit 1;
        cp ${data}/dataset/${dev_set}_${unit}${wp_type}${vocab}.tsv \
            ${data}/dataset_lm/${dev_set}_${unit}${wp_type}${vocab}.tsv || exit 1;
        cp ${data}/dataset/${test_set}_${unit}${wp_type}${vocab}.tsv \
            ${data}/dataset_lm/${test_set}_${unit}${wp_type}${vocab}.tsv || exit 1;

        touch ${data}/.done_stage_3_${unit}${wp_type}${vocab} && echo "Finish creating dataset for LM (stage: 3)."
    fi

    # NOTE: support only a single GPU for LM training
    CUDA_VISIBLE_DEVICES=${lm_gpu} ${NEURALSP_ROOT}/neural_sp/bin/lm/train.py \
        --corpus tedlium2 \
        --config ${lm_conf} \
        --n_gpus 1 \
        --train_set ${data}/dataset_lm/${train_set}_${unit}${wp_type}${vocab}.tsv \
        --dev_set ${data}/dataset_lm/${dev_set}_${unit}${wp_type}${vocab}.tsv \
        --eval_sets ${data}/dataset_lm/${test_set}_${unit}${wp_type}${vocab}.tsv \
        --unit ${unit} \
        --dict ${dict} \
        --wp_model ${wp_model}.model \
        --model_save_dir ${model}/lm \
        --pretrained_model ${lm_pretrained_model} \
        --stdout ${stdout} \
        --resume ${lm_resume} || exit 1;

    echo "Finish LM training (stage: 3)." && exit 1;
fi

if [ ${stage} -le 4 ]; then
    echo ============================================================================
    echo "                       ASR Training stage (stage:4)                        "
    echo ============================================================================

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/asr/train.py \
        --corpus tedlium2 \
        --config ${asr_conf} \
        --config2 ${asr_conf2} \
        --n_gpus ${n_gpus} \
        --train_set ${data}/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv \
        --dev_set ${data}/dataset/${dev_set}_${unit}${wp_type}${vocab}.tsv \
        --eval_sets ${data}/dataset/${test_set}_${unit}${wp_type}${vocab}.tsv \
        --unit ${unit} \
        --dict ${dict} \
        --wp_model ${wp_model}.model \
        --model_save_dir ${model}/asr \
        --pretrained_model ${pretrained_model} \
        --stdout ${stdout} \
        --resume ${resume} || exit 1;

    echo "Finish ASR model training (stage: 4)."
fi
