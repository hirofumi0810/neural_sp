#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                                   CSJ                                     "
echo ============================================================================

stage=0
gpu=
speed_perturb=false
stdout=false

### vocabulary
unit=wp      # word/wp/char/word_char
vocab=10000
wp_type=bpe  # bpe/unigram (for wordpiece)

#########################
# ASR configuration
#########################
asr_conf=conf/asr/rnn_seq2seq.yaml
pretrained_model=

# if [ ${speed_perturb} = true ]; then
#     n_epochs=20
#     convert_to_sgd_epoch=15
#     print_step=600
#     lr_decay_start_epoch=5
#     lr_decay_rate=0.8
# elif [ ${n_freq_masks} != 0 ] || [ ${n_time_masks} != 0 ]; then
#     n_epochs=50
#     convert_to_sgd_epoch=50
#     print_step=400
#     lr_decay_start_epoch=15
#     lr_decay_rate=0.9
# fi

#########################
# LM configuration
#########################
lm_conf=conf/lm/rnnlm.yaml
# lm_conf=conf/lm/gated_convlm.yaml
# lm_conf=conf/lm/transformerlm.yaml
lm_pretrained_model=

### path to save the model
model=/n/sd3/inaguma/result/csj

### path to the model directory to resume training
resume=
lm_resume=

### path to save preproecssed data
export data=/n/sd3/inaguma/corpus/csj

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
datasize=all
lm_datasize=all  # default is the same data as ASR
# NOTE: aps_other=default using "Academic lecture" and "other" data,
#       aps=using "Academic lecture" data,
#       sps=using "Academic lecture" data,
#       all_except_dialog=using All data except for "dialog" data,
#       all=using All data

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
lm_gpu=$(echo ${gpu} | cut -d "," -f 1)

train_set=train_nodev_${datasize}
dev_set=dev_${datasize}
test_set="eval1 eval2 eval3"
if [ ${speed_perturb} = true ]; then
    train_set=train_sp_nodev_${datasize}
    dev_set=dev_sp_${datasize}
    test_set="eval1_sp eval2_sp eval3_sp"
fi

if [ ${unit} = char ]; then
    vocab=
fi
if [ ${unit} != wp ]; then
    wp_type=
fi

if [ ${stage} -le 0 ] && [ ! -e ${data}/.done_stage_0_${datasize} ]; then
    echo ============================================================================
    echo "                       Data Preparation (stage:0)                          "
    echo ============================================================================

    mkdir -p ${data}
    local/csj_make_trans/csj_autorun.sh ${CSJDATATOP} ${data}/csj-data ${CSJVER} || exit 1;
    local/csj_data_prep.sh ${data}/csj-data ${datasize} || exit 1;
    for x in eval1 eval2 eval3; do
        local/csj_eval_data_prep.sh ${data}/csj-data/eval ${x} || exit 1;
    done

    # Remove <sp> and POS tag, and lowercase
    for x in train_${datasize} ${test_set}; do
        local/remove_pos.py ${data}/${x}/text | nkf -Z > ${data}/${x}/text.tmp
        mv ${data}/${x}/text.tmp ${data}/${x}/text
    done

    touch ${data}/.done_stage_0_${datasize} && echo "Finish data preparation (stage: 0)."
fi

if [ ${stage} -le 1 ] && [ ! -e ${data}/.done_stage_1_${datasize}_sp${speed_perturb} ]; then
    echo ============================================================================
    echo "                    Feature extranction (stage:1)                          "
    echo ============================================================================

    for x in train_${datasize} ${test_set}; do
        steps/make_fbank.sh --nj 32 --cmd "$train_cmd" --write_utt2num_frames true \
            ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1;
    done

    # Use the first 4k sentences from training data as dev set. (39 speakers.)
    utils/subset_data_dir.sh --first ${data}/train_${datasize} 4000 ${data}/${dev_set} || exit 1;  # 6hr 31min
    n=$[$(cat ${data}/train_${datasize}/segments | wc -l) - 4000]
    utils/subset_data_dir.sh --last ${data}/train_${datasize} ${n} ${data}/${train_set}.tmp || exit 1;

    # Finally, the full training set:
    utils/data/remove_dup_utts.sh 300 ${data}/${train_set}.tmp ${data}/${train_set} || exit 1;  # 233hr 36min
    rm -rf ${data}/*.tmp

    if [ ${speed_perturb} = true ]; then
        # speed-perturbed
        speed_perturb_3way.sh ${data} train_nodev_${datasize} ${train_set}

        cp -rf ${data}/dev ${data}/${dev_set}
        cp -rf ${data}/eval1 ${data}/eval1_sp
        cp -rf ${data}/eval2 ${data}/eval2_sp
        cp -rf ${data}/eval3 ${data}/eval3_sp
    fi

    # Compute global CMVN
    compute-cmvn-stats scp:${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark || exit 1;

    # Apply global CMVN & dump features
    dump_feat.sh --cmd "$train_cmd" --nj 80 \
        ${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${train_set} ${data}/dump/${train_set} || exit 1;
    dump_feat.sh --cmd "$train_cmd" --nj 32 \
        ${data}/${dev_set}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${dev_set} ${data}/dump/${dev_set} || exit 1;
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
    make_dataset.sh --feat ${data}/dump/${train_set}/feats.scp --unit ${unit} --wp_model ${wp_model} \
        ${data}/${train_set} ${dict} > ${data}/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv || exit 1;
    make_dataset.sh --feat ${data}/dump/${dev_set}/feats.scp --unit ${unit} --wp_model ${wp_model} \
        ${data}/${dev_set} ${dict} > ${data}/dataset/${dev_set}_${unit}${wp_type}${vocab}.tsv || exit 1;
    for x in ${test_set}; do
        dump_dir=${data}/dump/${x}_${datasize}
        make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit} --wp_model ${wp_model} \
            ${data}/${x} ${dict} > ${data}/dataset/${x}_${datasize}_${unit}${wp_type}${vocab}.tsv || exit 1;
    done

    touch ${data}/.done_stage_2_${datasize}_${unit}${wp_type}${vocab}_sp${speed_perturb} && echo "Finish creating dataset for ASR (stage: 2)."
fi

mkdir -p ${model}
if [ ${stage} -le 3 ]; then
    echo ============================================================================
    echo "                        LM Training stage (stage:3)                       "
    echo ============================================================================

    if [ ! -e ${data}/.done_stage_3_${datasize}${lm_datasize}_${unit}${wp_type}${vocab} ]; then
        [ ! -e ${data}/.done_stage_1_${datasize}_sp${speed_perturb} ] && echo "run ./run.sh --datasize ${lm_datasize} first" && exit 1;

        echo "Making dataset tsv files for LM ..."
        mkdir -p ${data}/dataset_lm
        if [ ${lm_datasize} = ${datasize} ]; then
            cp ${data}/dataset/train_nodev_${lm_datasize}_${unit}${wp_type}${vocab}.tsv \
                ${data}/dataset_lm/train_lm${lm_datasize}_asr${datasize}_${unit}${wp_type}${vocab}.tsv || exit 1;
        else
            make_dataset.sh --unit ${unit} --wp_model ${wp_model} \
                ${data}/train_nodev_${lm_datasize} ${dict} > ${data}/dataset_lm/train_lm${lm_datasize}_asr${datasize}_${unit}${wp_type}${vocab}.tsv || exit 1;
        fi
        for x in dev ${test_set}; do
            cp ${data}/dataset/${x}_${lm_datasize}_${unit}${wp_type}${vocab}.tsv \
                ${data}/dataset_lm/${x}_lm${lm_datasize}_asr${datasize}_${unit}${wp_type}${vocab}.tsv || exit 1;
        done

        touch ${data}/.done_stage_3_${datasize}${lm_datasize}_${unit}${wp_type}${vocab} && echo "Finish creating dataset for LM (stage: 3)."
    fi

    lm_test_set="${data}/dataset_lm/eval1_lm${lm_datasize}_asr${datasize}_${unit}${wp_type}${vocab}.tsv \
                 ${data}/dataset_lm/eval2_lm${lm_datasize}_asr${datasize}_${unit}${wp_type}${vocab}.tsv \
                 ${data}/dataset_lm/eval3_lm${lm_datasize}_asr${datasize}_${unit}${wp_type}${vocab}.tsv"

    # NOTE: support only a single GPU for LM training
    CUDA_VISIBLE_DEVICES=${lm_gpu} ${NEURALSP_ROOT}/neural_sp/bin/lm/train.py \
        --corpus csj \
        --config ${lm_conf} \
        --n_gpus 1 \
        --train_set ${data}/dataset_lm/train_lm${lm_datasize}_asr${datasize}_${unit}${wp_type}${vocab}.tsv \
        --dev_set ${data}/dataset_lm/dev_lm${lm_datasize}_asr${datasize}_${unit}${wp_type}${vocab}.tsv \
        --eval_sets ${lm_test_set} \
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
        --corpus csj \
        --config ${asr_conf} \
        --n_gpus ${n_gpus} \
        --train_set ${data}/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv \
        --dev_set ${data}/dataset/${dev_set}_${unit}${wp_type}${vocab}.tsv \
        --eval_sets ${data}/dataset/eval1_${datasize}_${unit}${wp_type}${vocab}.tsv \
        --unit ${unit} \
        --dict ${dict} \
        --wp_model ${wp_model}.model \
        --model_save_dir ${model}/asr \
        --pretrained_model ${pretrained_model} \
        --stdout ${stdout} \
        --resume ${resume} || exit 1;

    echo "Finish ASR model training (stage: 4)."
fi
