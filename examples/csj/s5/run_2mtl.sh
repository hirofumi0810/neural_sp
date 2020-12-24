#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                                   CSJ                                    "
echo ============================================================================

stage=0
stop_stage=5
gpu=
benchmark=true
speed_perturb=false
stdout=false

### vocabulary
unit=wp           # word/wp/word_char
vocab=10000
wp_type=bpe       # bpe/unigram (for wordpiece)
# unit_sub1=char
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
model=/n/work2/inaguma/results/csj

### path to the model directory to resume training
resume=

### path to save preproecssed data
export data=/n/work2/inaguma/corpus/csj

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

train_set=train_nodev_${datasize}
dev_set=dev_${datasize}
test_set="eval1 eval2 eval3"
if [ ${speed_perturb} = true ]; then
    train_set=train_nodev_sp_${datasize}
    dev_set=dev_sp_${datasize}
    test_set="eval1_sp eval2_sp eval3_sp"
fi

# main
if [ ${unit} = char ]; then
    vocab=
fi
if [ ${unit} != wp ]; then
    wp_type=
fi
# sub1
if [ ${unit_sub1} = char ] || [ ${unit_sub1} = phone ]; then
    vocab_sub1=
fi
if [ ${unit_sub1} != wp ]; then
    wp_type_sub1=
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ] && [ ! -e ${data}/.done_stage_0_${datasize} ]; then
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

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ ! -e ${data}/.done_stage_1_${datasize}_sp${speed_perturb} ]; then
    echo ============================================================================
    echo "                    Feature extranction (stage:1)                          "
    echo ============================================================================

    if [ ! -e ${data}/.done_stage_1_${datasize}_spfalse ]; then
        for x in train_${datasize} eval1 eval2 eval3; do
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
    fi

    if [ ${speed_perturb} = true ]; then
        speed_perturb_3way.sh ${data} train_nodev_${datasize} ${train_set}
        cp -rf ${data}/dev_${datasize} ${data}/${dev_set}
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

# main
dict=${data}/dict/${train_set}_${unit}${wp_type}${vocab}.txt; mkdir -p ${data}/dict
wp_model=${data}/dict/${train_set}_${wp_type}${vocab}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ ! -e ${data}/.done_stage_2_${datasize}_${unit}${wp_type}${vocab}_sp${speed_perturb} ]; then
    echo ============================================================================
    echo "                      Dataset preparation (stage:2, main)                  "
    echo ============================================================================

    if [ ${unit} = wp ]; then
        make_vocab.sh --unit ${unit} --speed_perturb ${speed_perturb} --character_coverage 0.9995 \
            --vocab ${vocab} --wp_type ${wp_type} --wp_model ${wp_model} \
            ${data} ${dict} ${data}/${train_set}/text || exit 1;
    else
        # character
        make_vocab.sh --unit ${unit} --speed_perturb ${speed_perturb} --character_coverage 0.9995 \
            ${data} ${dict} ${data}/${train_set}/text || exit 1;
    fi

    # Compute OOV rate
    if [ ${unit} = word ]; then
        mkdir -p ${data}/dict/word_count ${data}/dict/oov_rate
        echo "OOV rate:" > ${data}/dict/oov_rate/word${vocab}_${datasize}.txt
        for x in ${train_set} ${dev_set} ${test_set}; do
            cut -f 2- -d " " ${data}/${x}/text | tr " " "\n" | sort | uniq -c | sort -n -k1 -r \
                > ${data}/dict/word_count/${x}_${datasize}.txt || exit 1;
            compute_oov_rate.py ${data}/dict/word_count/${x}_${datasize}.txt ${dict} ${x} \
                >> ${data}/dict/oov_rate/word${vocab}_${datasize}.txt || exit 1;
            # NOTE: speed perturbation is not considered
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

# sub1
dict_sub1=${data}/dict/${train_set}_${unit_sub1}${wp_type_sub1}${vocab_sub1}.txt
wp_model_sub1=${data}/dict/${train_set}_${wp_type_sub1}${vocab_sub1}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ ! -e ${data}/.done_stage_2_${datasize}_${unit_sub1}${wp_type_sub1}${vocab_sub1}_sp${speed_perturb} ]; then
    echo ============================================================================
    echo "                      Dataset preparation (stage:2, sub1)                  "
    echo ============================================================================

    if [ ${unit_sub1} = phone ]; then
        lexicon=${data}/local/train_${datasize}/lexicon.txt
        for x in ${train_set} ${dev_set} ${test_set}; do
            map2phone.py --text ${data}/${x}/text --lexicon ${lexicon} > ${data}/${x}/text.phone
        done
        make_vocab.sh --unit ${unit_sub1} --speed_perturb ${speed_perturb} \
            ${data} ${dict_sub1} ${data}/${train_set}/text.phone || exit 1;
    else
        make_vocab.sh --unit ${unit_sub1} --speed_perturb ${speed_perturb} --character_coverage 0.9995 \
            ${data} ${dict_sub1} ${data}/${train_set}/text || exit 1;
        # NOTE: bpe is not supported here
    fi

    echo "Making dataset tsv files for ASR ..."
    if [ ${unit_sub1} = phone ]; then
        text="text.phone"
    else
        text="text"
    fi
    for x in ${train_set} ${dev_set}; do
        dump_dir=${data}/dump/${x}
        make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit_sub1} --wp_model ${wp_model_sub1} --text ${data}/${x}/${text} \
            ${data}/${x} ${dict_sub1} > ${data}/dataset/${x}_${unit_sub1}${wp_type_sub1}${vocab_sub1}.tsv || exit 1;
    done
    for x in ${test_set}; do
        dump_dir=${data}/dump/${x}_${datasize}
        make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit_sub1} --wp_model ${wp_model_sub1} --text ${data}/${x}/${text} \
            ${data}/${x} ${dict_sub1} > ${data}/dataset/${x}_${datasize}_${unit_sub1}${wp_type_sub1}${vocab_sub1}.tsv || exit 1;
    done

    touch ${data}/.done_stage_2_${datasize}_${unit_sub1}${wp_type_sub1}${vocab_sub1}_sp${speed_perturb} && echo "Finish creating dataset for ASR (stage: 2)."
fi

mkdir -p ${model}
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo ============================================================================
    echo "                       ASR Training stage (stage:4)                        "
    echo ============================================================================

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/asr/train.py \
        --corpus csj \
        --config ${conf} \
        --config2 ${conf2} \
        --n_gpus ${n_gpus} \
        --cudnn_benchmark ${benchmark} \
        --train_set ${data}/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv \
        --train_set_sub1 ${data}/dataset/${train_set}_${unit_sub1}${wp_type_sub1}${vocab_sub1}.tsv \
        --dev_set ${data}/dataset/${dev_set}_${unit}${wp_type}${vocab}.tsv \
        --dev_set_sub1 ${data}/dataset/${dev_set}_${unit_sub1}${wp_type_sub1}${vocab_sub1}.tsv \
        --eval_sets ${data}/dataset/eval1_${datasize}_${unit}${wp_type}${vocab}.tsv \
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
