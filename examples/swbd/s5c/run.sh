#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                              Switchboard                                  "
echo ============================================================================

stage=0
stop_stage=5
gpu=
speed_perturb=true
specaug=false
stdout=false

### vocabulary
unit=wp      # word/wp/char/word_char/phone
vocab=10000
wp_type=bpe  # bpe/unigram (for wordpiece)

#########################
# ASR configuration
#########################
conf=conf/asr/blstm_las.yaml
conf2=
asr_init=
lm_init=

#########################
# LM configuration
#########################
lm_conf=conf/lm/rnnlm.yaml

### path to save the model
model=/n/work1/inaguma/results/swbd

### path to the model directory to resume training
resume=
lm_resume=

### path to save preproecssed data
export data=/n/work1/inaguma/corpus/swbd

### path to original data
SWBD_AUDIOPATH=/n/rd21/corpora_7/swb
EVAL2000_AUDIOPATH=/n/rd21/corpora_7/hub5_english/LDC2002S09
EVAL2000_TRANSPATH=/n/rd21/corpora_7/hub5_english/LDC2002T43
RT03_PATH=
FISHER_PATH=/n/rd7/fisher_english

### data size
datasize=swbd
lm_datasize=fisher_swbd

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

if [ ${speed_perturb} = true ] || [ ${specaug} = true ]; then
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
dev_set=dev
test_set="eval2000"
if [ ${speed_perturb} = true ]; then
    train_set=train_nodev_sp_${datasize}
    dev_set=dev_sp
    test_set="eval2000_sp"
fi

if [ ${unit} = char ] || [ ${unit} = phone ]; then
    vocab=
fi
if [ ${unit} != wp ]; then
    wp_type=
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ] && [ ! -e ${data}/.done_stage_0 ]; then
    echo ============================================================================
    echo "                       Data Preparation (stage:0)                          "
    echo ============================================================================

    # prepare swbd data and put it under data/train_swbd
    local/swbd1_data_download.sh ${SWBD_AUDIOPATH} || exit 1;
    local/swbd1_prepare_dict.sh || exit 1;
    local/swbd1_data_prep.sh ${SWBD_AUDIOPATH} || exit 1;

    # prepare fisher data and put it under data/train_fisher
    local/fisher_data_prep.sh ${FISHER_PATH}
    local/fisher_swbd_prepare_dict.sh
    utils/fix_data_dir.sh ${data}/train_fisher

    # nomalization
    cp ${data}/train_fisher/text ${data}/train_fisher/text.tmp.0
    cut -f 2- -d " " ${data}/train_fisher/text.tmp.0 | \
        sed -e 's/\[laughter\]-/[laughter]/g' | \
        sed -e 's/\[noise\]-/[noise]/g' > ${data}/train_fisher/text.tmp.1
    paste -d " " <(cut -f 1 -d " " ${data}/train_fisher/text.tmp.0) \
        <(cat ${data}/train_fisher/text.tmp.1) > ${data}/train_fisher/text
    rm ${data}/train_fisher/text.tmp*

    # eval2000
    local/eval2000_data_prep.sh ${EVAL2000_AUDIOPATH} ${EVAL2000_TRANSPATH} || exit 1;
    [ ! -z ${RT03_PATH} ] && local/rt03_data_prep.sh ${RT03_PATH}

    # upsample audio from 8k to 16k
    # for x in train_swbd train_fisher eval2000 rt03; do
    #     sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" ${data}/${x}/wav.scp
    # done

    touch ${data}/.done_stage_0 && echo "Finish data preparation (stage: 0)."
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ ! -e ${data}/.done_stage_1_${datasize}_sp${speed_perturb} ]; then
    echo ============================================================================
    echo "                    Feature extranction (stage:1)                          "
    echo ============================================================================

    if [ ! -e ${data}/.done_stage_1_${datasize}_spfalse ]; then
        for x in train_swbd eval2000; do
            steps/make_fbank.sh --nj 32 --cmd "$train_cmd" --write_utt2num_frames true \
                ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1;
            utils/fix_data_dir.sh ${data}/${x}
        done

        # Use the first 4k sentences as dev set.
        utils/subset_data_dir.sh --first ${data}/train_swbd 4000 ${data}/${dev_set} || exit 1;  # 5hr 6min
        n=$[$(cat ${data}/train_swbd/segments | wc -l) - 4000]
        utils/subset_data_dir.sh --last ${data}/train_swbd ${n} ${data}/${train_set}.tmp || exit 1;

        # Finally, the full training set:
        utils/data/remove_dup_utts.sh 300 ${data}/${train_set}.tmp ${data}/train_nodev_swbd || exit 1;  # 286hr
        rm -rf ${data}/*.tmp

        if [ ${datasize} = fisher_swbd ]; then
            steps/make_fbank.sh --nj 32 --cmd "$train_cmd" --write_utt2num_frames true \
                ${data}/train_fisher ${data}/log/make_fbank/train_fisher ${data}/fbank || exit 1;
            utils/combine_data.sh --extra_files "utt2num_frames" ${data}/${train_set} ${data}/train_nodev_swbd ${data}/train_fisher || exit 1;
        fi
    fi

    if [ ${speed_perturb} = true ]; then
        speed_perturb_3way.sh ${data} train_nodev_${datasize} ${train_set}
        cp -rf ${data}/dev ${data}/${dev_set}
        cp -rf ${data}/eval2000 ${data}/eval2000_sp
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
        grep sp1.0 ${data}/${train_set}/text | cut -f 2- -d " " | grep -o -P '\[[^\]]*\]' | sort | uniq > ${nlsyms}
    else
        cut -f 2- -d " " ${data}/${train_set}/text | grep -o -P '\[[^\]]*\]' | sort | uniq > ${nlsyms}
    fi
    cat ${nlsyms}

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
        spm_train --user_defined_symbols=$(cat ${nlsyms} | tr "\n" ",") --input=${data}/dict/input.txt --vocab_size=${vocab} \
            --model_type=${wp_type} --model_prefix=${wp_model} --input_sentence_size=100000000 --character_coverage=1.0
        spm_encode --model=${wp_model}.model --output_format=piece < ${data}/dict/input.txt | tr ' ' '\n' | \
            sort | uniq -c | sort -n -k1 -r | sed -e 's/^[ ]*//g' | cut -d " " -f 2 | grep -v '^\s*$' | awk -v offset=${offset} '{print $1 " " NR+offset}' >> ${dict}
    elif [ ${unit} = phone ]; then
        map_lexicon.sh ${data}/${train_set} ${data}/local/dict_nosp/lexicon.txt > ${data}/${train_set}/text.phone
        map_lexicon.sh ${data}/${dev_set} ${data}/local/dict_nosp/lexicon.txt > ${data}/${dev_set}/text.phone
        text2dict.py ${data}/${train_set}/text.phone --unit ${unit} --nlsyms ${nlsyms} --speed_perturb ${speed_perturb} | \
            awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict} || exit 1;
    else
        text2dict.py ${data}/${train_set}/text --unit ${unit} --vocab ${vocab} --nlsyms ${nlsyms} --speed_perturb ${speed_perturb} | \
            awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict} || exit 1;
    fi
    echo "vocab size:" $(cat ${dict} | wc -l)

    # normalize eval2000
    # 1) convert upper to lower
    # 2) remove tags (%AH) (%HESITATION) (%UH)
    # 3) remove <B_ASIDE> <E_ASIDE>
    # 4) remove "(" or ")"
    paste -d " " <(awk '{print $1}' ${data}/${test_set}/text) <(cat ${data}/${test_set}/text | cut -f 2- -d " " | awk '{ print tolower($0) }' | \
        perl -pe 's| \(\%.*\)||g' | perl -pe 's| \<.*\>||g' | sed -e "s/(//g" -e "s/)//g" | sed -e 's/\s\+/ /g') \
        > ${data}/${test_set}/text.tmp
    mv ${data}/${test_set}/text.tmp ${data}/${test_set}/text

    grep -v '^en_' ${data}/${test_set}/text > ${data}/${test_set}/text.swbd
    grep -v '^sw_' ${data}/${test_set}/text > ${data}/${test_set}/text.ch

    # Compute OOV rate
    if [ ${unit} = word ]; then
        mkdir -p ${data}/dict/word_count ${data}/dict/oov_rate
        echo "OOV rate:" > ${data}/dict/oov_rate/word${vocab}_${datasize}.txt
        for x in ${train_set} ${dev_set}; do
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
        for set in "swbd" "ch"; do
            cut -f 2- -d " " ${data}/${test_set}/text.${set} | tr " " "\n" | sort | uniq -c | sort -n -k1 -r \
                > ${data}/dict/word_count/${test_set}_${set}.txt || exit 1;
            compute_oov_rate.py ${data}/dict/word_count/${test_set}_${set}.txt ${dict} ${test_set}_${set} \
                >> ${data}/dict/oov_rate/word${vocab}_${datasize}.txt || exit 1;
        done
        cat ${data}/dict/oov_rate/word${vocab}_${datasize}.txt
    fi

    echo "Making dataset tsv files for ASR ..."
    mkdir -p ${data}/dataset
    if [ ${unit} = phone ]; then
        make_dataset.sh --feat ${data}/dump/${train_set}/feats.scp --unit ${unit} --nlsyms ${nlsyms} --text ${data}/${x}/text.phone \
            ${data}/${train_set} ${dict} > ${data}/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv || exit 1;
    else
        make_dataset.sh --feat ${data}/dump/${train_set}/feats.scp --unit ${unit} --nlsyms ${nlsyms} --wp_model ${wp_model} \
            ${data}/${train_set} ${dict} > ${data}/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv || exit 1;
    fi
    for x in ${dev_set} ${test_set}; do
        dump_dir=${data}/dump/${x}_${datasize}
        if [ ${unit} = phone ] && [ ${x} != ${test_set} ]; then
            make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit} --nlsyms ${nlsyms} --text ${data}/${x}/text.phone  \
                ${data}/${x} ${dict} > ${data}/dataset/${x}_${datasize}_${unit}${wp_type}${vocab}.tsv || exit 1;
        else
            make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit} --nlsyms ${nlsyms} --wp_model ${wp_model} \
                ${data}/${x} ${dict} > ${data}/dataset/${x}_${datasize}_${unit}${wp_type}${vocab}.tsv || exit 1;
        fi
    done

    touch ${data}/.done_stage_2_${datasize}_${unit}${wp_type}${vocab}_sp${speed_perturb} && echo "Finish creating dataset for ASR (stage: 2)."
fi

mkdir -p ${model}
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && [ ${speed_perturb} = false ]; then
    echo ============================================================================
    echo "                        LM Training stage (stage:3)                       "
    echo ============================================================================

    if [ ! -e ${data}/.done_stage_3_${datasize}${lm_datasize}_${unit}${wp_type}${vocab} ]; then
        echo "Making dataset tsv files for LM ..."
        mkdir -p ${data}/dataset_lm
        if [ ${datasize} = swbd ] && [ ${lm_datasize} = fisher_swbd ]; then
            update_dataset.sh --unit ${unit} --nlsyms ${nlsyms} --wp_model ${wp_model} \
                ${data}/train_fisher/text ${dict} ${data}/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv \
                    > ${data}/dataset_lm/train_nodev_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv || exit 1;
        else
            cp ${data}/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv \
                ${data}/dataset_lm/train_nodev_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv || exit 1;
        fi
        cp ${data}/dataset/${dev_set}_${datasize}_${unit}${wp_type}${vocab}.tsv \
            ${data}/dataset_lm/dev_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv || exit 1;
        for x in ${test_set}; do
            make_dataset.sh --unit ${unit} --nlsyms ${nlsyms} --wp_model ${wp_model} --text ${data}/${test_set}/text.swbd \
                ${data}/${x} ${dict} > ${data}/dataset_lm/${x}_swbd_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv || exit 1;
            make_dataset.sh --unit ${unit} --nlsyms ${nlsyms} --wp_model ${wp_model} --text ${data}/${test_set}/text.ch \
                ${data}/${x} ${dict} > ${data}/dataset_lm/${x}_ch_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv || exit 1;
        done

        touch ${data}/.done_stage_3_${datasize}${lm_datasize}_${unit}${wp_type}${vocab} && echo "Finish creating dataset for LM (stage: 3)."
    fi

    lm_test_set="${data}/dataset_lm/${test_set}_swbd_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv \
                 ${data}/dataset_lm/${test_set}_ch_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv"

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/lm/train.py \
        --corpus swbd \
        --config ${lm_conf} \
        --n_gpus ${n_gpus} \
        --train_set ${data}/dataset_lm/train_nodev_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv \
        --dev_set ${data}/dataset_lm/dev_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv \
        --eval_sets ${lm_test_set} \
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
        --corpus swbd \
        --config ${conf} \
        --config2 ${conf2} \
        --n_gpus ${n_gpus} \
        --train_set ${data}/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv \
        --dev_set ${data}/dataset/${dev_set}_${datasize}_${unit}${wp_type}${vocab}.tsv \
        --unit ${unit} \
        --nlsyms ${nlsyms} \
        --dict ${dict} \
        --wp_model ${wp_model}.model \
        --model_save_dir ${model}/asr \
        --asr_init ${asr_init} \
        --lm_init ${lm_init} \
        --stdout ${stdout} \
        --resume ${resume} || exit 1;

    echo "Finish ASR model training (stage: 4)."
fi
