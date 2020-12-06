#!/bin/bash

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                                TEDLIUM2                                  "
echo ============================================================================

# NOTE: speed perturbation is adopted by default

stage=-1
stop_stage=5
gpu=
stdout=false

### vocabulary
unit=wp      # word/wp/char/word_char
vocab=10000
wp_type=bpe  # bpe/unigram (for wordpiece)

### path to save the model
model=/n/work2/inaguma/results/tedlium2

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

train_set=train_sp
dev_set=dev_streaming_sp
test_set="test_streaming_sp"

if [ ${unit} = char ]; then
    vocab=
fi
if [ ${unit} != wp ]; then
    wp_type=
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo ============================================================================
    echo "                       Data Download (stage:-1)                            "
    echo ============================================================================

    for part in dev2010 tst2010 tst2013 tst2014 tst2015; do
        local/download_and_untar.sh ${data} ${part}
    done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ] && [ ! -e ${data}/.done_streaming_stage_0 ]; then
    echo ============================================================================
    echo "                       Data Preparation (stage:0)                          "
    echo ============================================================================

    # for part in dev2010 tst2010 tst2013 tst2014 tst2015; do
    #     local/data_prep_eval.sh ${data} ${part}
    # done
    # exit 1

    for x in dev test; do
        mkdir -p ${data}/${x}_streaming
        cp -rf ${data}/${x}/wav.scp ${data}/${x}_streaming
        cp -rf ${data}/${x}/text ${data}/${x}_streaming

        awk '{
            segment=$1; split(segment,S,"[ ]");
            spkid=S[1]; print $1 " " $1
        }' ${data}/${x}_streaming/wav.scp | uniq | sort > ${data}/${x}_streaming/utt2spk
        cat ${data}/${x}_streaming/utt2spk | utils/utt2spk_to_spk2utt.pl | sort > ${data}/${x}_streaming/spk2utt

        # Concatenate references for the same speaker
        concat_ref.py ${data}/${x}_streaming/text ${data}/${x}/utt2spk > ${data}/${x}_streaming/text.sep
        cat ${data}/${x}_streaming/text.sep | sed -e 's/ <eos> / /g' > ${data}/${x}_streaming/text
    done

    touch ${data}/.done_streaming_stage_0 && echo "Finish data preparation (stage: 0)."
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ ! -e ${data}/.done_streaming_stage_1_sptrue ]; then
    echo ============================================================================
    echo "                    Feature extranction (stage:1)                          "
    echo ============================================================================

    if [ ! -e ${data}/.done_streaming_stage_1_spfalse ]; then
        for x in dev_streaming test_streaming; do
            steps/make_fbank.sh --nj 4 --cmd "$train_cmd" --write_utt2num_frames true \
                ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1;
        done
    fi

    cp -rf ${data}/dev_streaming ${data}/${dev_set}
    cp -rf ${data}/test_streaming ${data}/${test_set}

    # Apply global CMVN & dump features
    for x in ${dev_set} ${test_set}; do
        dump_dir=${data}/dump/${x}
        dump_feat.sh --cmd "$train_cmd" --nj 4 \
            ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x} ${dump_dir} || exit 1;
    done

    touch ${data}/.done_streaming_stage_1_sptrue && echo "Finish feature extranction (stage: 1)."
fi

dict=${data}/dict/${train_set}_${unit}${wp_type}${vocab}.txt
wp_model=${data}/dict/${train_set}_${wp_type}${vocab}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ ! -e ${data}/.done_streaming_stage_2_${unit}${wp_type}${vocab}_sptrue ]; then
    echo ============================================================================
    echo "                      Dataset preparation (stage:2)                        "
    echo ============================================================================

    echo "Making dataset tsv files for ASR ..."
    for x in ${dev_set} ${test_set}; do
        dump_dir=${data}/dump/${x}
        make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit} --wp_model ${wp_model} \
            ${data}/${x} ${dict} > ${data}/dataset/${x}_${unit}${wp_type}${vocab}.tsv || exit 1;
    done

    touch ${data}/.done_streaming_stage_2_${unit}${wp_type}${vocab}_sptrue && echo "Finish creating dataset for ASR (stage: 2)."
fi
