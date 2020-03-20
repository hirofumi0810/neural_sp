#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                                   CSJ                                    "
echo ============================================================================

stage=0
stop_stage=5
gpu=
speed_perturb=true  # default
specaug=false
stdout=false

### vocabulary
unit=wp      # word/wp/char/word_char
vocab=10000
wp_type=bpe  # bpe/unigram (for wordpiece)

### path to save the model
model=/n/work1/inaguma/results/csj

### path to save preproecssed data
export data=/n/work1/inaguma/corpus/csj

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

train_set=train_nodev_${datasize}
test_set="eval1_streaming eval2_streaming eval3_streaming"
if [ ${speed_perturb} = true ]; then
    train_set=train_nodev_sp_${datasize}
    test_set="eval1_streaming_sp eval2_streaming_sp eval3_streaming_sp"
fi

if [ ${unit} = char ]; then
    vocab=
fi
if [ ${unit} != wp ]; then
    wp_type=
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ] && [ ! -e ${data}/.done_streaming_stage_0_${datasize} ]; then
    echo ============================================================================
    echo "                       Data Preparation (stage:0)                          "
    echo ============================================================================

    for x in eval1 eval2 eval3; do
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

    touch ${data}/.done_streaming_stage_0_${datasize} && echo "Finish data preparation (stage: 0)."
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ ! -e ${data}/.done_streaming_stage_1_${datasize}_sp${speed_perturb} ]; then
    echo ============================================================================
    echo "                    Feature extranction (stage:1)                          "
    echo ============================================================================

    if [ ! -e ${data}/.done_streaming_stage_1_${datasize}_spfalse ]; then
        for x in eval1_streaming eval2_streaming eval3_streaming; do
            steps/make_fbank.sh --nj 4 --cmd "$train_cmd" --write_utt2num_frames true \
                ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1;
        done
    fi

    if [ ${speed_perturb} = true ]; then
        cp -rf ${data}/dev_streaming_${datasize} ${data}/${dev_set}
        cp -rf ${data}/eval1_streaming ${data}/eval1_streaming_sp
        cp -rf ${data}/eval2_streaming ${data}/eval2_streaming_sp
        cp -rf ${data}/eval3_streaming ${data}/eval3_streaming_sp
    fi

    # Apply global CMVN & dump features
    for x in ${test_set}; do
        dump_dir=${data}/dump/${x}_${datasize}
        dump_feat.sh --cmd "$train_cmd" --nj 4 \
            ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x}_${datasize} ${dump_dir} || exit 1;
    done

    touch ${data}/.done_streaming_stage_1_${datasize}_sp${speed_perturb} && echo "Finish feature extranction (stage: 1)."
fi

dict=${data}/dict/${train_set}_${unit}${wp_type}${vocab}.txt
wp_model=${data}/dict/${train_set}_${wp_type}${vocab}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ ! -e ${data}/.done_streaming_stage_2_${datasize}_${unit}${wp_type}${vocab}_sp${speed_perturb} ]; then
    echo ============================================================================
    echo "                      Dataset preparation (stage:2)                        "
    echo ============================================================================

    echo "Making dataset tsv files for ASR ..."
    for x in ${test_set}; do
        dump_dir=${data}/dump/${x}_${datasize}
        make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit} --wp_model ${wp_model} \
            ${data}/${x} ${dict} > ${data}/dataset/${x}_${datasize}_${unit}${wp_type}${vocab}.tsv || exit 1;
    done

    touch ${data}/.done_streaming_stage_2_${datasize}_${unit}${wp_type}${vocab}_sp${speed_perturb} && echo "Finish creating dataset for ASR (stage: 2)."
fi
