#!/usr/bin/env bash

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                                  AMI                                     "
echo ============================================================================

stage=0
stop_stage=5
gpu=
benchmark=true
deterministic=false
pin_memory=false
speed_perturb=true  # default
stdout=false
wandb_id=""
corpus=ami

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

#########################
# LM configuration
#########################
lm_conf=conf/lm/rnnlm.yaml

### path to save the model
model=/n/work2/inaguma/results/${corpus}

### path to the model directory to resume training
resume=
lm_resume=

### path to save preproecssed data
export data=/n/work2/inaguma/corpus/${corpus}

# You may set 'mic' to:
#  ihm [individual headset mic- the default which gives best results]
#  sdm1 [single distant microphone- the current script allows you only to select
#        the 1st of 8 microphones]
#  mdm8 [multiple distant microphones-- currently we only support averaging over
#       the 8 source microphones].
# ... by calling this script as, for example,
# ./run.sh --mic sdm1
# ./run.sh --mic mdm8
# mic=ihm
mic=sdm1

### path to download data
AMI_DIR=/n/work2/inaguma/corpus/ami/amicorpus
SWBD_AUDIOPATH=/n/rd21/corpora_7/swb
FISHER_PATH=/n/rd7/fisher_english

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

base_mic=$(echo $mic | sed 's/[0-9]//g') # sdm, ihm or mdm
nmics=$(echo $mic | sed 's/[a-z]//g') # e.g. 8 for mdm8.

set -e
set -u
set -o pipefail

train_set=train_${mic}
dev_set=dev_${mic}
test_set=eval_${mic}
if [ ${speed_perturb} = true ]; then
    train_set=train_sp_${mic}
    dev_set=dev_sp_${mic}
    test_set=eval_sp_${mic}
fi

if [ ${unit} = char ]; then
    vocab=
fi
if [ ${unit} != wp ]; then
    wp_type=
fi

use_wandb=false
if [ ! -z ${wandb_id} ]; then
    use_wandb=true
    wandb login ${wandb_id}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ] && [ ! -e ${data}/.done_stage_0_${mic} ]; then
    echo ============================================================================
    echo "                       Data Preparation (stage:0)                          "
    echo ============================================================================

    # Download AMI corpus, You need around 130GB of free space to get whole data ihm+mdm,
    if [ -d $AMI_DIR ] && ! touch $AMI_DIR/.foo 2>/dev/null; then
        echo "$0: directory $AMI_DIR seems to exist and not be owned by you."
        echo " ... Assuming the data does not need to be downloaded.  Please use --stage 1 or more."
        exit 1
    fi
    if [ -e data/local/downloads/wget_${mic}.sh ]; then
        echo "data/local/downloads/wget_${mic}.sh already exists, better quit than re-download... (use --stage N)"
        exit 1
    fi
    # local/ami_download.sh ${mic} $AMI_DIR
    local/ami_text_prep.sh ${data}/local/downloads
    local/ami_prepare_dict.sh

    if [ "$base_mic" == "mdm" ]; then
        PROCESSED_AMI_DIR=$AMI_DIR/beamformed
        if [ $stage -le 1 ]; then
            # for MDM data, do beamforming
            ! hash BeamformIt && echo "Missing BeamformIt, run 'cd ../../../tools/; extras/install_beamformit.sh; cd -;'" && exit 1
            local/ami_beamform.sh --cmd "$train_cmd" --nj 20 $nmics $AMI_DIR $PROCESSED_AMI_DIR
        fi
    else
        PROCESSED_AMI_DIR=$AMI_DIR
    fi

    echo ${base_mic}
    local/ami_${base_mic}_data_prep.sh $PROCESSED_AMI_DIR ${mic}
    local/ami_${base_mic}_scoring_data_prep.sh $PROCESSED_AMI_DIR ${mic} dev
    local/ami_${base_mic}_scoring_data_prep.sh $PROCESSED_AMI_DIR ${mic} eval

    for dset in train dev eval; do
        # this splits up the speakers (which for sdm and mdm just correspond
        # to recordings) into 30-second chunks.  It's like a very brain-dead form
        # of diarization; we can later replace it with 'real' diarization.
        seconds_per_spk_max=30
        [ "${mic}" == "ihm" ] && seconds_per_spk_max=120  # speaker info for ihm is real,
        # so organize into much bigger chunks.

        # Note: the 30 on the next line should have been $seconds_per_spk_max
        # (thanks: Pavel Denisov.  This is a bug but before fixing it we'd have to
        # test the WER impact.  I suspect it will be quite small and maybe hard to
        # measure consistently.
        utils/data/modify_speaker_info.sh --seconds-per-spk-max 30 ${data}/${mic}/${dset}_orig ${data}/${dset}_${mic}
    done

    # lowercasing
    for x in train_${mic} dev_${mic} eval_${mic}; do
        cp ${data}/${x}/text ${data}/${x}/text.org
        paste -d " " <(cut -f 1 -d " " ${data}/${x}/text.org) \
            <(cut -f 2- -d " " ${data}/${x}/text.org | awk '{print tolower($0)}') > ${data}/${x}/text
    done

    touch ${data}/.done_stage_0_${mic} && echo "Finish data preparation (stage: 0)."
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ ! -e ${data}/.done_stage_1_${mic}_sp${speed_perturb} ]; then
    echo ============================================================================
    echo "                    Feature extranction (stage:1)                          "
    echo ============================================================================

    if [ ! -e ${data}/.done_stage_1_${mic}_spfalse ]; then
        for x in train_${mic} dev_${mic} eval_${mic}; do
            steps/make_fbank.sh --nj 32 --cmd "$train_cmd" --write_utt2num_frames true \
                ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1;
            utils/fix_data_dir.sh ${data}/${x} || exit 1;
        done
    fi

    if [ ${speed_perturb} = true ]; then
        speed_perturb_3way.sh ${data} train_${mic} ${train_set}
        cp -rf ${data}/dev_${mic} ${data}/${dev_set}
        cp -rf ${data}/eval_${mic} ${data}/${test_set}
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

    touch ${data}/.done_stage_1_${mic}_sp${speed_perturb} && echo "Finish feature extranction (stage: 1)."
fi

dict=${data}/dict/${train_set}_${unit}${wp_type}${vocab}.txt; mkdir -p ${data}/dict
wp_model=${data}/dict/${train_set}_${wp_type}${vocab}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ ! -e ${data}/.done_stage_2_${mic}_${unit}${wp_type}${vocab}_sp${speed_perturb} ]; then
    echo ============================================================================
    echo "                      Dataset preparation (stage:2)                        "
    echo ============================================================================

    if [ ${unit} = wp ]; then
        make_vocab.sh --unit ${unit} --speed_perturb ${speed_perturb} \
            --vocab ${vocab} --wp_type ${wp_type} --wp_model ${wp_model} \
            ${data} ${dict} ${data}/${train_set}/text || exit 1;
    else
        make_vocab.sh --unit ${unit} --speed_perturb ${speed_perturb} \
            ${data} ${dict} ${data}/${train_set}/text || exit 1;
    fi

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

    touch ${data}/.done_stage_2_${mic}_${unit}${wp_type}${vocab}_sp${speed_perturb} && echo "Finish creating dataset for ASR (stage: 2)."
fi

if [ -z ${gpu} ]; then
    echo "Error: set GPU number." 1>&2
    echo "Usage: ./run.sh --gpu 0" 1>&2
    exit 1
fi
n_gpus=$(echo ${gpu} | tr "," "\n" | wc -l)

mkdir -p ${model}
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo ============================================================================
    echo "                        LM Training stage (stage:3)                       "
    echo ============================================================================

    # Extend dictionary for the external text data
    if [ ! -e ${data}/.done_stage_3_${unit}${wp_type}${vocab} ]; then
        echo "Making dataset tsv files for LM ..."
        mkdir -p ${data}/dataset_lm
        for x in train dev; do
            cp ${data}/dataset/${x}_${mic}_${unit}${wp_type}${vocab}.tsv \
                ${data}/dataset_lm/${x}_${unit}${wp_type}${vocab}.tsv || exit 1;
        done

        # augment with Switchboard+Fisher data
        if [ ! -z ${SWBD_AUDIOPATH} ]; then
            cur_dir=$(pwd)
            cd ../../swbd/s5c
            local/swbd1_data_download.sh ${SWBD_AUDIOPATH} || exit 1;
            local/swbd1_prepare_dict.sh || exit 1;
            local/swbd1_data_prep.sh ${SWBD_AUDIOPATH} || exit 1;
            cd ${cur_dir}

            update_dataset.sh --unit ${unit} --wp_model ${wp_model} \
                ${data}/train_swbd/text ${dict} ${data}/dataset_lm/train_${unit}${wp_type}${vocab}.tsv \
                > ${data}/dataset_lm/train_${unit}${wp_type}${vocab}.tsv.tmp || exit 1;
            mv ${data}/dataset_lm/train_${unit}${wp_type}${vocab}.tsv.tmp ${data}/dataset_lm/train_${unit}${wp_type}${vocab}.tsv
        fi
        if [ ! -z ${FISHER_PATH} ]; then
            cur_dir=$(pwd)
            cd ../../swbd/s5c
            local/fisher_data_prep.sh ${FISHER_PATH}
            local/fisher_swbd_prepare_dict.sh
            utils/fix_data_dir.sh ${data}/train_fisher
            cd ${cur_dir}

            update_dataset.sh --unit ${unit} --wp_model ${wp_model} \
                ${data}/train_fisher/text ${dict} ${data}/dataset_lm/train_${unit}${wp_type}${vocab}.tsv \
                > ${data}/dataset_lm/train_${unit}${wp_type}${vocab}.tsv.tmp || exit 1;
            mv ${data}/dataset_lm/train_${unit}${wp_type}${vocab}.tsv.tmp ${data}/dataset_lm/train_${unit}${wp_type}${vocab}.tsv
        fi

        touch ${data}/.done_stage_3_${unit}${wp_type}${vocab} && echo "Finish creating dataset for LM (stage: 3)."
    fi

    export OMP_NUM_THREADS=${n_gpus}
    CUDA_VISIBLE_DEVICES=${gpu} python -m torch.distributed.launch --nproc_per_node=${n_gpus} --nnodes=1 --node_rank=0 \
        ${NEURALSP_ROOT}/neural_sp/bin/lm/train.py --local_world_size=${n_gpus} \
        --corpus ${corpus} \
        --config ${lm_conf} \
        --n_gpus ${n_gpus} \
        --cudnn_benchmark ${benchmark} \
        --cudnn_deterministic ${deterministic} \
        --train_set ${data}/dataset_lm/${train_set}_${unit}${wp_type}${vocab}.tsv \
        --dev_set ${data}/dataset_lm/${dev_set}_${unit}${wp_type}${vocab}.tsv \
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

    export OMP_NUM_THREADS=${n_gpus}
    CUDA_VISIBLE_DEVICES=${gpu} python -m torch.distributed.launch --nproc_per_node=${n_gpus} --nnodes=1 --node_rank=0 \
        ${NEURALSP_ROOT}/neural_sp/bin/asr/train.py --local_world_size=${n_gpus} \
        --corpus ${corpus} \
        --use_wandb ${use_wandb} \
        --config ${conf} \
        --config2 ${conf2} \
        --n_gpus ${n_gpus} \
        --cudnn_benchmark ${benchmark} \
        --cudnn_deterministic ${deterministic} \
        --pin_memory ${pin_memory} \
        --train_set ${data}/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv \
        --dev_set ${data}/dataset/${dev_set}_${unit}${wp_type}${vocab}.tsv \
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
