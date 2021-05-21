#!/usr/bin/env bash

# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Modified from https://github.com/laboroai/LaboroTVSpeech/blob/main/kaldi/laborotv/s5/run.sh

echo ============================================================================
echo "                              LaboroTVSpeech                              "
echo ============================================================================

stage=0
stop_stage=5
gpu=
benchmark=true
deterministic=false
pin_memory=false
speed_perturb=false
stdout=false
wandb_id=""
corpus=laborotv

### vocabulary
unit=wp      # wp/char
vocab=10000
wp_type=bpe  # bpe/unigram (for wordpiece)

#########################
# ASR configuration
#########################
conf=conf/asr/conformer_kernel15_clamp10_hie_subsample8_las_ln.yaml
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

### path to original data
LABOROTV_DATA_ROOT="/n/rd5/LaboroTVSpeech_v1.0b"
TEDXJP_DATA_ROOT="/home/inaguma/inaguma/TEDxJP-10K/TEDxJP-10K_v1.1"

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

train_set=train_nodev
dev_set=dev_4k
test_set="dev tedx-jp-10k"
if [ ${speed_perturb} = true ]; then
    train_set=train_nodev_sp
    dev_set=dev_4k_sp
    test_set="dev_sp tedx-jp-10k_sp"
fi

use_wandb=false
if [ ! -z ${wandb_id} ]; then
    use_wandb=true
    wandb login ${wandb_id}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ] && [ ! -e ${data}/.done_stage_0 ]; then
    echo ============================================================================
    echo "                       Data Preparation (stage:0)                          "
    echo ============================================================================

    # Prepare train & dev set
    local/laborotv_data_prep.sh ${LABOROTV_DATA_ROOT} || exit 1;

    # Prepare TEDx-JP-10K to use as eval set.
    local/tedx-jp-10k_data_prep.sh ${TEDXJP_DATA_ROOT} || exit 1;

    local/prepare_dict.sh \
        ${LABOROTV_DATA_ROOT}/data/lexicon.txt ${data}/local/dict_nosp || exit 1;

    # Remove <sp> and POS tag, and lowercase
    for x in train dev; do
        local/remove_pos.py ${data}/${x}/text | nkf -Z > ${data}/${x}/text.tmp || exit 1;
        mv ${data}/${x}/text.tmp ${data}/${x}/text
    done

    touch ${data}/.done_stage_0 && echo "Finish data preparation (stage: 0)."
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ ! -e ${data}/.done_stage_1_sp${speed_perturb} ]; then
    echo ============================================================================
    echo "                    Feature extranction (stage:1)                          "
    echo ============================================================================

    if [ ! -e ${data}/.done_stage_1_spfalse ]; then
        for x in train dev tedx-jp-10k; do
            steps/make_fbank.sh --nj 32 --cmd "$train_cmd" --write_utt2num_frames true \
                ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1;
            utils/validate_data_dir.sh ${data}/${x}
        done

        # Use the first 4k sentences from training data as dev set.
        utils/subset_data_dir.sh --first ${data}/train 4000 ${data}/dev_4k || exit 1;
        n=$[$(cat ${data}/train/text | wc -l) - 4000]
        utils/subset_data_dir.sh --last ${data}/train ${n} ${data}/train_nodev || exit 1;
    fi

    if [ ${speed_perturb} = true ]; then
        speed_perturb_3way.sh ${data} train_nodev ${train_set}
        if [ ! -e ${data}/dev_sp ]; then
            cp -rf ${data}/dev_4k ${data}/dev_4k_sp
            cp -rf ${data}/dev ${data}/dev_sp
            cp -rf ${data}/tedx-jp-10k ${data}/tedx-jp-10k_sp
        fi
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
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ ! -e ${data}/.done_stage_2_${unit}${wp_type}${vocab}_sp${speed_perturb} ]; then
    echo ============================================================================
    echo "                      Dataset preparation (stage:2)                        "
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

    echo "Making dataset tsv files for ASR ..."
    mkdir -p ${data}/dataset
    for x in ${train_set} ${dev_set} ${test_set}; do
        dump_dir=${data}/dump/${x}
        make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit} --wp_model ${wp_model} \
            ${data}/${x} ${dict} > ${data}/dataset/${x}_${unit}${wp_type}${vocab}.tsv || exit 1;
    done

    touch ${data}/.done_stage_2_${unit}${wp_type}${vocab}_sp${speed_perturb} && echo "Finish creating dataset for ASR (stage: 2)."
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

    export OMP_NUM_THREADS=${n_gpus}
    CUDA_VISIBLE_DEVICES=${gpu} python -m torch.distributed.launch --nproc_per_node=${n_gpus} --nnodes=1 --node_rank=0 \
        ${NEURALSP_ROOT}/neural_sp/bin/lm/train.py --local_world_size=${n_gpus} \
        --corpus ${corpus} \
        --config ${lm_conf} \
        --n_gpus ${n_gpus} \
        --cudnn_benchmark ${benchmark} \
        --cudnn_deterministic ${deterministic} \
        --train_set ${data}/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv \
        --dev_set ${data}/dataset/${dev_set}_${unit}${wp_type}${vocab}.tsv \
        --unit ${unit} \
        --dict ${dict} \
        --wp_model ${wp_model}.model \
        --model_save_dir ${model}/lm \
        --stdout ${stdout} \
        --resume ${lm_resume} || exit 1;
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
