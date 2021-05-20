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
corpus=laborotv_csj

### vocabulary
unit=wp      # wp/char
vocab=10000
wp_type=bpe  # bpe/unigram (for wordpiece)

#########################
# ASR configuration
#########################
conf=conf/asr/conformer_kernel15_clamp10_hie_subsample8_las_ln_large.yaml
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
csj_dir=/n/work2/inaguma/corpus/csj
laborotv_dir=/n/work2/inaguma/corpus/laborotv

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

train_set=train_nodev
dev_set=dev_8k
test_set="dev_4k_laborotv dev_laborotv tedx-jp-10k dev_4k_csj eval1 eval2 eval3"
if [ ${speed_perturb} = true ]; then
    train_set=train_nodev_sp
    dev_set=dev_8k_sp
    test_set="dev_4k_laborotv_sp dev_laborotv_sp tedx-jp-10k_sp dev_4k_csj_sp eval1_sp eval2_sp eval3_sp"
fi

use_wandb=true
wandb login ${wandb_id}


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ ! -e ${data}/.done_stage_1_sp${speed_perturb} ]; then
    echo ============================================================================
    echo "                    Feature extranction (stage:1)                          "
    echo ============================================================================

    if [ ${speed_perturb} = true ]; then
        utils/combine_data.sh --extra_files "utt2num_frames" ${data}/${train_set} \
            ${laborotv_dir}/train_nodev_sp ${csj_dir}/train_nodev_sp_all || exit 1;
        cp -rf ${laborotv_dir}/dev_4k_sp ${data}/dev_4k_laborotv_sp
        cp -rf ${csj_dir}/dev_sp_all ${data}/dev_4k_csj_sp
        utils/combine_data.sh --extra_files "utt2num_frames" ${data}/dev_8k_sp \
            ${data}/dev_4k_laborotv_sp ${data}/dev_4k_csj_sp || exit 1;
        cp -rf ${laborotv_dir}/dev_sp ${data}/dev_laborotv_sp
        cp -rf ${laborotv_dir}/tedx-jp-10k_sp ${data}/tedx-jp-10k_sp
        cp -rf ${csj_dir}/eval1_sp ${data}/eval1_sp
        cp -rf ${csj_dir}/eval2_sp ${data}/eval2_sp
        cp -rf ${csj_dir}/eval3_sp ${data}/eval3_sp
    else
        utils/combine_data.sh --extra_files "utt2num_frames" ${data}/${train_set} \
            ${laborotv_dir}/train_nodev ${csj_dir}/train_nodev_all || exit 1;
        cp -rf ${laborotv_dir}/dev_4k ${data}/dev_4k_laborotv
        cp -rf ${csj_dir}/dev_all ${data}/dev_4k_csj
        utils/combine_data.sh --extra_files "utt2num_frames" ${data}/dev_8k \
            ${data}/dev_4k_laborotv ${data}/dev_4k_csj || exit 1;
        cp -rf ${laborotv_dir}/dev ${data}/dev_laborotv
        cp -rf ${laborotv_dir}/tedx-jp-10k ${data}/tedx-jp-10k
        cp -rf ${csj_dir}/eval1 ${data}/eval1
        cp -rf ${csj_dir}/eval2 ${data}/eval2
        cp -rf ${csj_dir}/eval3 ${data}/eval3
    fi

    # Compute global CMVN
    if [ ${speed_perturb} = true ]; then
        cat ${laborotv_dir}/train_nodev_sp/feats.scp ${csj_dir}/train_nodev_sp_all/feats.scp > ${data}/${train_set}/feats.scp
    else
        cat ${laborotv_dir}/train_nodev/feats.scp ${csj_dir}/train_nodev_all/feats.scp > ${data}/${train_set}/feats.scp
    fi
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
        --workers 3 \
        --resume ${resume} || exit 1;

    echo "Finish ASR model training (stage: 4)."
fi
