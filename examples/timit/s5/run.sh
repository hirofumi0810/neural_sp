#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                                  TIMIT                                    "
echo ============================================================================

stage=0
gpu=

#########################
# ASR configuration
#########################
### topology
n_splices=1
n_stacks=1
n_skips=1
sequence_summary_network=false
conv_in_channel=3
conv_channels=
conv_kernel_sizes=
conv_strides=
conv_poolings=
conv_batch_norm=false
conv_bottleneck_dim=0
subsample="1_1_1_1_1"
enc_type=bgru
enc_n_units=256
enc_n_projs=0
enc_n_layers=5
enc_residual=false
subsample_type=drop
attn_type=location
attn_dim=256
attn_n_heads=1
attn_sigmoid=false
dec_type=gru
dec_n_units=256
dec_n_projs=0
dec_n_layers=1
dec_loop_type=normal
dec_residual=false
input_feeding=false
emb_dim=256
tie_embedding=false
ctc_fc_list=""
### optimization
batch_size=32
optimizer=adam
learning_rate=1e-3
n_epochs=100
convert_to_sgd_epoch=40
print_step=10
decay_start_epoch=20
decay_rate=0.97
decay_patient_n_epochs=0
decay_type=epoch
not_improved_patient_n_epochs=20
eval_start_epoch=20
warmup_start_learning_rate=1e-4
warmup_n_steps=4000
### initialization
param_init=0.1
param_init_dist=uniform
pretrained_model=
### regularization
clip_grad_norm=5.0
dropout_in=0.2
dropout_enc=0.5
dropout_dec=0.2
dropout_emb=0.2
dropout_att=0.0
weight_decay=1e-6
ss_prob=0.0
ss_type=constant
lsm_prob=0.0
layer_norm=false
focal_loss=0.0
### MTL
ctc_weight=0.0
bwd_weight=0.0
mtl_per_batch=true
task_specific_layer=false

### path to save the model
model=/n/sd8/inaguma/result/timit

### path to the model directory to resume training
resume=

### path to save preproecssed data
export data=/n/sd8/inaguma/corpus/timit

### path to original data
TIMITDATATOP=/n/rd21/corpora_1/TIMIT

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

train_set=train
dev_set=dev
test_set="test"

if [ ${stage} -le 0 ] && [ ! -e ${data}/.done_stage_0 ]; then
    echo ============================================================================
    echo "                       Data Preparation (stage:0)                          "
    echo ============================================================================

    mkdir -p ${data}
    local/timit_data_prep.sh ${TIMITDATATOP} || exit 1;
    local/timit_format_data.sh || exit 1;

    touch ${data}/.done_stage_0 && echo "Finish data preparation (stage: 0)."
fi

if [ ${stage} -le 1 ] && [ ! -e ${data}/.done_stage_1 ]; then
    echo ============================================================================
    echo "                    Feature extranction (stage:1)                          "
    echo ============================================================================

    for x in train dev test; do
        steps/make_fbank.sh --nj 32 --cmd "$train_cmd" --write_utt2num_frames true \
            ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1;
    done

    # Compute global CMVN
    compute-cmvn-stats scp:${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark || exit 1;

    # Apply global CMVN & dump features
    dump_feat.sh --cmd "$train_cmd" --nj 80 --add_deltas true \
        ${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${train_set} ${data}/dump/${train_set} || exit 1;
    for x in ${dev_set} ${test_set}; do
        dump_dir=${data}/dump/${x}
        dump_feat.sh --cmd "$train_cmd" --nj 32 --add_deltas true \
            ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x} ${dump_dir} || exit 1;
    done

    touch ${data}/.done_stage_1 && echo "Finish feature extranction (stage: 1)."
fi

dict=${data}/dict/${train_set}.txt; mkdir -p ${data}/dict
if [ ${stage} -le 2 ] && [ ! -e ${data}/.done_stage_2 ]; then
    echo ============================================================================
    echo "                      Dataset preparation (stage:2)                        "
    echo ============================================================================

    echo "Making a dictionary..."
    echo "<unk> 1" > ${dict}  # <unk> must be 1, 0 will be used for "blank" in CTC
    echo "<eos> 2" >> ${dict}  # <sos> and <eos> share the same index
    echo "<pad> 3" >> ${dict}
    offset=$(cat ${dict} | wc -l)
    text2dict.py ${data}/${train_set}/text --unit phone | \
        awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict} || exit 1;
    echo "vocab size:" $(cat ${dict} | wc -l)

    echo "Making dataset tsv files for ASR ..."
    mkdir -p ${data}/dataset
    for x in ${train_set} ${dev_set} ${test_set}; do
        dump_dir=${data}/dump/${x}
        make_dataset.sh --feat ${dump_dir}/feats.scp --unit phone \
            ${data}/${x} ${dict} > ${data}/dataset/${x}.tsv || exit 1;
    done

    touch ${data}/.done_stage_2 && echo "Finish creating dataset for ASR (stage: 2)."
fi

# NOTE: skip LM training (stage:3)

mkdir -p ${model}
if [ ${stage} -le 4 ]; then
    echo ============================================================================
    echo "                       ASR Training stage (stage:4)                        "
    echo ============================================================================

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/asr/train.py \
        --corpus timit \
        --n_gpus ${n_gpus} \
        --train_set ${data}/dataset/${train_set}.tsv \
        --dev_set ${data}/dataset/${dev_set}.tsv \
        --eval_sets ${data}/dataset/${test_set}.tsv \
        --dict ${dict} \
        --model ${model}/asr \
        --unit phone \
        --n_splices ${n_splices} \
        --n_stacks ${n_stacks} \
        --n_skips ${n_skips} \
        --sequence_summary_network ${sequence_summary_network} \
        --conv_in_channel ${conv_in_channel} \
        --conv_channels ${conv_channels} \
        --conv_kernel_sizes ${conv_kernel_sizes} \
        --conv_strides ${conv_strides} \
        --conv_poolings ${conv_poolings} \
        --conv_batch_norm ${conv_batch_norm} \
        --conv_bottleneck_dim ${conv_bottleneck_dim} \
        --enc_type ${enc_type} \
        --enc_n_units ${enc_n_units} \
        --enc_n_projs ${enc_n_projs} \
        --enc_n_layers ${enc_n_layers} \
        --enc_residual ${enc_residual} \
        --subsample ${subsample} \
        --subsample_type ${subsample_type} \
        --attn_type ${attn_type} \
        --attn_dim ${attn_dim} \
        --attn_n_heads ${attn_n_heads} \
        --attn_sigmoid ${attn_sigmoid} \
        --dec_type ${dec_type} \
        --dec_n_units ${dec_n_units} \
        --dec_n_projs ${dec_n_projs} \
        --dec_n_layers ${dec_n_layers} \
        --dec_loop_type ${dec_loop_type} \
        --dec_residual ${dec_residual} \
        --input_feeding ${input_feeding} \
        --emb_dim ${emb_dim} \
        --tie_embedding ${tie_embedding} \
        --ctc_fc_list ${ctc_fc_list} \
        --batch_size ${batch_size} \
        --optimizer ${optimizer} \
        --learning_rate ${learning_rate} \
        --n_epochs ${n_epochs} \
        --convert_to_sgd_epoch ${convert_to_sgd_epoch} \
        --print_step ${print_step} \
        --decay_start_epoch ${decay_start_epoch} \
        --decay_rate ${decay_rate} \
        --decay_type ${decay_type} \
        --decay_patient_n_epochs ${decay_patient_n_epochs} \
        --not_improved_patient_n_epochs ${not_improved_patient_n_epochs} \
        --eval_start_epoch ${eval_start_epoch} \
        --warmup_start_learning_rate ${warmup_start_learning_rate} \
        --warmup_n_steps ${warmup_n_steps} \
        --param_init ${param_init} \
        --param_init_dist ${param_init_dist} \
        --pretrained_model ${pretrained_model} \
        --clip_grad_norm ${clip_grad_norm} \
        --dropout_in ${dropout_in} \
        --dropout_enc ${dropout_enc} \
        --dropout_dec ${dropout_dec} \
        --dropout_emb ${dropout_emb} \
        --dropout_att ${dropout_att} \
        --weight_decay ${weight_decay} \
        --ss_prob ${ss_prob} \
        --ss_type ${ss_type} \
        --lsm_prob ${lsm_prob} \
        --layer_norm ${layer_norm} \
        --focal_loss_weight ${focal_loss} \
        --ctc_weight ${ctc_weight} \
        --bwd_weight ${bwd_weight} \
        --mtl_per_batch ${mtl_per_batch} \
        --task_specific_layer ${task_specific_layer} \
        --resume ${resume} || exit 1;

    echo "Finish model training (stage: 4)."
fi
