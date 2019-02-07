#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                                  TIMIT                                    "
echo ============================================================================

stage=0
gpu=

### path to save preproecssed data
export data=/n/sd8/inaguma/corpus/timit

#########################
# ASR configuration
#########################
### topology
nsplices=1
nstacks=1
nskips=1
conv_in_channel=3
conv_channels=
conv_kernel_sizes=
conv_strides=
conv_poolings=
conv_batch_norm=
# VGG
# conv_channels="64_64_128_128"
# conv_kernel_sizes="(3,3)_(3,3)_(3,3)_(3,3)"
# conv_strides="(1,1)_(1,1)_(1,1)_(1,1)"
# conv_poolings="(1,1)_(2,2)_(1,1)_(2,2)"
subsample="1_1_1_1_1"
enc_type=blstm
enc_nunits=320
enc_type=bgru
enc_nunits=256
enc_nprojs=0
enc_nlayers=5
enc_residual=
subsample_type=drop
attn_type=location
attn_dim=256
attn_nheads=1
attn_sigmoid=
dec_type=gru
dec_nunits=256
dec_nprojs=0
dec_nlayers=1
dec_loop_type=normal
dec_residual=
input_feeding=
emb_dim=256
tie_embedding=
ctc_fc_list=""
### optimization
batch_size=32
optimizer=adam
learning_rate=1e-3
nepochs=100
convert_to_sgd_epoch=40
print_step=10
decay_start_epoch=20
decay_rate=0.97
decay_patient_epoch=0
decay_type=epoch
not_improved_patient_epoch=20
eval_start_epoch=20
warmup_start_learning_rate=1e-4
warmup_nsteps=4000
warmup_nepochs=0
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
focal_loss=0.0
### MTL
ctc_weight=0.0
bwd_weight=0.0
twin_net_weight=0.0
mtl_per_batch=true
task_specific_layer=

### path to save the model
model=/n/sd8/inaguma/result/timit

### path to the model directory to restart training
resume=

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
ngpus=`echo ${gpu} | tr "," "\n" | wc -l`
rnnlm_gpu=`echo ${gpu} | cut -d "," -f 1`

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
  for x in ${train_set} ${dev_set} ${test_set}; do
    dump_dir=${data}/dump/${x}
    dump_feat.sh --cmd "$train_cmd" --nj 32 --add_deltadelta true \
      ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x} ${dump_dir} || exit 1;
  done

  touch ${data}/.done_stage_1 && echo "Finish feature extranction (stage: 1)."
fi

dict=${data}/dict/${train_set}.txt; mkdir -p ${data}/dict
if [ ${stage} -le 2 ] && [ ! -e ${data}/.done_stage_2 ]; then
  echo ============================================================================
  echo "                      Dataset preparation (stage:2)                        "
  echo ============================================================================

  # Make a dictionary
  echo "<unk> 1" > ${dict}  # <unk> must be 1, 0 will be used for "blank" in CTC
  echo "<eos> 2" >> ${dict}  # <sos> and <eos> share the same index
  echo "<pad> 3" >> ${dict}
  offset=`cat ${dict} | wc -l`
  echo "Making a dictionary..."
  text2dict.py ${data}/${train_set}/text --unit phone | \
    sort | uniq | grep -v -e '^\s*$' | awk -v offset=${offset} '{print $0 " " NR+offset-1}' >> ${dict} || exit 1;
  echo "vocab size:" `cat ${dict} | wc -l`

  # Make datset csv files
  mkdir -p ${data}/dataset
  for x in ${train_set} ${dev_set} ${test_set}; do
    echo "Making a csv file for ${x}..."
    dump_dir=${data}/dump/${x}
    make_dataset.sh --feat ${dump_dir}/feats.scp --unit phone \
      ${data}/${x} ${dict} > ${data}/dataset/${x}.csv || exit 1;
  done

  touch ${data}/.done_stage_2 && echo "Finish creating dataset (stage: 2)."
fi

# NOTE: skip RNNLM training (stage:3)

mkdir -p ${model}
if [ ${stage} -le 4 ]; then
  echo ============================================================================
  echo "                       ASR Training stage (stage:4)                        "
  echo ============================================================================

  CUDA_VISIBLE_DEVICES=${gpu} ../../../neural_sp/bin/asr/train.py \
    --ngpus ${ngpus} \
    --train_set ${data}/dataset/${train_set}.csv \
    --dev_set ${data}/dataset/${dev_set}.csv \
    --eval_sets ${data}/dataset/${test_set}.csv \
    --dict ${dict} \
    --model ${model}/asr \
    --unit phone \
    --nsplices ${nsplices} \
    --nstacks ${nstacks} \
    --nskips ${nskips} \
    --conv_in_channel ${conv_in_channel} \
    --conv_channels ${conv_channels} \
    --conv_kernel_sizes ${conv_kernel_sizes} \
    --conv_strides ${conv_strides} \
    --conv_poolings ${conv_poolings} \
    --conv_batch_norm ${conv_batch_norm} \
    --enc_type ${enc_type} \
    --enc_nunits ${enc_nunits} \
    --enc_nprojs ${enc_nprojs} \
    --enc_nlayers ${enc_nlayers} \
    --enc_residual ${enc_residual} \
    --subsample ${subsample} \
    --subsample_type ${subsample_type} \
    --attn_type ${attn_type} \
    --attn_dim ${attn_dim} \
    --attn_nheads ${attn_nheads} \
    --attn_sigmoid ${attn_sigmoid} \
    --dec_type ${dec_type} \
    --dec_nunits ${dec_nunits} \
    --dec_nprojs ${dec_nprojs} \
    --dec_nlayers ${dec_nlayers} \
    --dec_loop_type ${dec_loop_type} \
    --dec_residual ${dec_residual} \
    --input_feeding ${input_feeding} \
    --emb_dim ${emb_dim} \
    --tie_embedding ${tie_embedding} \
    --ctc_fc_list ${ctc_fc_list} \
    --batch_size ${batch_size} \
    --optimizer ${optimizer} \
    --learning_rate ${learning_rate} \
    --nepochs ${nepochs} \
    --convert_to_sgd_epoch ${convert_to_sgd_epoch} \
    --print_step ${print_step} \
    --decay_start_epoch ${decay_start_epoch} \
    --decay_rate ${decay_rate} \
    --decay_type ${decay_type} \
    --decay_patient_epoch ${decay_patient_epoch} \
    --not_improved_patient_epoch ${not_improved_patient_epoch} \
    --eval_start_epoch ${eval_start_epoch} \
    --warmup_start_learning_rate ${warmup_start_learning_rate} \
    --warmup_nsteps ${warmup_nsteps} \
    --warmup_nepochs ${warmup_nepochs} \
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
    --focal_loss_weight ${focal_loss} \
    --ctc_weight ${ctc_weight} \
    --bwd_weight ${bwd_weight} \
    --twin_net_weight ${twin_net_weight} \
    --mtl_per_batch ${mtl_per_batch} \
    --task_specific_layer ${task_specific_layer} \
    --resume ${resume} || exit 1;

  echo "Finish model training (stage: 4)."
fi
