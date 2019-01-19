#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                              LibriSpeech                                  "
echo ============================================================================

stage=0
gpu=

### path to save preproecssed data
export data=/n/sd8/inaguma/corpus/librispeech

### vocabulary
unit=wp      # or word or word_char
vocab_size=10000
wp_type=bpe  # or unigram (for wordpiece)
unit_sub1=char
wp_type_sub1=bpe  # or unigram (for wordpiece)
vocab_size_sub1=

#########################
# ASR configuration
#########################
### topology
nsplices=1
nstacks=1
nskips=1
conv_in_channel=1
conv_channels=
conv_kernel_sizes=
conv_strides=
conv_poolings=
conv_batch_norm=
enc_type=blstm
enc_nunits=320
enc_nprojs=0
enc_nlayers=5
enc_nlayers_sub1=4
enc_residual=
subsample="1_2_2_2_1"
subsample_type=drop
attn_type=location
attn_dim=320
attn_nheads=1
attn_sigmoid=
dec_type=lstm
dec_nunits=320
dec_nprojs=0
dec_nlayers=1
dec_nlayers_sub1=1
dec_loop_type=normal
dec_residual=
input_feeding=
emb_dim=320
tie_embedding=
ctc_fc_list="320"
ctc_fc_list_sub1=""
### optimization
batch_size=50
optimizer=adam
learning_rate=1e-3
nepochs=25
convert_to_sgd_epoch=20
print_step=500
decay_start_epoch=10
decay_rate=0.9
decay_patient_epoch=0
decay_type=epoch
not_improved_patient_epoch=5
eval_start_epoch=1
warmup_start_learning_rate=1e-4
warmup_step=0
warmup_epoch=0
### initialization
param_init=0.1
param_init_dist=uniform
pretrained_model=
### regularization
clip_grad_norm=5.0
dropout_in=0.0
dropout_enc=0.2
dropout_dec=0.2
dropout_emb=0.2
dropout_att=0.0
weight_decay=1e-6
ss_prob=0.2
ss_type=constant
lsm_prob=0.1
focal_loss=0.0
### MTL
ctc_weight=0.0
ctc_weight_sub1=0.2
bwd_weight=0.0
bwd_weight_sub1=0.0
twin_net_weight=0.0
sub1_weight=0.2
mtl_per_batch=true
task_specific_layer=true
### LM integration
cold_fusion=
rnnlm_cold_fusion=
rnnlm_init=
lmobj_weight=
share_lm_softmax=

### path to save the model
model=/n/sd8/inaguma/result/librispeech

### path to the model directory to restart training
resume=

### path to download data
data_download_path=/n/rd21/corpora_7/librispeech/

### data size
data_size=960

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

# Base url for downloads.
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11

train_set=train_${data_size}
dev_set=dev_${data_size}
test_set="dev_clean dev_other test_clean test_other"

# main
if [ ${unit} = char ]; then
  vocab_size=
fi
if [ ${unit} != wp ]; then
  wp_type=
fi
# sub1
if [ ${unit_sub1} = char ]; then
  vocab_size_sub1=
fi
if [ ${unit_sub1} != wp ]; then
  wp_type_sub1=
fi

if [ ${stage} -le 0 ] && [ ! -e ${data}/.done_stage_0_${data_size} ]; then
  echo ============================================================================
  echo "                       Data Preparation (stage:0)                          "
  echo ============================================================================

  # download data
  # mkdir -p ${data}
  # for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
  #   local/download_and_untar.sh ${data_download_path} ${data_url} ${part} || exit 1;
  # done

  # download the LM resources
  # local/download_lm.sh ${lm_url} ${data}/local/lm || exit 1;

  # format the data as Kaldi data directories
  for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    # use underscore-separated names in data directories.
    local/data_prep.sh ${data_download_path}/LibriSpeech/${part} ${data}/$(echo ${part} | sed s/-/_/g) || exit 1;
  done

  # lowercasing
  for x in dev_clean test_clean dev_other test_other train_clean_100 train_clean_360 train_other_500; do
    cp ${data}/${x}/text ${data}/${x}/text.tmp
    paste -d " " <(cut -f 1 -d" " ${data}/${x}/text.tmp) \
                 <(cut -f 2- -d" " ${data}/${x}/text.tmp | awk '{$1=""; print tolower($0)}') > ${data}/${x}/text
    rm ${data}/${x}/text.tmp
  done

  touch ${data}/.done_stage_0_${data_size} && echo "Finish data preparation (stage: 0)."
fi

if [ ${stage} -le 1 ] && [ ! -e ${data}/.done_stage_1_${data_size} ]; then
  echo ============================================================================
  echo "                    Feature extranction (stage:1)                          "
  echo ============================================================================

  for x in dev_clean test_clean dev_other test_other train_clean_100 train_clean_360 train_other_500; do
    steps/make_fbank.sh --nj 16 --cmd "$train_cmd" --write_utt2num_frames true \
      ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1;
  done

  utils/combine_data.sh --extra_files "utt2num_frames" ${data}/${train_set} ${data}/train_clean_100 ${data}/train_clean_360 ${data}/train_other_500 || exit 1;
  utils/combine_data.sh --extra_files "utt2num_frames" ${data}/${dev_set} ${data}/dev_clean ${data}/dev_other || exit 1;

  # Compute global CMVN
  compute-cmvn-stats scp:${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark || exit 1;

  # Apply global CMVN & dump features
  for x in ${train_set} ${dev_set}; do
    dump_dir=${data}/dump/${x}
    dump_feat.sh --cmd "$train_cmd" --nj 16 --add_deltadelta false \
      ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x} ${dump_dir} || exit 1;
  done
  for x in ${test_set}; do
    dump_dir=${data}/dump/${x}_${data_size}
    dump_feat.sh --cmd "$train_cmd" --nj 16 --add_deltadelta false \
      ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x}_${data_size} ${dump_dir} || exit 1;
  done

  touch ${data}/.done_stage_1_${data_size} && echo "Finish feature extranction (stage: 1)."
fi

# main
dict=${data}/dict/${train_set}_${unit}${wp_type}${vocab_size}.txt
wp_model=${data}/dict/${train_set}_${wp_type}${vocab_size}
# sub1
dict_sub1=${data}/dict/${train_set}_${unit_sub1}${wp_type_sub1}${vocab_size_sub1}.txt
wp_model_sub1=${data}/dict/${train_set}_${wp_type_sub1}${vocab_size_sub1}

if [ ! -f ${dict} ]; then
  echo "There is no file such as "${dict}
  exit 1
fi

if [ ! -f ${dict_sub1} ]; then
  echo "There is no file such as "${dict_sub1}
  exit 1
fi

mkdir -p ${model}
if [ ${stage} -le 4 ]; then
  echo ============================================================================
  echo "                       ASR Training stage (stage:4)                        "
  echo ============================================================================

  CUDA_VISIBLE_DEVICES=${gpu} ../../../neural_sp/bin/asr/train.py \
    --ngpus ${ngpus} \
    --train_set ${data}/dataset/${train_set}_${unit}${wp_type}${vocab_size}.csv \
    --train_set_sub1 ${data}/dataset/${train_set}_${unit_sub1}${wp_type_sub1}${vocab_size_sub1}.csv \
    --dev_set ${data}/dataset/${dev_set}_${unit}${wp_type}${vocab_size}.csv \
    --dev_set_sub1 ${data}/dataset/${dev_set}_${unit_sub1}${wp_type_sub1}${vocab_size_sub1}.csv \
    --dict ${dict} \
    --dict_sub1 ${dict_sub1} \
    --wp_model ${wp_model}.model \
    --wp_model_sub1 ${wp_model_sub1}.model \
    --model ${model}/asr \
    --unit ${unit} \
    --unit_sub1 ${unit_sub1} \
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
    --enc_nlayers_sub1 ${enc_nlayers_sub1} \
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
    --dec_nlayers_sub1 ${dec_nlayers_sub1} \
    --dec_loop_type ${dec_loop_type} \
    --dec_residual ${dec_residual} \
    --input_feeding ${input_feeding} \
    --emb_dim ${emb_dim} \
    --tie_embedding ${tie_embedding} \
    --ctc_fc_list ${ctc_fc_list} \
    --ctc_fc_list_sub1 ${ctc_fc_list_sub1} \
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
    --warmup_step ${warmup_step} \
    --warmup_epoch ${warmup_epoch} \
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
    --ctc_weight_sub1 ${ctc_weight_sub1} \
    --bwd_weight ${bwd_weight} \
    --bwd_weight_sub1 ${bwd_weight_sub1} \
    --twin_net_weight ${twin_net_weight} \
    --sub1_weight ${sub1_weight} \
    --mtl_per_batch ${mtl_per_batch} \
    --task_specific_layer ${task_specific_layer} \
    --cold_fusion ${cold_fusion} \
    --rnnlm_cold_fusion =${rnnlm_cold_fusion} \
    --rnnlm_init ${rnnlm_init} \
    --lmobj_weight ${lmobj_weight} \
    --share_lm_softmax ${share_lm_softmax} || exit 1;
    # --resume ${resume} || exit 1;

  echo "Finish model training (stage: 4)."
fi
