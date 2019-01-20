#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                              Switchboard                                  "
echo ============================================================================

stage=0
gpu=

### path to save preproecssed data
export data=/n/sd8/inaguma/corpus/swbd

### vocabulary
unit=wp      # or word or char or word_char
vocab_size=10000
wp_type=bpe  # or unigram (for wordpiece)

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
# conv_channels="64_64_128_128"
# conv_kernel_sizes="(3,3)_(3,3)_(3,3)_(3,3)"
# conv_strides="(1,1)_(1,1)_(1,1)_(1,1)"
# conv_poolings="(1,1)_(2,2)_(1,1)_(2,2)"
conv_batch_norm=
enc_type=blstm
enc_nunits=320
enc_nprojs=0
enc_nlayers=5
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
dec_loop_type=normal
dec_residual=
input_feeding=
emb_dim=320
tie_embedding=
ctc_fc_list="320"
### optimization
batch_size=50
optimizer=adam
learning_rate=1e-3
nepochs=25
convert_to_sgd_epoch=20
print_step=200
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
bwd_weight=0.0
agreement_weight=0.0
twin_net_weight=0.0
mtl_per_batch=true
task_specific_layer=
### LM integration
cold_fusion=
rnnlm_cold_fusion=
rnnlm_init=
lmobj_weight=
share_lm_softmax=

#########################
# RNNLM configuration
#########################
# topology
lm_rnn_type=lstm
lm_nunits=1024
lm_nprojs=0
lm_nlayers=2
lm_emb_dim=1024
lm_tie_weights=true
lm_residual=true
# optimization
lm_batch_size=256
lm_bptt=100
lm_optimizer=adam
lm_learning_rate=1e-3
lm_nepochs=50
lm_convert_to_sgd_epoch=50
lm_print_step=20
lm_decay_start_epoch=10
lm_decay_rate=0.9
lm_decay_patient_epoch=0
lm_not_improved_patient_epoch=10
lm_eval_start_epoch=1
# initialization
lm_param_init=0.1
lm_param_init_dist=uniform
lm_pretrained_model=
# regularization
lm_clip_grad_norm=5.0
lm_dropout_hidden=0.5
lm_dropout_out=0.0
lm_dropout_emb=0.2
lm_weight_decay=1e-6
lm_backward=
# data size
lm_data_size=  # default is the same data as ASR

### path to save the model
model=/n/sd8/inaguma/result/swbd

### path to the model directory to restart training
rnnlm_resume=
resume=

### path to original data
SWBD_AUDIOPATH=/n/rd21/corpora_7/swb
EVAL2000_AUDIOPATH=/n/rd21/corpora_7/hub5_english/LDC2002S09
EVAL2000_TRANSPATH=/n/rd21/corpora_7/hub5_english/LDC2002T43
RT03_PATH=
FISHER_PATH=/n/rd7/fisher_english

### data size
data_size=swbd

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

train_set=train_${data_size}
dev_set=dev_${data_size}
test_set="eval2000"

if [ ${unit} = char ]; then
  vocab_size=
fi
if [ ${unit} != wp ]; then
  wp_type=
fi

if [ ${stage} -le 0 ] && [ ! -e ${data}/.done_stage_0_${data_size} ]; then
  echo ============================================================================
  echo "                       Data Preparation (stage:0)                          "
  echo ============================================================================

  local/swbd1_data_download.sh ${SWBD_AUDIOPATH} || exit 1;
  local/swbd1_prepare_dict.sh || exit 1;
  local/swbd1_data_prep.sh ${SWBD_AUDIOPATH} || exit 1;
  local/eval2000_data_prep.sh ${EVAL2000_AUDIOPATH} ${EVAL2000_TRANSPATH} || exit 1;

  # if [ -d ${RT03_PATH} ]; then
  #   local/rt03_data_prep.sh ${RT03_PATH}
  # fi

  # prepare fisher data for language models (optional)
  # if [ -d ${FISHER_PATH} ]; then
  #   # prepare fisher data and put it under data/train_fisher
  #   local/fisher_data_prep.sh ${FISHER_PATH}
  #   local/fisher_swbd_prepare_dict.sh
  #
  #   # merge two datasets into one
  #   mkdir -p ${data}/train_swbd_fisher
  #   for f in spk2utt utt2spk wav.scp text segments; do
  #     cat ${data}/train_fisher/$f ${data}/train_swbd/$f > ${data}/train_swbd_fisher/$f
  #   done
  # fi

  touch ${data}/.done_stage_0_${data_size} && echo "Finish data preparation (stage: 0)."
fi

if [ ${stage} -le 1 ] && [ ! -e ${data}/.done_stage_1_${data_size} ]; then
  echo ============================================================================
  echo "                    Feature extranction (stage:1)                          "
  echo ============================================================================

  for x in ${train_set} ${test_set}; do
    steps/make_fbank.sh --nj 16 --cmd "$train_cmd" --write_utt2num_frames true \
      ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1;
  done

  # Use the first 4k sentences as dev set.
  utils/subset_data_dir.sh --first ${data}/${train_set} 4000 ${data}/${dev_set} || exit 1;  # 5hr 6min
  n=$[`cat ${data}/${train_set}/segments | wc -l` - 4000]
  utils/subset_data_dir.sh --last ${data}/${train_set} ${n} ${data}/${train_set}.tmp || exit 1;

  # Finally, the full training set:
  utils/data/remove_dup_utts.sh 300 ${data}/${train_set}.tmp ${data}/${train_set} || exit 1;  # 286hr
  rm -rf ${data}/*.tmp

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

dict=${data}/dict/${train_set}_${unit}${wp_type}${vocab_size}.txt; mkdir -p ${data}/dict
nlsyms=${data}/dict/non_linguistic_symbols_${data_size}.txt
wp_model=${data}/dict/${train_set}_${wp_type}${vocab_size}
if [ ${stage} -le 2 ] && [ ! -e ${data}/.done_stage_2_${data_size}_${unit}${wp_type}${vocab_size} ]; then
  echo ============================================================================
  echo "                      Dataset preparation (stage:2)                        "
  echo ============================================================================

  echo "make a non-linguistic symbol list"
  cut -f 2- -d " " ${data}/${train_set}/text | tr " " "\n" | sort | uniq | grep "\[" > ${nlsyms}
  cat ${nlsyms}

  # Make a dictionary
  echo "<unk> 1" > ${dict}  # <unk> must be 1, 0 will be used for "blank" in CTC
  echo "<eos> 2" >> ${dict}  # <sos> and <eos> share the same index
  echo "<pad> 3" >> ${dict}
  if [ ${unit} = char ]; then
    echo "<space> 4" >> ${dict}
  fi
  offset=`cat ${dict} | wc -l`
  echo "Making a dictionary..."
  if [ ${unit} = wp ]; then
    cut -f 2- -d " " ${data}/${train_set}/text > ${data}/dict/input.txt
    spm_train --user_defined_symbols=`cat ${nlsyms} | tr "\n" ","` --input=${data}/dict/input.txt --vocab_size=${vocab_size} --model_type=${wp_type} --model_prefix=${wp_model} --input_sentence_size=100000000 --character_coverage=1.0
    spm_encode --model=${wp_model}.model --output_format=piece < ${data}/dict/input.txt | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}
  else
    text2dict.py ${data}/${train_set}/text --unit ${unit} --vocab_size ${vocab_size} --nlsyms ${nlsyms} \
      --wp_type ${wp_type} --wp_model ${wp_model} | \
      sort | uniq | grep -v -e '^\s*$' | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict} || exit 1;
  fi
  echo "vocab size:" `cat ${dict} | wc -l`

  # Compute OOV rate
  if [ ${unit} = word ]; then
    mkdir -p ${data}/dict/word_count ${data}/dict/oov_rate
    echo "OOV rate:" > ${data}/dict/oov_rate/word_${vocab_size}_${data_size}.txt
    for x in ${train_set} ${dev_set}; do
      cut -f 2- -d " " ${data}/${x}/text | tr " " "\n" | sort | uniq -c | sort -n -k1 -r \
        > ${data}/dict/word_count/${x}_${data_size}.txt || exit 1;
      compute_oov_rate.py ${data}/dict/word_count/${x}_${data_size}.txt ${dict} ${x} \
        >> ${data}/dict/oov_rate/word_${vocab_size}_${data_size}.txt || exit 1;
    done

    # 1) convert upper to lower
    # 2) remove tags (%AH) (%HESITATION) (%UH)
    # 3) remove <B_ASIDE> <E_ASIDE>
    # 4) remove "(" or ")"
    # swichboard
    grep -v en ${data}/${test_set}/text | cut -f 2- -d " " | awk '{ print tolower($0) }' | \
      perl -pe 's| \(\%.*\)||g' | perl -pe 's| \<.*\>||g' | sed -e "s/(//g" -e "s/)//g" | sed -e 's/\s\+/ /g' | \
      tr " " "\n" | sort | uniq -c | sort -n -k1 -r \
      > ${data}/dict/word_count/${test_set}_swbd.txt || exit 1;
    compute_oov_rate.py ${data}/dict/word_count/${test_set}_swbd.txt ${dict} ${test_set}_swbd \
      >> ${data}/dict/oov_rate/word_${vocab_size}.txt || exit 1;
    # callhome
    grep -v sw ${data}/${test_set}/text | cut -f 2- -d " " | awk '{ print tolower($0) }' | \
      perl -pe 's| \(\%.*\)||g' | perl -pe 's| \<.*\>||g' | sed -e "s/(//g" -e "s/)//g" | sed -e 's/\s\+/ /g' | \
      tr " " "\n" | sort | uniq -c | sort -n -k1 -r \
      > ${data}/dict/word_count/${test_set}_callhm.txt || exit 1;
    compute_oov_rate.py ${data}/dict/word_count/${test_set}_callhm.txt ${dict} ${test_set}_callhm \
      >> ${data}/dict/oov_rate/word_${vocab_size}.txt || exit 1;
    cat ${data}/dict/oov_rate/word_${vocab_size}_${data_size}.txt
  fi

  # Make datset csv files for the ASR task
  mkdir -p ${data}/dataset
  for x in ${train_set} ${dev_set}; do
    echo "Making a ASR csv file for ${x}..."
    dump_dir=${data}/dump/${x}
    make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit} --nlsyms ${nlsyms} --wp_model ${wp_model} \
      ${data}/${x} ${dict} > ${data}/dataset/${x}_${unit}${wp_type}${vocab_size}.csv || exit 1;
  done
  for x in ${test_set}; do
    echo "Making a ASR csv file for ${x}..."
    dump_dir=${data}/dump/${x}_${data_size}
    make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit} --nlsyms ${nlsyms} --wp_model ${wp_model} \
      ${data}/${x} ${dict} > ${data}/dataset/${x}_${data_size}_${unit}${wp_type}${vocab_size}.csv || exit 1;
  done

  touch ${data}/.done_stage_2_${data_size}_${unit}${wp_type}${vocab_size} && echo "Finish creating dataset for ASR (stage: 2)."
fi

mkdir -p ${model}
if [ ${stage} -le 3 ]; then
  echo ============================================================================
  echo "                      RNNLM Training stage (stage:3)                       "
  echo ============================================================================

  if [ -z ${lm_data_size} ]; then
    lm_data_size=${data_size}
  fi

  if [ ! -e ${data}/.done_stage_3_${lm_data_size}_${unit}${wp_type}${vocab_size} ]; then
    if [ ! -e ${data}/.done_stage_1_${data_size} ]; then
      echo "run ./run.sh --data_size `${lm_data_size}` first" && exit 1
    fi

    # Make datset csv files for the LM task
    mkdir -p ${data}/dataset_lm
    for x in train_${lm_data_size} dev_${lm_data_size}; do
      echo "Making a LM csv file for ${x}..."
      if [ ${lm_data_size} == ${data_size} ]; then
        cp ${data}/dataset/${x}_${unit}${wp_type}${vocab_size}.csv ${data}/dataset_lm/${x}_${train_set}_${unit}${wp_type}${vocab_size}.csv || exit 1;
      else
        dump_dir=${data}/dump/${x}
        make_dataset.sh --unit ${unit} --wp_model ${wp_model} \
          ${data}/${x} ${dict} > ${data}/dataset_lm/${x}_${train_set}_${unit}${wp_type}${vocab_size}.csv || exit 1;
      fi
    done

    touch ${data}/.done_stage_3_${lm_data_size}_${unit}${wp_type}${vocab_size} && echo "Finish creating dataset for LM (stage: 3)."
  fi

  lm_train_set=${data}/dataset_lm/train_${lm_data_size}_${train_set}_${unit}${wp_type}${vocab_size}.csv
  lm_dev_set=${data}/dataset_lm/dev_${lm_data_size}_${train_set}_${unit}${wp_type}${vocab_size}.csv

  # NOTE: support only a single GPU for RNNLM training
  CUDA_VISIBLE_DEVICES=${rnnlm_gpu} ../../../neural_sp/bin/lm/train.py \
    --ngpus 1 \
    --train_set ${lm_train_set} \
    --dev_set ${lm_dev_set} \
    --dict ${dict} \
    --wp_model ${wp_model}.model \
    --model ${model}/rnnlm \
    --unit ${unit} \
    --rnn_type ${lm_rnn_type} \
    --nunits ${lm_nunits} \
    --nprojs ${lm_nprojs} \
    --nlayers ${lm_nlayers} \
    --emb_dim ${lm_emb_dim} \
    --tie_weights ${lm_tie_weights} \
    --residual ${lm_residual} \
    --batch_size ${lm_batch_size} \
    --bptt ${lm_bptt} \
    --optimizer ${lm_optimizer} \
    --learning_rate ${lm_learning_rate} \
    --nepochs ${lm_nepochs} \
    --convert_to_sgd_epoch ${lm_convert_to_sgd_epoch} \
    --print_step ${lm_print_step} \
    --decay_start_epoch ${lm_decay_start_epoch} \
    --decay_rate ${lm_decay_rate} \
    --decay_patient_epoch ${lm_decay_patient_epoch} \
    --not_improved_patient_epoch ${lm_not_improved_patient_epoch} \
    --eval_start_epoch ${lm_eval_start_epoch} \
    --param_init ${lm_param_init} \
    --param_init_dist ${lm_param_init_dist} \
    --pretrained_model ${lm_pretrained_model} \
    --clip_grad_norm ${lm_clip_grad_norm} \
    --dropout_hidden ${lm_dropout_hidden} \
    --dropout_out ${lm_dropout_out} \
    --dropout_emb ${lm_dropout_emb} \
    --weight_decay ${lm_weight_decay} \
    --backward ${lm_backward} || exit 1;
    # --resume ${rnnlm_resume} || exit 1;

  echo "Finish RNNLM training (stage: 3)."
fi

if [ ${stage} -le 4 ]; then
  echo ============================================================================
  echo "                       ASR Training stage (stage:4)                        "
  echo ============================================================================

  CUDA_VISIBLE_DEVICES=${gpu} ../../../neural_sp/bin/asr/train.py \
    --ngpus ${ngpus} \
    --train_set ${data}/dataset/${train_set}_${unit}${wp_type}${vocab_size}.csv \
    --dev_set ${data}/dataset/${dev_set}_${unit}${wp_type}${vocab_size}.csv \
    --dict ${dict} \
    --wp_model ${wp_model}.model \
    --model ${model}/asr \
    --unit ${unit} \
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
    --bwd_weight ${bwd_weight} \
    --agreement_weight ${agreement_weight} \
    --twin_net_weight ${twin_net_weight} \
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
