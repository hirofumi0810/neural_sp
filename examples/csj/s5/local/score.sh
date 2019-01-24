#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

model=
model1=
model2=
model3=
model4=
model5=
model6=
model7=
model_bwd=
gpu=

### path to save preproecssed data
data=/n/sd8/inaguma/corpus/csj

epoch=-1
batch_size=1
beam_width=5
min_len_ratio=0.0
max_len_ratio=1.0
length_penalty=0.0
coverage_penalty=0.6
coverage_threshold=0.0
rnnlm=
rnnlm_bwd=
rnnlm_weight=0.0
ctc_weight=0.0  # 1.0 for joint CTC-attention means decoding with CTC
resolving_unk=0
fwd_bwd_attention=false
checkpoint_ensemble=1  # the number of checkpoints to use
recog_unit=

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

if [ -z ${gpu} ]; then
  echo "Error: set GPU number." 1>&2
  echo "Usage: local/score.sh --gpu 0" 1>&2
  exit 1
fi
gpu=`echo ${gpu} | cut -d "," -f 1`

for set in eval1 eval2 eval3; do
  if [ ${ctc_weight} != 0.0 ]; then
    length_penalty=0.0
    coverage_penalty=0.0
  fi

  recog_dir=${model}/decode_${set}_ep${epoch}_beam${beam_width}_lp${length_penalty}_cp${coverage_penalty}_${min_len_ratio}_${max_len_ratio}_rnnlm${rnnlm_weight}
  if [ ${ctc_weight} != 0.0 ]; then
    recog_dir=${recog_dir}_ctc${ctc_weight}
  fi
  if [ ! -z ${recog_unit} ]; then
      recog_dir=${recog_dir}_${recog_unit}
  fi
  if ${fwd_bwd_attention}; then
    recog_dir=${recog_dir}_fwdbwd
  fi
  if [ ${checkpoint_ensemble} != 1 ]; then
    recog_dir=${recog_dir}_checkpoint${checkpoint_ensemble}
  fi
  if [ ! -z ${model7} ]; then
    recog_dir=${recog_dir}_ensemble8
  elif [ ! -z ${model6} ]; then
    recog_dir=${recog_dir}_ensemble7
  elif [ ! -z ${model5} ]; then
    recog_dir=${recog_dir}_ensemble6
  elif [ ! -z ${model4} ]; then
    recog_dir=${recog_dir}_ensemble5
  elif [ ! -z ${model3} ]; then
    recog_dir=${recog_dir}_ensemble4
  elif [ ! -z ${model2} ]; then
    recog_dir=${recog_dir}_ensemble3
  elif [ ! -z ${model1} ]; then
    recog_dir=${recog_dir}_ensemble2
  fi
  mkdir -p ${recog_dir}

  if [ `echo ${model} | grep 'train_sp'` ]; then
    if [ `echo ${model} | grep 'all'` ]; then
      recog_set=${data}/dataset/${set}_sp_all_wpbpe30000.csv
    elif [ `echo ${model} | grep 'aps_other'` ]; then
      recog_set=${data}/dataset/${set}_sp_aps_other_wpbpe10000.csv
    fi
  else
    if [ `echo ${model} | grep 'all'` ]; then
      recog_set=${data}/dataset/${set}_all_wpbpe30000.csv
    elif [ `echo ${model} | grep 'aps_other'` ]; then
      recog_set=${data}/dataset/${set}_aps_other_wpbpe10000.csv
    fi
  fi

  CUDA_VISIBLE_DEVICES=${gpu} ../../../neural_sp/bin/asr/eval.py \
    --recog_sets ${recog_set} \
    --recog_model ${model} ${model1} ${model2} ${model3} ${model4} ${model5} ${model6} ${model7} \
    --recog_model_bwd ${model_bwd} \
    --recog_epoch ${epoch} \
    --recog_batch_size ${batch_size} \
    --recog_beam_width ${beam_width} \
    --recog_max_len_ratio ${max_len_ratio} \
    --recog_min_len_ratio ${min_len_ratio} \
    --recog_length_penalty ${length_penalty} \
    --recog_coverage_penalty ${coverage_penalty} \
    --recog_coverage_threshold ${coverage_threshold} \
    --recog_rnnlm ${rnnlm} \
    --recog_rnnlm_bwd ${rnnlm_bwd} \
    --recog_rnnlm_weight ${rnnlm_weight} \
    --recog_ctc_weight ${ctc_weight} \
    --recog_resolving_unk ${resolving_unk} \
    --recog_fwd_bwd_attention ${fwd_bwd_attention} \
    --recog_checkpoint_ensemble ${checkpoint_ensemble} \
    --recog_unit ${recog_unit} \
    --recog_dir ${recog_dir} || exit 1;

  echo ${set}
  sclite -r ${recog_dir}/ref.trn trn -h ${recog_dir}/hyp.trn trn -i rm -o all stdout > ${recog_dir}/result.txt
  grep -e Avg -e SPKR -m 2 ${recog_dir}/result.txt > ${recog_dir}/RESULTS
  cat ${recog_dir}/RESULTS
done
