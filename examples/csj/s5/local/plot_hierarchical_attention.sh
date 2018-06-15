#!/bin/bash

. ./path.sh
set -e

### Select GPU
if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./plot_hierarchical_attention.sh path_to_saved_model gpu_id" 1>&2
  exit 1
fi

### Set path to save dataset
data=/n/sd8/inaguma/corpus/csj/kaldi

beam_width=4
beam_width_sub=4
length_penalty=0
coverage_penalty=0
rnnlm_weight=0.3
rnnlm_weight_sub=0.3
rnnlm_path=
# rnnlm_path=/n/sd8/inaguma/result/csj/pytorch/rnnlm/word/aps_other/lstm512H2Lemb128_adam_lr1e-3_drophidden0.2out0.2emb0.2
# rnnlm_path=/n/sd8/inaguma/result/csj/pytorch/rnnlm/word/all/lstm512H2Lemb128_adam_lr1e-3_drophidden0.2out0.2emb0.2_aps_other
rnnlm_path_sub=
# rnnlm_path_sub=/n/sd8/inaguma/result/csj/pytorch/rnnlm/character_wb/aps_other/lstm512H2Lemb64_adam_lr1e-3_drophidden0.2out0.2emb0.2
# rnnlm_path_sub=/n/sd8/inaguma/result/csj/pytorch/rnnlm/character_wb/all/lstm512H2Lemb64_adam_lr1e-3_drophidden0.2out0.2emb0.2_aps_other

CUDA_VISIBLE_DEVICES=$2 ${PYTHON} exp/visualization/plot_hierarchical_attention_weights.py \
  --data_save_path ${data} \
  --model_path $1 \
  --epoch -1 \
  --eval_batch_size 1 \
  --beam_width ${beam_width} \
  --beam_width_sub ${beam_width_sub} \
  --length_penalty ${length_penalty} \
  --coverage_penalty ${coverage_penalty} \
  --rnnlm_weight ${rnnlm_weight} \
  --rnnlm_path ${rnnlm_path} \
  --rnnlm_weight_sub ${rnnlm_weight_sub} \
  --rnnlm_path_sub ${rnnlm_path_sub}
