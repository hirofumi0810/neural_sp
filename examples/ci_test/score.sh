#!/bin/bash

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

model=
gpu=
stdout=false
n_threads=1

### path to save preproecssed data
data=./data

batch_size=1
beam_width=2
min_len_ratio=0.0
max_len_ratio=1.0
length_penalty=0.1
length_norm=false
coverage_penalty=0.0
coverage_threshold=0.0
gnmt_decoding=false
eos_threshold=1.0
lm=
lm_weight=0.5
ctc_weight=0.0  # 1.0 for joint CTC-attention means decoding with CTC
lm_state_carry_over=true
n_average=2  # for Transformer
oracle=false
block_sync=false  # for MoChA
block_size=40  # for MoChA
mma_delay_threshold=-1

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

if [ -z ${gpu} ]; then
    # CPU
    n_gpus=0
    export OMP_NUM_THREADS=${n_threads}
else
    n_gpus=$(echo ${gpu} | tr "," "\n" | wc -l)
fi

for set in train; do
    recog_dir=$(dirname ${model})/decode_${set}_beam${beam_width}_lp${length_penalty}_cp${coverage_penalty}_${min_len_ratio}_${max_len_ratio}
    if [ ${length_norm} = true ]; then
        recog_dir=${recog_dir}_norm
    fi
    if [ ! -z ${lm} ] && [ ${lm_weight} != 0 ]; then
        recog_dir=${recog_dir}_lm${lm_weight}
    fi
    if [ ${ctc_weight} != 0.0 ]; then
        recog_dir=${recog_dir}_ctc${ctc_weight}
    fi
    if [ ${gnmt_decoding} = true ]; then
        recog_dir=${recog_dir}_gnmt
    fi
    if [ ${block_sync} = true ]; then
        recog_dir=${recog_dir}_blocksync${block_size}
    fi
    if [ ${n_average} != 1 ]; then
        recog_dir=${recog_dir}_average${n_average}
    fi
    if [ ! -z ${lm} ] && [ ${lm_weight} != 0 ] && [ ${lm_state_carry_over} = true ]; then
        recog_dir=${recog_dir}_LMcarryover
    fi
    if [ ${mma_delay_threshold} != -1 ]; then
        recog_dir=${recog_dir}_epswait${mma_delay_threshold}
    fi
    mkdir -p ${recog_dir}

    CUDA_VISIBLE_DEVICES=${gpu} coverage run -a ${NEURALSP_ROOT}/neural_sp/bin/asr/eval.py \
        --recog_n_gpus ${n_gpus} \
        --recog_sets ${data}/dataset/${set}_char.tsv \
        --recog_dir ${recog_dir} \
        --recog_model ${model} \
        --recog_batch_size ${batch_size} \
        --recog_beam_width ${beam_width} \
        --recog_max_len_ratio ${max_len_ratio} \
        --recog_min_len_ratio ${min_len_ratio} \
        --recog_length_penalty ${length_penalty} \
        --recog_length_norm ${length_norm} \
        --recog_coverage_penalty ${coverage_penalty} \
        --recog_coverage_threshold ${coverage_threshold} \
        --recog_gnmt_decoding ${gnmt_decoding} \
        --recog_eos_threshold ${eos_threshold} \
        --recog_lm ${lm} \
        --recog_lm_weight ${lm_weight} \
        --recog_ctc_weight ${ctc_weight} \
        --recog_lm_state_carry_over ${lm_state_carry_over} \
        --recog_block_sync ${block_sync} \
        --recog_block_sync_size ${block_size} \
        --recog_n_average ${n_average} \
        --recog_mma_delay_threshold ${mma_delay_threshold} \
        --recog_stdout ${stdout} || exit 1;
done
