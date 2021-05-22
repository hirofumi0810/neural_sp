#!/usr/bin/env bash

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

model=
model1=
model2=
model3=
gpu=
stdout=false
n_threads=1
eval_set="train"
cmd_coverage="coverage run -a"

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
lm_second=
lm_bwd=
lm_weight=0.5
lm_second_weight=0.3
lm_bwd_weight=0.3
ctc_weight=0.0  # 1.0 for joint CTC-attention means decoding with CTC
softmax_smoothing=0.7  ###
lm_state_carry_over=true
n_average=2  # for Transformer
oracle=false
longform_max_n_frames=0
mma_delay_threshold=-1  # for MMA

# for streaming
streaming_encoding=false
block_sync=false
block_size=40
vad_free=false
blank_threshold=40  # 400ms
spike_threshold=0.1
n_accum_frames=1600  # 16s

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

for set in ${eval_set}; do
    recog_dir=$(dirname ${model})/decode_${set}_beam${beam_width}_lp${length_penalty}_cp${coverage_penalty}_${min_len_ratio}_${max_len_ratio}
    if [ ${length_norm} = true ]; then
        recog_dir=${recog_dir}_norm
    fi
    if [ ! -z ${lm} ] && [ ${lm_weight} != 0 ]; then
        recog_dir=${recog_dir}_lm${lm_weight}
    fi
    if [ ! -z ${lm_second} ] && [ ${lm_second_weight} != 0 ]; then
        recog_dir=${recog_dir}_rescore${lm_second_weight}
    fi
    if [ ! -z ${lm_bwd} ] && [ ${lm_bwd_weight} != 0 ]; then
        recog_dir=${recog_dir}_bwd${lm_bwd_weight}
    fi
    if [ ${ctc_weight} != 0.0 ]; then
        recog_dir=${recog_dir}_ctc${ctc_weight}
    fi
    if [ ${softmax_smoothing} != 1.0 ]; then
        recog_dir=${recog_dir}_smooth${softmax_smoothing}
    fi
    if [ ${gnmt_decoding} = true ]; then
        recog_dir=${recog_dir}_gnmt
    fi
    if [ ${longform_max_n_frames} != 0 ]; then
        recog_dir=${recog_dir}_longform${longform_max_n_frames}
    fi
    if [ ${streaming_encoding} = true ]; then
        recog_dir=${recog_dir}_streaming_encoding${block_size}
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
    if [ ! -z ${model3} ]; then
        recog_dir=${recog_dir}_ensemble4
    elif [ ! -z ${model2} ]; then
        recog_dir=${recog_dir}_ensemble3
    elif [ ! -z ${model1} ]; then
        recog_dir=${recog_dir}_ensemble2
    fi
    mkdir -p ${recog_dir}

    CUDA_VISIBLE_DEVICES=${gpu} ${cmd_coverage} ${NEURALSP_ROOT}/neural_sp/bin/asr/eval.py \
        --recog_n_gpus ${n_gpus} \
        --recog_sets ${data}/dataset/${set}_char.tsv \
        --recog_dir ${recog_dir} \
        --recog_model ${model} ${model1} ${model2} ${model3} \
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
        --recog_lm_second ${lm_second} \
        --recog_lm_bwd ${lm_bwd} \
        --recog_lm_weight ${lm_weight} \
        --recog_lm_second_weight ${lm_second_weight} \
        --recog_lm_bwd_weight ${lm_bwd_weight} \
        --recog_ctc_weight ${ctc_weight} \
        --recog_softmax_smoothing ${softmax_smoothing} \
        --recog_lm_state_carry_over ${lm_state_carry_over} \
        --recog_n_average ${n_average} \
        --recog_longform_max_n_frames ${longform_max_n_frames} \
        --recog_streaming_encoding ${streaming_encoding} \
        --recog_block_sync ${block_sync} \
        --recog_block_sync_size ${block_size} \
        --recog_mma_delay_threshold ${mma_delay_threshold} \
        --recog_ctc_vad ${vad_free} \
        --recog_ctc_vad_blank_threshold ${blank_threshold} \
        --recog_ctc_vad_spike_threshold ${spike_threshold} \
        --recog_ctc_vad_n_accum_frames ${n_accum_frames} \
        --recog_stdout ${stdout} || exit 1;
done
