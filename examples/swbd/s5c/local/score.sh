#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

model=
model1=
model2=
model3=
model_bwd=
gpu=
stdout=false

### path to save preproecssed data
data=/n/work1/inaguma/corpus/swbd

unit=
metric=edit_distance
batch_size=1
beam_width=5
min_len_ratio=0.0
max_len_ratio=1.0
length_penalty=0.0
length_norm=false
coverage_penalty=0.0
coverage_threshold=0.0
gnmt_decoding=false
eos_threshold=1.0
lm=
lm_second=
lm_bwd=
lm_weight=0.2
lm_second_weight=0.2
lm_bwd_weight=0.2
ctc_weight=0.0  # 1.0 for joint CTC-attention means decoding with CTC
resolving_unk=false
fwd_bwd_attention=false
bwd_attention=false
reverse_lm_rescoring=false
asr_state_carry_over=false
lm_state_carry_over=true
n_average=1  # for Transformer
oracle=false
chunk_sync=false  # for MoChA

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

if [ -z ${gpu} ]; then
    n_gpus=0
else
    n_gpus=$(echo ${gpu} | tr "," "\n" | wc -l)
fi

# for set in eval2000 rt03; do
for set in eval2000; do
    recog_dir=$(dirname ${model})/decode_${set}_beam${beam_width}_lp${length_penalty}_cp${coverage_penalty}_${min_len_ratio}_${max_len_ratio}
    if [ ! -z ${unit} ]; then
        recog_dir=${recog_dir}_${unit}
    fi
    if [ ${length_norm} = true ]; then
        recog_dir=${recog_dir}_norm
    fi
    if [ ${metric} != 'edit_distance' ]; then
        recog_dir=${recog_dir}_${metric}
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
    if [ ${gnmt_decoding} = true ]; then
        recog_dir=${recog_dir}_gnmt
    fi
    if [ ${resolving_unk} = true ]; then
        recog_dir=${recog_dir}_resolvingOOV
    fi
    if [ ${fwd_bwd_attention} = true ]; then
        recog_dir=${recog_dir}_fwdbwd
    fi
    if [ ${bwd_attention} = true ]; then
        recog_dir=${recog_dir}_bwd
    fi
    if [ ${reverse_lm_rescoring} = true ]; then
        recog_dir=${recog_dir}_revLM
    fi
    if [ ${asr_state_carry_over} = true ]; then
        recog_dir=${recog_dir}_ASRcarryover
    fi
    if [ ${n_average} != 1 ]; then
        recog_dir=${recog_dir}_average${n_average}
    fi
    if [ ! -z ${lm} ] && [ ${lm_weight} != 0 ] && [ ${lm_state_carry_over} = true ]; then
        recog_dir=${recog_dir}_LMcarryover
    fi
    if [ ${oracle} = true ]; then
        recog_dir=${recog_dir}_oracle
    fi
    if [ ! -z ${model3} ]; then
        recog_dir=${recog_dir}_ensemble4
    elif [ ! -z ${model2} ]; then
        recog_dir=${recog_dir}_ensemble3
    elif [ ! -z ${model1} ]; then
        recog_dir=${recog_dir}_ensemble2
    fi
    mkdir -p ${recog_dir}

    if [ $(echo ${model} | grep 'train_sp') ]; then
        recog_set=${data}/dataset/${set}_sp_swbd_wpbpe10000.tsv
    else
        if [ $(echo ${model} | grep 'fisher_swbd') ]; then
            # recog_set=${data}/dataset/${set}_fisher_swbd_wpbpe34000.tsv
            recog_set=${data}/dataset/${set}_fisher_swbd_wpbpe10000.tsv
        else
            recog_set=${data}/dataset/${set}_swbd_wpbpe10000.tsv
        fi
    fi

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/asr/eval.py \
        --recog_n_gpus ${n_gpus} \
        --recog_sets ${recog_set} \
        --recog_dir ${recog_dir} \
        --recog_unit ${unit} \
        --recog_metric ${metric} \
        --recog_model ${model} ${model1} ${model2} ${model3} \
        --recog_model_bwd ${model_bwd} \
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
        --recog_resolving_unk ${resolving_unk} \
        --recog_fwd_bwd_attention ${fwd_bwd_attention} \
        --recog_bwd_attention ${bwd_attention} \
        --recog_reverse_lm_rescoring ${reverse_lm_rescoring} \
        --recog_asr_state_carry_over ${asr_state_carry_over} \
        --recog_lm_state_carry_over ${lm_state_carry_over} \
        --recog_n_average ${n_average} \
        --recog_oracle ${oracle} \
        --recog_stdout ${stdout} || exit 1;

    if [ ${metric} = 'edit_distance' ]; then
        echo ${set}
        if [ ${set} = 'dev' ]; then
            sclite -r ${recog_dir}/ref.trn trn -h ${recog_dir}/hyp.trn trn -i rm -o all stdout > ${recog_dir}/result.txt
            grep -e Avg -e SPKR -m 2 ${recog_dir}/result.txt > ${recog_dir}/RESULTS
        else
            local/score_sclite.sh ${data} ${recog_dir} ${set} > ${recog_dir}/RESULTS
        fi
        cat ${recog_dir}/RESULTS
    fi
done
