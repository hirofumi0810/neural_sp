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
vocab_size=30000
wp_type=bpe  # or unigram (for wordpiece)

#########################
# ASR configuration
#########################
### topology
n_splices=1
n_stacks=1
n_skips=1
conv_in_channel=1
conv_channels=
conv_kernel_sizes=
conv_strides=
conv_poolings=
conv_batch_norm=
subsample="1_2_2_2_1"
# VGG
# conv_channels="64_64_128_128"
# conv_kernel_sizes="(3,3)_(3,3)_(3,3)_(3,3)"
# conv_strides="(1,1)_(1,1)_(1,1)_(1,1)"
# conv_poolings="(1,1)_(2,2)_(1,1)_(2,2)"
# subsample="1_1_1_1_1"
enc_type=blstm
enc_n_units=320
enc_n_projs=0
enc_n_layers=5
enc_residual=
enc_add_ffl=
subsample_type=drop
attn_type=location
attn_dim=320
attn_n_heads=1
attn_sigmoid=
dec_type=lstm
dec_n_units=320
dec_n_projs=0
dec_n_layers=1
dec_loop_type=normal
dec_residual=
dec_add_ffl=
dec_layerwise_attention=
input_feeding=
emb_dim=320
tie_embedding=
ctc_fc_list="320"
### optimization
batch_size=50
optimizer=adam
learning_rate=1e-3
n_epochs=30
convert_to_sgd_epoch=25
print_step=1000
decay_start_epoch=10
decay_rate=0.9
decay_patient_n_epochs=0
decay_type=epoch
not_improved_patient_n_epochs=5
eval_start_epoch=1
warmup_start_learning_rate=1e-4
warmup_n_steps=0
warmup_n_epochs=0
### initialization
param_init=0.1
param_init_dist=uniform
pretrained_model=
### regularization
clip_grad_norm=5.0
dropout_in=0.0
dropout_enc=0.4
dropout_dec=0.4
dropout_emb=0.4
dropout_att=0.0
weight_decay=1e-6
ss_prob=0.2
ss_type=constant
lsm_prob=0.1
layer_norm=
focal_loss=0.0
### MTL
ctc_weight=0.0
bwd_weight=0.0
mtl_per_batch=true
task_specific_layer=
### LM integration
lm_fusion_type=hidden
rnnlm_fusion=
rnnlm_init=
lmobj_weight=0.0
share_lm_softmax=
# contextualization
concat_prev_n_utterances=0
n_caches=0

#########################
# RNNLM configuration
#########################
# topology
lm_rnn_type=lstm
lm_n_units=1024
lm_n_projs=0
lm_n_layers=2
lm_emb_dim=1024
lm_tie_embedding=true
lm_residual=true
lm_use_glu=true
# optimization
lm_batch_size=64
lm_bptt=200
lm_optimizer=adam
lm_learning_rate=1e-3
lm_n_epochs=40
lm_convert_to_sgd_epoch=40
lm_print_step=50
lm_decay_start_epoch=10
lm_decay_rate=0.9
lm_decay_patient_n_epochs=0
lm_not_improved_patient_n_epochs=5
lm_eval_start_epoch=1
# initialization
lm_param_init=0.05
lm_param_init_dist=uniform
lm_pretrained_model=
# regularization
lm_clip_grad_norm=1.0
lm_dropout_hidden=0.5
lm_dropout_out=0.0
lm_dropout_emb=0.2
lm_weight_decay=1e-6
lm_backward=
# contextualization
lm_serialize=true

### path to save the model
model=/n/sd8/inaguma/result/swbd

### path to the model directory to resume training
resume=
rnnlm_resume=

### path to original data
SWBD_AUDIOPATH=/n/rd21/corpora_7/swb
EVAL2000_AUDIOPATH=/n/rd21/corpora_7/hub5_english/LDC2002S09
EVAL2000_TRANSPATH=/n/rd21/corpora_7/hub5_english/LDC2002T43
RT03_PATH=
FISHER_PATH=/n/rd7/fisher_english

### data size
data_size=fisher_swbd
lm_data_size=fisher_swbd

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
rnnlm_gpu=$(echo ${gpu} | cut -d "," -f 1)

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

    if [ ! -e ${data}/.done_stage_0_swbd ]; then
        echo "run ./run.sh first" && exit 1
    fi

    # prepare fisher data and put it under data/train_fisher
    local/fisher_data_prep.sh ${FISHER_PATH}
    local/fisher_swbd_prepare_dict.sh
    utils/fix_data_dir.sh ${data}/train_fisher

    # nomalization
    cp ${data}/train_fisher/text ${data}/train_fisher/text.tmp.0
    cut -f 2- -d " " ${data}/train_fisher/text.tmp.0 | \
        sed -e 's/\[laughter\]-/[laughter]/g' |
    sed -e 's/\[noise\]-/[noise]/g' > ${data}/train_fisher/text.tmp.1

    paste -d " " <(cut -f 1 -d " " ${data}/train_fisher/text.tmp.0) \
        <(cat ${data}/train_fisher/text.tmp.1) > ${data}/train_fisher/text
    rm ${data}/train_fisher/text.tmp*

    touch ${data}/.done_stage_0_${data_size} && echo "Finish data preparation (stage: 0)."
fi

if [ ${stage} -le 1 ] && [ ! -e ${data}/.done_stage_1_${data_size} ]; then
    echo ============================================================================
    echo "                    Feature extranction (stage:1)                          "
    echo ============================================================================

    if [ ! -e ${data}/.done_stage_1_swbd ]; then
        echo "run ./run.sh first" && exit 1
    fi

    steps/make_fbank.sh --nj 32 --cmd "$train_cmd" --write_utt2num_frames true \
        ${data}/train_fisher ${data}/log/make_fbank/train_fisher ${data}/fbank || exit 1;

    utils/combine_data.sh --extra_files "utt2num_frames" ${data}/${train_set} ${data}/train_swbd ${data}/train_fisher || exit 1;
    cp -rf ${data}/dev_swbd ${data}/${dev_set}

    # Compute global CMVN
    compute-cmvn-stats scp:${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark || exit 1;

    # Apply global CMVN & dump features
    dump_feat.sh --cmd "$train_cmd" --nj 2000 \
        ${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${train_set} ${data}/dump/${train_set} || exit 1;
    dump_feat.sh --cmd "$train_cmd" --nj 32 \
        ${data}/${dev_set}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${dev_set} ${data}/dump/${dev_set} || exit 1;
    for x in ${test_set}; do
        dump_dir=${data}/dump/${x}_${data_size}
        dump_feat.sh --cmd "$train_cmd" --nj 32 \
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
    offset=$(cat ${dict} | wc -l)
    echo "Making a dictionary..."
    if [ ${unit} = wp ]; then
        cut -f 2- -d " " ${data}/${train_set}/text > ${data}/dict/input.txt
        spm_train --user_defined_symbols=$(cat ${nlsyms} | tr "\n" ",") --input=${data}/dict/input.txt --vocab_size=${vocab_size} \
            --model_type=${wp_type} --model_prefix=${wp_model} --input_sentence_size=100000000 --character_coverage=1.0
        spm_encode --model=${wp_model}.model --output_format=piece < ${data}/dict/input.txt | tr ' ' '\n' | \
            sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}
    else
        text2dict.py ${data}/${train_set}/text --unit ${unit} --vocab_size ${vocab_size} --nlsyms ${nlsyms} \
            --wp_type ${wp_type} --wp_model ${wp_model} | \
            sort | uniq | grep -v -e '^\s*$' | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict} || exit 1;
    fi
    echo "vocab size:" $(cat ${dict} | wc -l)

    # Compute OOV rate
    if [ ${unit} = word ]; then
        mkdir -p ${data}/dict/word_count ${data}/dict/oov_rate
        echo "OOV rate:" > ${data}/dict/oov_rate/word${vocab_size}_${data_size}.txt
        for x in ${train_set} ${dev_set}; do
            cut -f 2- -d " " ${data}/${x}/text | tr " " "\n" | sort | uniq -c | sort -n -k1 -r \
                > ${data}/dict/word_count/${x}_${data_size}.txt || exit 1;
            compute_oov_rate.py ${data}/dict/word_count/${x}_${data_size}.txt ${dict} ${x} \
                >> ${data}/dict/oov_rate/word${vocab_size}_${data_size}.txt || exit 1;
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
            >> ${data}/dict/oov_rate/word${vocab_size}_${data_size}.txt || exit 1;
        # callhome
        grep -v sw ${data}/${test_set}/text | cut -f 2- -d " " | awk '{ print tolower($0) }' | \
            perl -pe 's| \(\%.*\)||g' | perl -pe 's| \<.*\>||g' | sed -e "s/(//g" -e "s/)//g" | sed -e 's/\s\+/ /g' | \
            tr " " "\n" | sort | uniq -c | sort -n -k1 -r \
            > ${data}/dict/word_count/${test_set}_callhm.txt || exit 1;
        compute_oov_rate.py ${data}/dict/word_count/${test_set}_callhm.txt ${dict} ${test_set}_callhm \
            >> ${data}/dict/oov_rate/word${vocab_size}_${data_size}.txt || exit 1;
        cat ${data}/dict/oov_rate/word${vocab_size}_${data_size}.txt
    fi

    # Make datset tsv files for the ASR task
    echo "Making dataset tsv files for ASR ..."
    mkdir -p ${data}/dataset
    for x in ${train_set} ${dev_set}; do
        dump_dir=${data}/dump/${x}
        make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit} --nlsyms ${nlsyms} --wp_model ${wp_model} \
            ${data}/${x} ${dict} > ${data}/dataset/${x}_${unit}${wp_type}${vocab_size}.tsv || exit 1;
    done
    for x in ${test_set}; do
        dump_dir=${data}/dump/${x}_${data_size}
        make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit} --nlsyms ${nlsyms} --wp_model ${wp_model} \
            ${data}/${x} ${dict} > ${data}/dataset/${x}_${data_size}_${unit}${wp_type}${vocab_size}.tsv || exit 1;
    done

    touch ${data}/.done_stage_2_${data_size}_${unit}${wp_type}${vocab_size} && echo "Finish creating dataset for ASR (stage: 2)."
fi

mkdir -p ${model}
if [ ${stage} -le 3 ]; then
    echo ============================================================================
    echo "                      RNNLM Training stage (stage:3)                       "
    echo ============================================================================

    if [ ! -e ${data}/.done_stage_3_${lm_data_size}_${unit}${wp_type}${vocab_size} ]; then
        # Make datset tsv files for the LM task
        echo "Making dataset tsv files for LM ..."
        mkdir -p ${data}/dataset_lm
        for x in train_${lm_data_size} dev_${lm_data_size}; do
            cp ${data}/dataset/${x}_${unit}${wp_type}${vocab_size}.tsv \
                ${data}/dataset_lm/${x}_${train_set}_${unit}${wp_type}${vocab_size}.tsv || exit 1;
        done

        # normlization for evasl2000 sets
        for x in ${test_set}; do
            cp ${data}/${test_set}/text ${data}/${test_set}/text.tmp.0
            cut -f 2- -d " " ${data}/${test_set}/text.tmp.0 | awk '{ print tolower($0) }' | \
                perl -pe 's| \(\%.*\)||g' | perl -pe 's| \<.*\>||g' | sed -e "s/(//g" -e "s/)//g" | sed -e 's/\s\+/ /g' \
                > ${data}/${test_set}/text.tmp.1
            paste -d " " <(cut -f 1 -d " " ${data}/${test_set}/text.tmp.0) \
                <(cat ${data}/${test_set}/text.tmp.1) > ${data}/${test_set}/text.lm
            rm ${data}/${test_set}/text.tmp*

            grep -v en ${data}/${test_set}/text.lm > ${data}/${test_set}/text.lm.swbd
            grep -v sw ${data}/${test_set}/text.lm > ${data}/${test_set}/text.lm.ch

            make_dataset.sh --unit ${unit} --nlsyms ${nlsyms} --wp_model ${wp_model} --text ${data}/${test_set}/text.lm.swbd \
                ${data}/${x} ${dict} > ${data}/dataset_lm/${x}_swbd_${lm_data_size}_${train_set}_${unit}${wp_type}${vocab_size}.tsv || exit 1;
            make_dataset.sh --unit ${unit} --nlsyms ${nlsyms} --wp_model ${wp_model} --text ${data}/${test_set}/text.lm.ch \
                ${data}/${x} ${dict} > ${data}/dataset_lm/${x}_ch_${lm_data_size}_${train_set}_${unit}${wp_type}${vocab_size}.tsv || exit 1;
        done

        touch ${data}/.done_stage_3_${lm_data_size}_${unit}${wp_type}${vocab_size} && echo "Finish creating dataset for LM (stage: 3)."
    fi

    lm_test_set="${data}/dataset_lm/${test_set}_swbd_${lm_data_size}_${train_set}_${unit}${wp_type}${vocab_size}.tsv \
                 ${data}/dataset_lm/${test_set}_ch_${lm_data_size}_${train_set}_${unit}${wp_type}${vocab_size}.tsv"

    # NOTE: support only a single GPU for RNNLM training
    CUDA_VISIBLE_DEVICES=${rnnlm_gpu} ${NEURALSP_ROOT}/neural_sp/bin/lm/train.py \
        --corpus swbd \
        --n_gpus 1 \
        --train_set ${data}/dataset_lm/train_${lm_data_size}_${train_set}_${unit}${wp_type}${vocab_size}.tsv \
        --dev_set ${data}/dataset_lm/dev_${lm_data_size}_${train_set}_${unit}${wp_type}${vocab_size}.tsv \
        --eval_sets ${lm_test_set} \
        --dict ${dict} \
        --wp_model ${wp_model}.model \
        --model ${model}/rnnlm \
        --unit ${unit} \
        --rnn_type ${lm_rnn_type} \
        --n_units ${lm_n_units} \
        --n_projs ${lm_n_projs} \
        --n_layers ${lm_n_layers} \
        --emb_dim ${lm_emb_dim} \
        --tie_embedding ${lm_tie_embedding} \
        --residual ${lm_residual} \
        --use_glu ${lm_use_glu} \
        --batch_size ${lm_batch_size} \
        --bptt ${lm_bptt} \
        --optimizer ${lm_optimizer} \
        --learning_rate ${lm_learning_rate} \
        --n_epochs ${lm_n_epochs} \
        --convert_to_sgd_epoch ${lm_convert_to_sgd_epoch} \
        --print_step ${lm_print_step} \
        --decay_start_epoch ${lm_decay_start_epoch} \
        --decay_rate ${lm_decay_rate} \
        --decay_patient_n_epochs ${lm_decay_patient_n_epochs} \
        --not_improved_patient_n_epochs ${lm_not_improved_patient_n_epochs} \
        --eval_start_epoch ${lm_eval_start_epoch} \
        --param_init ${lm_param_init} \
        --param_init_dist ${lm_param_init_dist} \
        --pretrained_model ${lm_pretrained_model} \
        --clip_grad_norm ${lm_clip_grad_norm} \
        --dropout_hidden ${lm_dropout_hidden} \
        --dropout_out ${lm_dropout_out} \
        --dropout_emb ${lm_dropout_emb} \
        --weight_decay ${lm_weight_decay} \
        --backward ${lm_backward} \
        --serialize ${lm_serialize} || exit 1;
    # --resume ${rnnlm_resume} || exit 1;

    echo "Finish RNNLM training (stage: 3)." && exit 1;
fi

if [ ${stage} -le 4 ]; then
    echo ============================================================================
    echo "                       ASR Training stage (stage:4)                        "
    echo ============================================================================

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/asr/train.py \
        --corpus swbd \
        --n_gpus ${n_gpus} \
        --train_set ${data}/dataset/${train_set}_${unit}${wp_type}${vocab_size}.tsv \
        --dev_set ${data}/dataset/${dev_set}_${unit}${wp_type}${vocab_size}.tsv \
        --dict ${dict} \
        --wp_model ${wp_model}.model \
        --model ${model}/asr \
        --unit ${unit} \
        --n_splices ${n_splices} \
        --n_stacks ${n_stacks} \
        --n_skips ${n_skips} \
        --conv_in_channel ${conv_in_channel} \
        --conv_channels ${conv_channels} \
        --conv_kernel_sizes ${conv_kernel_sizes} \
        --conv_strides ${conv_strides} \
        --conv_poolings ${conv_poolings} \
        --conv_batch_norm ${conv_batch_norm} \
        --enc_type ${enc_type} \
        --enc_n_units ${enc_n_units} \
        --enc_n_projs ${enc_n_projs} \
        --enc_n_layers ${enc_n_layers} \
        --enc_residual ${enc_residual} \
        --enc_add_ffl ${enc_add_ffl} \
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
        --dec_add_ffl ${dec_add_ffl} \
        --dec_layerwise_attention ${dec_layerwise_attention} \
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
        --warmup_n_epochs ${warmup_n_epochs} \
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
        --lm_fusion_type ${lm_fusion_type} \
        --rnnlm_fusion ${rnnlm_fusion} \
        --rnnlm_init ${rnnlm_init} \
        --lmobj_weight ${lmobj_weight} \
        --share_lm_softmax ${share_lm_softmax} \
        --concat_prev_n_utterances ${concat_prev_n_utterances} \
        --n_caches ${n_caches} \
        --resume ${resume} || exit 1;

    echo "Finish model training (stage: 4)."
fi
