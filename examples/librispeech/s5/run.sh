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
unit=wp      # word/wp/char/word_char
vocab_size=30000
wp_type=bpe  # bpe/unigram (for wordpiece)

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
conv_batch_norm=false
conv_bottleneck_dim=0
subsample="1_2_2_2_1"
# VGG
# conv_channels="64_64_128_128"
# conv_kernel_sizes="(3,3)_(3,3)_(3,3)_(3,3)"
# conv_strides="(1,1)_(1,1)_(1,1)_(1,1)"
# conv_poolings="(1,1)_(2,2)_(1,1)_(2,2)"
# subsample="1_1_1_1_1"
enc_type=blstm
enc_n_units=512
enc_n_projs=0
enc_n_layers=5
enc_residual=false
subsample_type=drop
attn_type=location
attn_dim=512
attn_n_heads=1
attn_sigmoid=false
dec_type=lstm
dec_n_units=1024
dec_n_projs=0
dec_n_layers=1
dec_loop_type=normal
dec_residual=false
input_feeding=false
emb_dim=512
tie_embedding=false
ctc_fc_list="512"
### optimization
batch_size=50
optimizer=adam
learning_rate=1e-3
n_epochs=25
convert_to_sgd_epoch=25
print_step=500
decay_start_epoch=10
decay_rate=0.85
decay_patient_n_epochs=0
decay_type=epoch
not_improved_patient_n_epochs=5
eval_start_epoch=1
warmup_start_learning_rate=1e-4
warmup_n_steps=4000
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
layer_norm=false
focal_loss=0.0
### MTL
ctc_weight=0.0
bwd_weight=0.0
mtl_per_batch=true
task_specific_layer=false
### LM integration
lm_fusion_type=cold
lm_fusion=
lm_init=
lmobj_weight=0.0
share_lm_softmax=false
# contextualization
n_caches=0

#########################
# LM configuration
#########################
# topology
lm_type=lstm
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

### path to save the model
model=/n/sd8/inaguma/result/librispeech

### path to the model directory to resume training
resume=
lm_resume=

### path to download data
data_download_path=/n/rd21/corpora_7/librispeech/

### data size
data_size=960     # 100/460/969
lm_data_size=960  # default is the same data as ASR

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
lm_gpu=$(echo ${gpu} | cut -d "," -f 1)

# Base url for downloads.
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11

train_set=train_${data_size}
dev_set=dev_${data_size}
test_set="dev_clean dev_other test_clean test_other"

if [ ${unit} = char ]; then
    vocab_size=
fi
if [ ${unit} != wp ]; then
    wp_type=
fi

if [ ${stage} -le 0 ] && [ ! -e ${data}/.done_stage_0 ]; then
    echo ============================================================================
    echo "                       Data Preparation (stage:0)                          "
    echo ============================================================================

    # download data
    mkdir -p ${data}
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        local/download_and_untar.sh ${data_download_path} ${data_url} ${part} || exit 1;
    done

    # download the LM resources
    local/download_lm.sh ${lm_url} ${data}/local/lm || exit 1;

    # format the data as Kaldi data directories
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        # use underscore-separated names in data directories.
        local/data_prep.sh ${data_download_path}/LibriSpeech/${part} ${data}/$(echo ${part} | sed s/-/_/g) || exit 1;
    done

    # lowercasing
    for x in dev_clean test_clean dev_other test_other train_clean_100 train_clean_360 train_other_500; do
        cp ${data}/${x}/text ${data}/${x}/text.tmp
        paste -d "" <(cut -f 1 -d " " ${data}/${x}/text.tmp) \
            <(cut -f 2- -d " " ${data}/${x}/text.tmp | awk '{$1=""; print tolower($0)}') > ${data}/${x}/text
        rm ${data}/${x}/text.tmp
    done

    touch ${data}/.done_stage_0 && echo "Finish data preparation (stage: 0)."
fi

if [ ${stage} -le 1 ] && [ ! -e ${data}/.done_stage_1_${data_size} ]; then
    echo ============================================================================
    echo "                    Feature extranction (stage:1)                          "
    echo ============================================================================

    for x in dev_clean test_clean dev_other test_other train_clean_100 train_clean_360 train_other_500; do
        steps/make_fbank.sh --nj 32 --cmd "$train_cmd" --write_utt2num_frames true \
            ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1;
    done

    if [ ${data_size} == '100' ]; then
        utils/combine_data.sh --extra_files "utt2num_frames" ${data}/${train_set} \
            ${data}/train_clean_100 || exit 1;
    elif [ ${data_size} == '460' ]; then
        utils/combine_data.sh --extra_files "utt2num_frames" ${data}/${train_set} \
            ${data}/train_clean_100 ${data}/train_clean_360 || exit 1;
    elif [ ${data_size} == '960' ]; then
        utils/combine_data.sh --extra_files "utt2num_frames" ${data}/${train_set} \
            ${data}/train_clean_100 ${data}/train_clean_360 ${data}/train_other_500 || exit 1;
    else
        echo "data_size is 100 or 460 or 960." && exit 1;
    fi
    cp -rf ${data}/dev_clean ${data}/${dev_set}

    # Compute global CMVN
    compute-cmvn-stats scp:${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark || exit 1;

    # Apply global CMVN & dump features
    dump_feat.sh --cmd "$train_cmd" --nj 800 \
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
wp_model=${data}/dict/${train_set}_${wp_type}${vocab_size}
if [ ${stage} -le 2 ] && [ ! -e ${data}/.done_stage_2_${data_size}_${unit}${wp_type}${vocab_size} ]; then
    echo ============================================================================
    echo "                      Dataset preparation (stage:2)                        "
    echo ============================================================================

    echo "Making a dictionary..."
    echo "<unk> 1" > ${dict}  # <unk> must be 1, 0 will be used for "blank" in CTC
    echo "<eos> 2" >> ${dict}  # <sos> and <eos> share the same index
    echo "<pad> 3" >> ${dict}
    if [ ${unit} = char ]; then
        echo "<space> 4" >> ${dict}
    fi
    offset=$(cat ${dict} | wc -l)
    if [ ${unit} = wp ]; then
        cut -f 2- -d " " ${data}/${train_set}/text > ${data}/dict/input.txt
        spm_train --input=${data}/dict/input.txt --vocab_size=${vocab_size} \
            --model_type=${wp_type} --model_prefix=${wp_model} --input_sentence_size=100000000 --character_coverage=1.0
        spm_encode --model=${wp_model}.model --output_format=piece < ${data}/dict/input.txt | tr ' ' '\n' | \
            sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}
    else
        text2dict.py ${data}/${train_set}/text --unit ${unit} --vocab_size ${vocab_size} \
            --wp_type ${wp_type} --wp_model ${wp_model} | \
            sort | uniq | grep -v -e '^\s*$' | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict} || exit 1;
    fi
    echo "vocab size:" $(cat ${dict} | wc -l)

    # Compute OOV rate
    if [ ${unit} = word ]; then
        mkdir -p ${data}/dict/word_count ${data}/dict/oov_rate
        echo "OOV rate:" > ${data}/dict/oov_rate/word${vocab_size}_${data_size}.txt
        for x in ${train_set} ${dev_set} ${test_set}; do
            cut -f 2- -d " " ${data}/${x}/text | tr " " "\n" | sort | uniq -c | sort -n -k1 -r \
                > ${data}/dict/word_count/${x}_${data_size}.txt || exit 1;
            compute_oov_rate.py ${data}/dict/word_count/${x}_${data_size}.txt ${dict} ${x} \
                >> ${data}/dict/oov_rate/word${vocab_size}_${data_size}.txt || exit 1;
        done
        cat ${data}/dict/oov_rate/word${vocab_size}_${data_size}.txt
    fi

    echo "Making dataset tsv files for ASR ..."
    mkdir -p ${data}/dataset
    make_dataset.sh --feat ${data}/dump/${train_set}/feats.scp --unit ${unit} --wp_model ${wp_model} \
        ${data}/${train_set} ${dict} > ${data}/dataset/${train_set}_${unit}${wp_type}${vocab_size}.tsv || exit 1;
    make_dataset.sh --feat ${data}/dump/${dev_set}/feats.scp --unit ${unit} --wp_model ${wp_model} \
        ${data}/${dev_set} ${dict} > ${data}/dataset/${dev_set}_${unit}${wp_type}${vocab_size}.tsv || exit 1;
    for x in ${test_set}; do
        dump_dir=${data}/dump/${x}_${data_size}
        make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit} --wp_model ${wp_model} \
            ${data}/${x} ${dict} > ${data}/dataset/${x}_${data_size}_${unit}${wp_type}${vocab_size}.tsv || exit 1;
    done

    touch ${data}/.done_stage_2_${data_size}_${unit}${wp_type}${vocab_size} && echo "Finish creating dataset for ASR (stage: 2)."
fi

mkdir -p ${model}
if [ ${stage} -le 3 ]; then
    echo ============================================================================
    echo "                        LM Training stage (stage:3)                       "
    echo ============================================================================

    if [ ! -e ${data}/.done_stage_3_${data_size}${lm_data_size}_${unit}${wp_type}${vocab_size} ]; then
        if [ ! -e ${data}/.done_stage_1_${data_size} ]; then
            echo "run ./run.sh --data_size ${lm_data_size} first" && exit 1
        fi

        echo "Making dataset tsv files for LM ..."
        mkdir -p ${data}/dataset_lm
        for x in train_${lm_data_size} dev_${lm_data_size}; do
            if [ ${lm_data_size} = ${data_size} ]; then
                cp ${data}/dataset/${x}_${unit}${wp_type}${vocab_size}.tsv \
                    ${data}/dataset_lm/${x}_${train_set}_${unit}${wp_type}${vocab_size}.tsv || exit 1;
            else
                make_dataset.sh --unit ${unit} --wp_model ${wp_model} \
                    ${data}/${x} ${dict} > ${data}/dataset_lm/${x}_${train_set}_${unit}${wp_type}${vocab_size}.tsv || exit 1;
            fi
        done

        touch ${data}/.done_stage_3_${data_size}${lm_data_size}_${unit}${wp_type}${vocab_size} && echo "Finish creating dataset for LM (stage: 3)."
    fi

    # NOTE: support only a single GPU for LM training
    CUDA_VISIBLE_DEVICES=${lm_gpu} ${NEURALSP_ROOT}/neural_sp/bin/lm/train.py \
        --corpus librispeech \
        --n_gpus 1 \
        --train_set ${data}/dataset_lm/train_${lm_data_size}_${train_set}_${unit}${wp_type}${vocab_size}.tsv \
        --dev_set ${data}/dataset_lm/dev_${lm_data_size}_${train_set}_${unit}${wp_type}${vocab_size}.tsv \
        --dict ${dict} \
        --wp_model ${wp_model}.model \
        --model ${model}/lm \
        --unit ${unit} \
        --lm_type ${lm_type} \
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
        --backward ${lm_backward} || exit 1;
    # --resume ${lm_resume} || exit 1;

    echo "Finish LM training (stage: 3)." && exit 1;
fi

if [ ${stage} -le 4 ]; then
    echo ============================================================================
    echo "                       ASR Training stage (stage:4)                        "
    echo ============================================================================

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/asr/train.py \
        --corpus librispeech \
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
        --conv_bottleneck_dim ${conv_bottleneck_dim} \
        --conv_batch_norm ${conv_batch_norm} \
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
        --lm_fusion ${lm_fusion} \
        --lm_init ${lm_init} \
        --lmobj_weight ${lmobj_weight} \
        --share_lm_softmax ${share_lm_softmax} \
        --n_caches ${n_caches} \
        --resume ${resume} || exit 1;

    echo "Finish model training (stage: 4)."
fi
