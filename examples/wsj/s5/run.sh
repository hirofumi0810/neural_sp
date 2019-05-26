#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                                   WSJ                                     "
echo ============================================================================

stage=0
gpu=
skip_lm=true

### vocabulary
unit=wp      # word/wp/char/word_char
vocab_size=1000
wp_type=bpe  # bpe/unigram (for wordpiece)

#########################
# ASR configuration
#########################
### topology
n_splices=1
n_stacks=1
n_skips=1
sequence_summary_network=false
conv_in_channel=1
conv_channels="32_32"
conv_kernel_sizes="(3,3)_(3,3)"
conv_strides="(1,1)_(1,1)"
conv_poolings="(2,2)_(2,2)"
conv_batch_norm=false
conv_residual=false
conv_bottleneck_dim=0
subsample="1_2_2_1_1"
enc_type=conv_blstm
enc_n_units=512
enc_n_projs=0
enc_n_layers=5
enc_residual=false
enc_nin=false
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
dec_bottleneck_dim=1024
emb_dim=512
tie_embedding=false
ctc_fc_list=""
### optimization
batch_size=20
optimizer=adam
learning_rate=1e-3
n_epochs=25
convert_to_sgd_epoch=20
print_step=100
decay_type=epoch
decay_start_epoch=10
decay_rate=0.85
decay_patient_n_epochs=0
sort_stop_epoch=100
not_improved_patient_n_epochs=5
eval_start_epoch=1
warmup_start_learning_rate=1e-4
warmup_n_steps=4000
### initialization
param_init=0.1
pretrained_model=
teacher=
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
focal_loss=0.0
### MTL
ctc_weight=0.2
bwd_weight=0.0
mtl_per_batch=true
task_specific_layer=false
### LM integration
lm_fusion_type=cold
lm_fusion=
lm_init=
lmobj_weight=0.0
share_lm_softmax=false

# GatedConv
# conv_channels="100_100_100_125_125_150_175_200_225_250_250_250_300_300_375"
# conv_kernel_sizes="(13,1)_(3,1)_(4,1)_(5,1)_(6,1)_(7,1)_(8,1)_(9,1)_(10,1)_(11,1)_(12,1)_(13,1)_(14,1)_(15,1)_(21,1)"

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
lm_batch_size=128
lm_bptt=200
lm_optimizer=adam
lm_learning_rate=1e-3
lm_n_epochs=40
lm_convert_to_sgd_epoch=40
lm_print_step=200
lm_decay_start_epoch=10
lm_decay_rate=0.9
lm_decay_patient_n_epochs=0
lm_decay_type=epoch
lm_not_improved_patient_n_epochs=10
lm_eval_start_epoch=1
# initialization
lm_param_init=0.05
lm_param_init_dist=uniform
lm_pretrained_model=
# regularization
lm_clip_grad_norm=1.0
lm_dropout_hidden=0.2
lm_dropout_out=0.0
lm_dropout_emb=0.2
lm_weight_decay=1e-6
lm_backward=

### path to save the model
model=/n/sd8/inaguma/result/wsj

### path to the model directory to resume training
resume=
lm_resume=

### path to save preproecssed data
export data=/n/sd8/inaguma/corpus/wsj

### path to original data
wsj0=/n/rd21/corpora_1/WSJ/wsj0
wsj1=/n/rd21/corpora_1/WSJ/wsj1

# Sometimes, we have seen WSJ distributions that do not have subdirectories
# like '11-13.1', but instead have 'doc', 'si_et_05', etc. directly under the
# wsj0 or wsj1 directories. In such cases, try the following:
CSTR_WSJTATATOP=/n/rd21/corpora_1/WSJ
# $CSTR_WSJTATATOP must contain a 'wsj0' and a 'wsj1' subdirectory for this to work.

directory_type=cstr # or original

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

train_set=train_si284
dev_set=test_dev93
test_set="test_eval92"

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

    case ${directory_type} in
        original) local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.? || exit 1; ;;
        cstr) local/cstr_wsj_data_prep.sh $CSTR_WSJTATATOP || exit 1; ;;
    esac

    # "nosp" refers to the dictionary before silence probabilities and pronunciation
    # probabilities are added.
    local/wsj_prepare_dict.sh --dict-suffix "_nosp" || exit 1;
    local/wsj_format_data.sh --lang-suffix "_nosp" || exit 1;
    case ${directory_type} in
        original) local/wsj_extend_dict.sh --dict-suffix "_nosp" ${wsj1}/13-32.1 || exit 1; ;;
        cstr) local/cstr_wsj_extend_dict.sh --dict-suffix "_nosp" $CSTR_WSJTATATOP/wsj1/doc/ || exit 1; ;;
    esac

    # lowercasing
    for x in ${train_set} ${dev_set} ${test_set}; do
        cp ${data}/${x}/text ${data}/${x}/text.org
        paste -d " " <(cut -f 1 -d " " ${data}/${x}/text.org) \
            <(cut -f 2- -d " " ${data}/${x}/text.org | awk '{print tolower($0)}') > ${data}/${x}/text
    done

    # nomalization
    for x in ${train_set} ${dev_set}; do
        local/normalize_trans.sh ${data}/${x}
    done

    touch ${data}/.done_stage_0 && echo "Finish data preparation (stage: 0)."
fi

if [ ${stage} -le 1 ] && [ ! -e ${data}/.done_stage_1 ]; then
    echo ============================================================================
    echo "                    Feature extranction (stage:1)                          "
    echo ============================================================================

    for x in train_si284 test_dev93 test_eval92; do
        steps/make_fbank.sh --nj 32 --cmd "$train_cmd" --write_utt2num_frames true \
            ${data}/${x} ${data}/log/make_fbank/${x} ${data}/fbank || exit 1;
    done

    # Compute global CMVN
    compute-cmvn-stats scp:${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark || exit 1;

    # Apply global CMVN & dump features
    dump_feat.sh --cmd "$train_cmd" --nj 200 \
        ${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${train_set} ${data}/dump/${train_set} || exit 1;
    for x in ${dev_set} ${test_set}; do
        dump_dir=${data}/dump/${x}
        dump_feat.sh --cmd "$train_cmd" --nj 32 \
            ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x} ${dump_dir} || exit 1;
    done

    touch ${data}/.done_stage_1 && echo "Finish feature extranction (stage: 1)."
fi

dict=${data}/dict/${train_set}_${unit}${wp_type}${vocab_size}.txt; mkdir -p ${data}/dict
nlsyms=${data}/dict/nlsyms.txt
wp_model=${data}/dict/${train_set}_${wp_type}${vocab_size}
if [ ${stage} -le 2 ] && [ ! -e ${data}/.done_stage_2_${unit}${wp_type}${vocab_size} ]; then
    echo ============================================================================
    echo "                      Dataset preparation (stage:2)                        "
    echo ============================================================================

    echo "make a non-linguistic symbol list"
    cut -f 2- -d " " ${data}/${train_set}/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}

    echo "Making a dictionary..."
    echo "<unk> 1" > ${dict}  # <unk> must be 1, 0 will be used for "blank" in CTC
    echo "<eos> 2" >> ${dict}  # <sos> and <eos> share the same index
    echo "<pad> 3" >> ${dict}
    [ ${unit} = char ] && echo "<space> 4" >> ${dict}
    offset=$(cat ${dict} | wc -l)
    if [ ${unit} = wp ]; then
        cut -f 2- -d " " ${data}/${train_set}/text > ${data}/dict/input.txt
        spm_train --user_defined_symbols=$(cat ${nlsyms} | tr "\n" ",") --input=${data}/dict/input.txt --vocab_size=${vocab_size} \
            --model_type=${wp_type} --model_prefix=${wp_model} --input_sentence_size=100000000 --character_coverage=1.0
        spm_encode --model=${wp_model}.model --output_format=piece < ${data}/dict/input.txt | tr ' ' '\n' | \
            sort | uniq -c | sort -n -k1 -r | sed -e 's/^[ ]*//g' | awk -v offset=${offset} '{print $2 " " NR+offset}' >> ${dict}
        # NOTE: sort by frequency
    else
        text2dict.py ${data}/${train_set}/text --unit ${unit} --vocab_size ${vocab_size} --nlsyms ${nlsyms} | \
            awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict} || exit 1;
    fi
    echo "vocab size:" $(cat ${dict} | wc -l)

    # Compute OOV rate
    if [ ${unit} = word ]; then
        mkdir -p ${data}/dict/word_count ${data}/dict/oov_rate
        echo "OOV rate:" > ${data}/dict/oov_rate/word${vocab_size}.txt
        for x in ${train_set} ${dev_set} ${test_set}; do
            cut -f 2- -d " " ${data}/${x}/text | tr " " "\n" | sort | uniq -c | sort -n -k1 -r \
                > ${data}/dict/word_count/${x}.txt || exit 1;
            compute_oov_rate.py ${data}/dict/word_count/${x}.txt ${dict} ${x} \
                >> ${data}/dict/oov_rate/word${vocab_size}.txt || exit 1;
        done
        cat ${data}/dict/oov_rate/word${vocab_size}.txt
    fi

    echo "Making dataset tsv files for ASR ..."
    mkdir -p ${data}/dataset
    for x in ${train_set} ${dev_set} ${test_set}; do
        dump_dir=${data}/dump/${x}
        make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit} --nlsyms ${nlsyms} --wp_model ${wp_model} \
            ${data}/${x} ${dict} > ${data}/dataset/${x}_${unit}${wp_type}${vocab_size}.tsv || exit 1;
    done

    touch ${data}/.done_stage_2_${unit}${wp_type}${vocab_size} && echo "Finish creating dataset for ASR (stage: 2)."
fi

mkdir -p ${model}
if ! ${skip_lm} && [ ${stage} -le 3 ]; then
    echo ============================================================================
    echo "                        LM Training stage (stage:3)                       "
    echo ============================================================================

    # Extend dictionary for the external text data
    if [ ! -e ${data}/.done_stage_3_${unit}${wp_type}${vocab_size} ]; then
        echo "Making dataset tsv files for LM ..."
        mkdir -p ${data}/dataset_lm
        cat ${data}/local/dict_nosp_larger/cleaned | tr "[:upper:]" "[:lower:]" > ${data}/dataset_lm/cleaned
        awk '{print "unpaired-text-"NR, $0}' ${data}/dataset_lm/cleaned > ${data}/dataset_lm/text

        update_dataset.sh --unit ${unit} --nlsyms ${nlsyms} --wp_model ${wp_model} \
            ${data}/dataset_lm/text ${dict} ${data}/dataset/${train_set}_${unit}${wp_type}${vocab_size}.tsv \
            > ${data}/dataset_lm/${train_set}_${unit}${wp_type}${vocab_size}.tsv || exit 1;
        cp ${data}/dataset/${dev_set}_${unit}${wp_type}${vocab_size}.tsv \
            ${data}/dataset_lm/${dev_set}_${unit}${wp_type}${vocab_size}.tsv || exit 1;

        touch ${data}/.done_stage_3_${unit}${wp_type}${vocab_size} && echo "Finish creating dataset for LM (stage: 3)."
    fi

    # NOTE: support only a single GPU for LM training
    CUDA_VISIBLE_DEVICES=${lm_gpu} ${NEURALSP_ROOT}/neural_sp/bin/lm/train.py \
        --corpus wsj \
        --n_gpus 1 \
        --train_set ${data}/dataset_lm/${train_set}_${unit}${wp_type}${vocab_size}.tsv \
        --dev_set ${data}/dataset_lm/${dev_set}_${unit}${wp_type}${vocab_size}.tsv \
        --nlsyms ${nlsyms} \
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
        --decay_type ${lm_decay_type} \
        --not_improved_patient_n_epochs ${lm_not_improved_patient_n_epochs} \
        --eval_start_epoch ${lm_eval_start_epoch} \
        --param_init ${lm_param_init} \
        --pretrained_model ${lm_pretrained_model} \
        --clip_grad_norm ${lm_clip_grad_norm} \
        --dropout_hidden ${lm_dropout_hidden} \
        --dropout_out ${lm_dropout_out} \
        --dropout_emb ${lm_dropout_emb} \
        --weight_decay ${lm_weight_decay} \
        --backward ${lm_backward} \
        --resume ${lm_resume} || exit 1;

    echo "Finish LM training (stage: 3)." && exit 1;
fi

if [ ${stage} -le 4 ]; then
    echo ============================================================================
    echo "                       ASR Training stage (stage:4)                        "
    echo ============================================================================

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/asr/train.py \
        --corpus wsj \
        --n_gpus ${n_gpus} \
        --train_set ${data}/dataset/${train_set}_${unit}${wp_type}${vocab_size}.tsv \
        --dev_set ${data}/dataset/${dev_set}_${unit}${wp_type}${vocab_size}.tsv \
        --eval_sets ${data}/dataset/${test_set}_${unit}${wp_type}${vocab_size}.tsv \
        --nlsyms ${nlsyms} \
        --dict ${dict} \
        --wp_model ${wp_model}.model \
        --model ${model}/asr \
        --unit ${unit} \
        --n_splices ${n_splices} \
        --n_stacks ${n_stacks} \
        --n_skips ${n_skips} \
        --sequence_summary_network ${sequence_summary_network} \
        --conv_in_channel ${conv_in_channel} \
        --conv_channels ${conv_channels} \
        --conv_kernel_sizes ${conv_kernel_sizes} \
        --conv_strides ${conv_strides} \
        --conv_poolings ${conv_poolings} \
        --conv_batch_norm ${conv_batch_norm} \
        --conv_residual ${conv_residual} \
        --conv_bottleneck_dim ${conv_bottleneck_dim} \
        --enc_type ${enc_type} \
        --enc_n_units ${enc_n_units} \
        --enc_n_projs ${enc_n_projs} \
        --enc_n_layers ${enc_n_layers} \
        --enc_residual ${enc_residual} \
        --enc_nin ${enc_nin} \
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
        --dec_bottleneck_dim ${dec_bottleneck_dim} \
        --emb_dim ${emb_dim} \
        --tie_embedding ${tie_embedding} \
        --ctc_fc_list ${ctc_fc_list} \
        --batch_size ${batch_size} \
        --optimizer ${optimizer} \
        --learning_rate ${learning_rate} \
        --n_epochs ${n_epochs} \
        --convert_to_sgd_epoch ${convert_to_sgd_epoch} \
        --print_step ${print_step} \
        --decay_type ${decay_type} \
        --decay_start_epoch ${decay_start_epoch} \
        --decay_rate ${decay_rate} \
        --decay_patient_n_epochs ${decay_patient_n_epochs} \
        --not_improved_patient_n_epochs ${not_improved_patient_n_epochs} \
        --sort_stop_epoch ${sort_stop_epoch} \
        --eval_start_epoch ${eval_start_epoch} \
        --warmup_start_learning_rate ${warmup_start_learning_rate} \
        --warmup_n_steps ${warmup_n_steps} \
        --param_init ${param_init} \
        --pretrained_model ${pretrained_model} \
        --teacher ${teacher} \
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
        --mtl_per_batch ${mtl_per_batch} \
        --task_specific_layer ${task_specific_layer} \
        --lm_fusion_type ${lm_fusion_type} \
        --lm_fusion ${lm_fusion} \
        --lm_init ${lm_init} \
        --lmobj_weight ${lmobj_weight} \
        --share_lm_softmax ${share_lm_softmax} \
        --resume ${resume} || exit 1;

    echo "Finish model training (stage: 4)."
fi
