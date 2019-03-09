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
unit=wp      # or word or char or word_char or phone
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
conv_batch_norm=
subsample="1_2_2_2_1"
# VGG
# conv_channels="64_64_128_128"
# conv_kernel_sizes="(3,3)_(3,3)_(3,3)_(3,3)"
# conv_strides="(1,1)_(1,1)_(1,1)_(1,1)"
# conv_poolings="(1,1)_(2,2)_(1,1)_(2,2)"
# subsample="1_1_1_1_1"
enc_type=blstm
enc_nunits=320
enc_nprojs=0
enc_nlayers=5
enc_residual=
enc_add_ffl=
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
nepochs=15
convert_to_sgd_epoch=10
print_step=200
decay_start_epoch=5
decay_rate=0.8
decay_patient_epoch=0
decay_type=epoch
not_improved_patient_epoch=5
eval_start_epoch=1
warmup_start_learning_rate=1e-4
warmup_nsteps=0
warmup_nepochs=0
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
gaussian_noise_std=0.0
gaussian_noise_timing=constant
### MTL
ctc_weight=0.0
bwd_weight=0.0
mtl_per_batch=true
task_specific_layer=
### LM integration
cold_fusion=
rnnlm_cold_fusion=
rnnlm_init=
lmobj_weight=0.0
share_lm_softmax=

### path to save the model
model=/n/sd8/inaguma/result/swbd

### path to the model directory to restart training
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
ngpus=$(echo ${gpu} | tr "," "\n" | wc -l)

train_set=train_sp_${data_size}
dev_set=dev_sp_${data_size}
test_set="eval2000_sp"

if [ ${unit} = char ] || [ ${unit} = phone ]; then
    vocab_size=
fi
if [ ${unit} != wp ]; then
    wp_type=
fi

if [ ${stage} -le 0 ] && [ ! -e ${data}/.done_stage_0_${data_size} ]; then
    echo ============================================================================
    echo "                       Data Preparation (stage:0)                          "
    echo ============================================================================

    if [ ! -e ${data}/.done_stage_0_${data_size} ]; then
        echo "run ./run.sh first" && exit 1
    fi
fi

if [ ${stage} -le 1 ] && [ ! -e ${data}/.done_stage_1_${data_size}_sp ]; then
    echo ============================================================================
    echo "                    Feature extranction (stage:1)                          "
    echo ============================================================================

    if [ ! -e ${data}/.done_stage_1_${data_size} ]; then
        echo "run ./run.sh first" && exit 1
    fi

    # speed-perturbed
    utils/perturb_data_dir_speed.sh 0.9 ${data}/train_swbd ${data}/temp1
    utils/perturb_data_dir_speed.sh 1.0 ${data}/train_swbd ${data}/temp2
    utils/perturb_data_dir_speed.sh 1.1 ${data}/train_swbd ${data}/temp3
    utils/combine_data.sh --extra-files utt2uniq ${data}/${train_set} ${data}/temp1 ${data}/temp2 ${data}/temp3
    rm -r ${data}/temp1 ${data}/temp2 ${data}/temp3
    steps/make_fbank.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
        ${data}/${train_set} ${data}/log/make_fbank/${train_set} ${data}/fbank
    cat ${data}/train_swbd/utt2spk | awk -v p="sp0.9-" '{printf("%s %s%s\n", $1, p, $1);}' > ${data}/${train_set}/utt_map
    utils/apply_map.pl -f 1 ${data}/${train_set}/utt_map <${data}/train_swbd/text >${data}/${train_set}/text
    cat ${data}/train_swbd/utt2spk | awk -v p="sp1.0-" '{printf("%s %s%s\n", $1, p, $1);}' > ${data}/${train_set}/utt_map
    utils/apply_map.pl -f 1 ${data}/${train_set}/utt_map <${data}/train_swbd/text >>${data}/${train_set}/text
    cat ${data}/train_swbd/utt2spk | awk -v p="sp1.1-" '{printf("%s %s%s\n", $1, p, $1);}' > ${data}/${train_set}/utt_map
    utils/apply_map.pl -f 1 ${data}/${train_set}/utt_map <${data}/train_swbd/text >>${data}/${train_set}/text

    utils/fix_data_dir.sh ${data}/${train_set}

    cp -rf ${data}/dev_swbd ${data}/${dev_set}
    cp -rf ${data}/eval2000 ${data}/${test_set}

    # Compute global CMVN
    compute-cmvn-stats scp:${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark || exit 1;

    # Apply global CMVN & dump features
    dump_feat.sh --cmd "$train_cmd" --nj 80 \
        ${data}/${train_set}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${train_set} ${data}/dump/${train_set} || exit 1;
    dump_feat.sh --cmd "$train_cmd" --nj 32 \
        ${data}/${dev_set}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${dev_set} ${data}/dump/${dev_set} || exit 1;
    for x in ${test_set}; do
        dump_dir=${data}/dump/${x}_${data_size}
        dump_feat.sh --cmd "$train_cmd" --nj 32 \
            ${data}/${x}/feats.scp ${data}/${train_set}/cmvn.ark ${data}/log/dump_feat/${x}_${data_size} ${dump_dir} || exit 1;
    done

    touch ${data}/.done_stage_1_${data_size}_sp && echo "Finish feature extranction (stage: 1)."
fi

dict=${data}/dict/${train_set}_${unit}${wp_type}${vocab_size}.txt; mkdir -p ${data}/dict
nlsyms=${data}/dict/non_linguistic_symbols_${data_size}.txt
wp_model=${data}/dict/${train_set}_${wp_type}${vocab_size}
if [ ${stage} -le 2 ] && [ ! -e ${data}/.done_stage_2_${data_size}_${unit}${wp_type}${vocab_size}_sp ]; then
    echo ============================================================================
    echo "                      Dataset preparation (stage:2)                        "
    echo ============================================================================

    cat ${data}/${train_set}/text | grep sp1.0 > ${data}/${train_set}/text.org

    echo "make a non-linguistic symbol list"
    cut -f 2- -d " " ${data}/${train_set}/text.org | tr " " "\n" | sort | uniq | grep "\[" > ${nlsyms}
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
        cut -f 2- -d " " ${data}/${train_set}/text.org > ${data}/dict/input.txt
        spm_train --user_defined_symbols=$(cat ${nlsyms} | tr "\n" ",") --input=${data}/dict/input.txt --vocab_size=${vocab_size} \
            --model_type=${wp_type} --model_prefix=${wp_model} --input_sentence_size=100000000 --character_coverage=1.0
        spm_encode --model=${wp_model}.model --output_format=piece < ${data}/dict/input.txt | tr ' ' '\n' | \
            sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}
    elif [ ${unit} = phone ]; then
        map_lexicon.sh ${data}/${train_set} ${data}/local/dict_nosp/lexicon.txt
        map_lexicon.sh ${data}/${dev_set} ${data}/local/dict_nosp/lexicon.txt
        text2dict.py ${data}/${train_set}/text.phone --unit ${unit} | \
            sort | uniq | grep -v -e '^\s*$' | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict} || exit 1;
    else
        text2dict.py ${data}/${train_set}/text.org --unit ${unit} --vocab_size ${vocab_size} --nlsyms ${nlsyms} \
            --wp_type ${wp_type} --wp_model ${wp_model} | \
            sort | uniq | grep -v -e '^\s*$' | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict} || exit 1;
    fi
    echo "vocab size:" $(cat ${dict} | wc -l)

    # Compute OOV rate
    if [ ${unit} = word ]; then
        mkdir -p ${data}/dict/word_count ${data}/dict/oov_rate
        echo "OOV rate:" > ${data}/dict/oov_rate/word${vocab_size}_${data_size}.txt
        for x in ${train_set} ${dev_set}; do
            cut -f 2- -d " " ${data}/${x}/text.org | tr " " "\n" | sort | uniq -c | sort -n -k1 -r \
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
    mkdir -p ${data}/dataset
    for x in ${train_set} ${dev_set}; do
        dump_dir=${data}/dump/${x}
        if [ ${unit} = phone ]; then
            make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit} --text ${data}/${x}/text.phone \
                ${data}/${x} ${dict} > ${data}/dataset/${x}_${unit}${wp_type}${vocab_size}.tsv || exit 1;
        else
            make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit} --nlsyms ${nlsyms} --wp_model ${wp_model} \
                ${data}/${x} ${dict} > ${data}/dataset/${x}_${unit}${wp_type}${vocab_size}.tsv || exit 1;
        fi
    done
    if [ ${unit} != phone ]; then
        for x in ${test_set}; do
            dump_dir=${data}/dump/${x}_${data_size}
            make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit} --nlsyms ${nlsyms} --wp_model ${wp_model} \
                ${data}/${x} ${dict} > ${data}/dataset/${x}_${data_size}_${unit}${wp_type}${vocab_size}.tsv || exit 1;
        done
    fi

    touch ${data}/.done_stage_2_${data_size}_${unit}${wp_type}${vocab_size}_sp && echo "Finish creating dataset for ASR (stage: 2)."
fi

mkdir -p ${model}
if [ ${stage} -le 4 ]; then
    echo ============================================================================
    echo "                       ASR Training stage (stage:4)                        "
    echo ============================================================================

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/asr/train.py \
        --ngpus ${ngpus} \
        --train_set ${data}/dataset/${train_set}_${unit}${wp_type}${vocab_size}.tsv \
        --dev_set ${data}/dataset/${dev_set}_${unit}${wp_type}${vocab_size}.tsv \
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
        --enc_add_ffl ${enc_add_ffl} \
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
        --dec_add_ffl ${dec_add_ffl} \
        --dec_layerwise_attention ${dec_layerwise_attention} \
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
        --warmup_nsteps ${warmup_nsteps} \
        --warmup_nepochs ${warmup_nepochs} \
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
        --gaussian_noise_std ${gaussian_noise_std} \
        --gaussian_noise_timing ${gaussian_noise_timing} \
        --ctc_weight ${ctc_weight} \
        --bwd_weight ${bwd_weight} \
        --mtl_per_batch ${mtl_per_batch} \
        --task_specific_layer ${task_specific_layer} \
        --cold_fusion ${cold_fusion} \
        --rnnlm_cold_fusion =${rnnlm_cold_fusion} \
        --rnnlm_init ${rnnlm_init} \
        --lmobj_weight ${lmobj_weight} \
        --share_lm_softmax ${share_lm_softmax} \
        --resume ${resume} || exit 1;

    echo "Finish model training (stage: 4)."
fi
