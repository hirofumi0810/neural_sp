#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                                  PTB                                     "
echo ============================================================================

stage=0
stop_stage=3
gpu=
benchmark=true

### vocabulary
unit=word
vocab_size=10000

#########################
# LM configuration
#########################
conf=conf/rnnlm.yaml

### path to save the model
model=/n/work2/inaguma/results/ptb

### path to the model directory to resume training
resume=

### path to save preproecssed data
data=/n/work2/inaguma/corpus/ptb

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

train_set=train
dev_set=valid
test_set=test

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ] && [ ! -e ${data}/.done_stage_0 ]; then
    echo ============================================================================
    echo "                       Data Preparation (stage:0)                          "
    echo ============================================================================

    mkdir -p ${data}
    wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz -P ${data}
    tar xf ${data}/simple-examples.tgz -C ${data}
    # NOTE: OOV words are already replaced with <unk>

    for x in ${train_set} ${dev_set} ${test_set}; do
        mkdir -p ${data}/${x}
        cat ${data}/simple-examples/data/ptb.${x}.txt | grep -v '^\s*$' | sed -e 's/^[ ]*//g' | awk '{print NR, $0}' > ${data}/${x}/text
        # NOTE: Skip empty line
    done

    touch ${data}/.done_stage_0 && echo "Finish data preparation (stage: 0)."
fi

dict=${data}/dict/${train_set}_${unit}${vocab_size}.txt; mkdir -p ${data}/dict
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ ! -e ${data}/.done_stage_2 ]; then
    echo ============================================================================
    echo "                      Dataset preparation (stage:2)                        "
    echo ============================================================================

    echo "Making a dictionary..."
    echo "<unk> 1" > ${dict}  # <unk> must be 1, 0 will be used for "blank" in CTC
    echo "<eos> 2" >> ${dict}  # <sos> and <eos> share the same index
    echo "<pad> 3" >> ${dict}
    offset=$(cat ${dict} | wc -l)
    text2dict.py ${data}/${train_set}/text --unit ${unit} --vocab_size ${vocab_size} | grep -v "<unk>" | \
        awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict} || exit 1;
    echo "vocab size:" $(cat ${dict} | wc -l)

    echo "Making dataset tsv files for LM ..."
    mkdir -p ${data}/dataset
    for x in ${train_set} ${dev_set} ${test_set}; do
        make_dataset.sh --unit ${unit} ${data}/${x} ${dict} > ${data}/dataset/${x}_${unit}${vocab_size}.tsv || exit 1;
    done

    touch ${data}/.done_stage_2 && echo "Finish creating dataset for LM (stage: 2)."
fi

mkdir -p ${model}
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo ============================================================================
    echo "                        LM Training stage (stage:3)                       "
    echo ============================================================================

    # NOTE: support only a single GPU for LM training
    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/lm/train.py \
        --corpus ptb \
        --config ${conf} \
        --n_gpus ${n_gpus} \
        --cudnn_benchmark ${benchmark} \
        --train_set ${data}/dataset/${train_set}_${unit}${vocab_size}.tsv \
        --dev_set ${data}/dataset/${dev_set}_${unit}${vocab_size}.tsv \
        --dict ${dict} \
        --model ${model}/lm \
        --unit ${unit} \
        --resume ${resume} || exit 1;

    echo "Finish LM training (stage: 3)." && exit 1;
fi
