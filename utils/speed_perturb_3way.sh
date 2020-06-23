#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

nj=32

. utils/parse_options.sh

if [ $# != 3 ]; then
    echo "Usage: $0 <data> <train_set_original> <train_set>";
    exit 1;
fi

data=$1
train_set_original=$2
train_set=$3
tmpdir=$(mktemp -d ${data}/${train_set_original}/tmp-XXXXX)
trap 'rm -rf ${tmpdir}' EXIT

utils/perturb_data_dir_speed.sh 0.9 ${data}/${train_set_original} ${tmpdir}/temp1
utils/perturb_data_dir_speed.sh 1.0 ${data}/${train_set_original} ${tmpdir}/temp2
utils/perturb_data_dir_speed.sh 1.1 ${data}/${train_set_original} ${tmpdir}/temp3
utils/combine_data.sh --extra-files utt2uniq ${data}/${train_set} ${tmpdir}/temp1 ${tmpdir}/temp2 ${tmpdir}/temp3
rm -r ${tmpdir}/temp1 ${tmpdir}/temp2 ${tmpdir}/temp3
steps/make_fbank.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
    ${data}/${train_set} ${data}/log/make_fbank/${train_set} ${data}/fbank
awk -v p="sp0.9-" '{printf("%s %s%s\n", $1, p, $1);}' ${data}/${train_set_original}/utt2spk > ${data}/${train_set}/utt_map
utils/apply_map.pl -f 1 ${data}/${train_set}/utt_map <${data}/${train_set_original}/text >${data}/${train_set}/text
awk -v p="sp1.0-" '{printf("%s %s%s\n", $1, p, $1);}' ${data}/${train_set_original}/utt2spk > ${data}/${train_set}/utt_map
utils/apply_map.pl -f 1 ${data}/${train_set}/utt_map <${data}/${train_set_original}/text >>${data}/${train_set}/text
awk -v p="sp1.1-" '{printf("%s %s%s\n", $1, p, $1);}' ${data}/${train_set_original}/utt2spk > ${data}/${train_set}/utt_map
utils/apply_map.pl -f 1 ${data}/${train_set}/utt_map <${data}/${train_set_original}/text >>${data}/${train_set}/text
utils/fix_data_dir.sh ${data}/${train_set}
