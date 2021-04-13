#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

nj=32
speeds="0.9 1.0 1.1"

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

if [ ${train_set_original} = ${train_set} ];then
  echo "train_set_original and train_set should be different names"
fi

for speed in ${speeds}; do
  utils/perturb_data_dir_speed.sh ${speed} ${data}/${train_set_original} ${tmpdir}/temp${speed}
done
utils/combine_data.sh --extra-files utt2uniq ${data}/${train_set} ${tmpdir}/temp*
rm -r ${tmpdir}/temp*
steps/make_fbank.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
    ${data}/${train_set} ${data}/log/make_fbank/${train_set} ${data}/fbank
touch ${data}/${train_set}/text.tmp
for speed in ${speeds}; do
    awk -v p="sp${speed}-" '{printf("%s %s%s\n", $1, p, $1);}' ${data}/${train_set_original}/utt2spk > ${data}/${train_set}/utt_map
    utils/apply_map.pl -f 1 ${data}/${train_set}/utt_map <${data}/${train_set_original}/text >>${data}/${train_set}/text.tmp
done
mv ${data}/${train_set}/text.tmp ${data}/${train_set}/text
utils/fix_data_dir.sh ${data}/${train_set}
