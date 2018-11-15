#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

. utils/parse_options.sh

if [ $# != 1 ]; then
    echo "Usage: $0 <decode-dir>";
    exit 1;
fi

decode_dir=$1
phonemap="conf/phones.60-48-39.map"

# Map reference to 39 phone classes:
cat ${decode_dir}/ref.trn | local/timit_norm_trans.pl -i - -m ${phonemap} -from 60 -to 39 > ${decode_dir}/ref.trn.filt
cat ${decode_dir}/hyp.trn | local/timit_norm_trans.pl -i - -m ${phonemap} -from 60 -to 39 > ${decode_dir}/hyp.trn.filt

sed -e "s/<eos>/ /g" ${decode_dir}/ref.trn.filt > ${decode_dir}/ref.trn.filt.clean
sed -e "s/<eos>/ /g" ${decode_dir}/hyp.trn.filt > ${decode_dir}/hyp.trn.filt.clean

sclite -r ${decode_dir}/ref.trn trn -h ${decode_dir}/hyp.trn trn -i rm -o all stdout > ${decode_dir}/result.txt
grep -e Avg -e SPKR -m 2 ${decode_dir}/result.txt > ${decode_dir}/RESULTS
cat ${decode_dir}/RESULTS
