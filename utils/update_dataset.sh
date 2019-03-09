#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

unit=""
remove_space=false
unk="<unk>"
space="<space>"
nlsyms=""
wp_model=""

. utils/parse_options.sh

if [ $# != 3 ]; then
    echo "Usage: $0 <data_dir> <dict> <tsv>";
    exit 1;
fi

text=$1
dict=$2
tsv=$3
tmpdir=$(mktemp -d $(dirname ${text})/tmp-XXXXX)
trap 'rm -rf ${tmpdir}' EXIT

cp ${tsv} ${tmpdir}/tmp.tsv

# For additional unpaired text
make_tsv.py --text ${text} \
    --dict ${dict} \
    --unit ${unit} \
    --remove_space ${remove_space} \
    --unk ${unk} \
    --space ${space} \
    --nlsyms ${nlsyms} \
    --wp_model ${wp_model}  | sed -e '1d' >> ${tmpdir}/tmp.tsv

cat ${tmpdir}/tmp.tsv

rm -fr ${tmpdir}
