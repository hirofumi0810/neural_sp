#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

feat="" # feat.scp
unit=""
remove_space=false
unk="<unk>"
space="<space>"
nlsyms=""
wp_model=""

. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "Usage: $0 <data_dir> <dict>";
    exit 1;
fi

data=$1
dict=$2

# csv: utt_id, feat_path, xlen, text, tokenid, ylen
make_csv.py --feat ${feat} \
            --utt2num_frames ${data}/utt2num_frames \
            --text ${data}/text \
            --dict ${dict} \
            --unit ${unit} \
            --remove_space ${remove_space} \
            --unk ${unk} \
            --space ${space} \
            --nlsyms ${nlsyms} \
            --wp_model ${wp_model}
