#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "Usage: $0 <data> <lexicon>";
    exit 1;
fi

data=$1
lexicon=$2

map2phone.py --text ${data}/text \
    --lexicon ${lexicon} > ${data}/text.phone
