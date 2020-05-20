#!/bin/bash

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

unit=""
nlsyms=""
vocab=""
wp_model=""
wp_type=""
character_coverage=1.0
speed_perturb=false

. utils/parse_options.sh

if [ $# != 3 ]; then
    echo "Usage: $0 <data_dir> <dict> <text>";
    exit 1;
fi

data=$1
dict=$2
text=$3

if [ -z ${unit} ]; then
    echo "Set --unit";
    exit 1;
fi

if [ -z ${vocab} ]; then
    vocab=0
fi

echo "Making a dictionary..."
echo "<unk> 1" > ${dict}  # <unk> must be 1, 0 will be used for "blank" in CTC
echo "<eos> 2" >> ${dict}  # <sos> and <eos> share the same index
echo "<pad> 3" >> ${dict}
[ ${unit} = char ] && echo "<space> 4" >> ${dict}
[ ${unit} = char_space ] && echo "<space> 4" >> ${dict}
offset=$(cat ${dict} | wc -l)
if [ ${unit} = wp ]; then
    if [ ${speed_perturb} = true ]; then
        grep sp1.0 ${text} | cut -f 2- -d " " > ${data}/dict/input.txt
    else
        cut -f 2- -d " " ${text} > ${data}/dict/input.txt
    fi
    if [ -z ${nlsyms} ]; then
        spm_train --input=${data}/dict/input.txt --vocab_size=${vocab} \
            --model_type=${wp_type} --model_prefix=${wp_model} --input_sentence_size=100000000 --character_coverage=${character_coverage}
    else
        spm_train --user_defined_symbols=$(cat ${nlsyms} | tr "\n" ",") --input=${data}/dict/input.txt --vocab_size=${vocab} \
            --model_type=${wp_type} --model_prefix=${wp_model} --input_sentence_size=100000000 --character_coverage=${character_coverage}
    fi
    spm_encode --model=${wp_model}.model --output_format=piece < ${data}/dict/input.txt | tr ' ' '\n' | \
        sort | uniq -c | sort -n -k1 -r | sed -e 's/^[ ]*//g' | cut -d " " -f 2 | grep -v '^\s*$' | awk -v offset=${offset} '{print $1 " " NR+offset}' >> ${dict}

elif [ ${unit} = phone ]; then
    echo "phone is not implemented yet.";
    exit 1;

else
    # character
    text2dict.py ${text} --unit ${unit} --vocab_size ${vocab} --nlsyms ${nlsyms} | \
        awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict} || exit 1;
fi
echo "vocab size:" $(cat ${dict} | wc -l)
