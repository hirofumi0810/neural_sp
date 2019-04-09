#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

if [ $# -ne 1 ]; then
  exit 1;
fi

data=$1

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail


cp ${data}/text ${data}/text.tmp.0
cut -f 2- -d " " ${data}/text.tmp.0 | \
    sed -e 's/*//g' | \
    sed -e "s/)/\'/g" | \
    sed -e 's/.period/period/g' | \
    sed -e 's/,comma/comma/g' | \
    sed -e 's/:colon/colon/g' | \
    sed -e 's/://g' | \
    sed -e 's/;semi-colon/semi-colon/g' | \
    sed -e 's/;//g' | \
    sed -e 's/\/slash/slash/g' | \
    sed -e 's/&ampersand/ampersand/g' | \
    sed -e 's/?question-mark/question-mark/g' | \
    sed -e 's/!exclamation-point/exclamation-point/g' | \
    sed -e 's/!//g' | \
    sed -e 's/-dash/dash/g' | \
    sed -e 's/-hyphen/hyphen/g' | \
    sed -e 's/(paren/paren/g' | \
    sed -e 's/)paren/paren/g' | \
    sed -e 's/)un-parentheses/un-parentheses/g' | \
    sed -e 's/)close_paren/close-paren/g' | \
    sed -e 's/)close-paren/close-paren/g' | \
    sed -e 's/)end-the-paren/end-the-paren/g' | \
    sed -e 's/(left-paren/left-paren/g' | \
    sed -e 's/)right-paren/right-paren/g' | \
    sed -e 's/(begin-parens/begin-parens/g' | \
    sed -e 's/)end-parens/end-parens/g' | \
    sed -e 's/(brace/brace/g' | \
    sed -e 's/)close-brace/close-brace/g' | \
    sed -e 's/{left-brace/left-brace/g' | \
    sed -e 's/}right-brace/right-brace/g' | \
    sed -e "s/\'single-quote/single-quote/g" | \
    sed -e 's/\"quote/quote/g' | \
    sed -e 's/\"in-quotes/in-quotes/g' | \
    sed -e 's/\"double-quote/double-quote/g' | \
    sed -e 's/\"unquote/quote/g' | \
    sed -e 's/\"close-quote/close-quote/g' | \
    sed -e 's/\"end-quote/end-quote/g' | \
    sed -e 's/\"end-of-quote/end-of-quote/g' |

sed -e 's/\<nperiod/nperiod/g' > ${data}/text.tmp.1

paste -d " " <(cut -f 1 -d " " ${data}/text.tmp.0) \
    <(cat ${data}/text.tmp.1 | awk '{print tolower($0)}') > ${data}/text
rm ${data}/text.tmp*
