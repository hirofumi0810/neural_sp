#!/bin/bash

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh
. ./conf/lang.conf

# langs="101 103 104 105 107 201 204 205 207 404"  # 10 langs
langs="101 102 103 104 105 106 107 201 202 203 204 205 206 207 404"  # 15 langs
recog="102 106 202 203 206"
FLP=true

. ./utils/parse_options.sh

set -e
set -o pipefail

all_langs=""
for l in `cat <(echo ${langs}) <(echo ${recog}) | tr " " "\n" | sort -u`; do
  all_langs="${l} ${all_langs}"
done
all_langs=${all_langs%% }

# Save top-level directory
cwd=$(utils/make_absolute.sh `pwd`)
echo "Stage 0: Setup Language Specific Directories"

echo " --------------------------------------------"
echo "Languagues: ${all_langs}"

# Basic directory prep
for l in ${all_langs}; do
  [ -d ${data}/${l} ] || mkdir -p ${data}/${l}
  cd ${data}/${l}

  ln -sf ${cwd}/local .
  for f in ${cwd}/{utils,steps,conf}; do
    link=`make_absolute.sh $f`
    ln -sf $link .
  done

  cp ${cwd}/cmd.sh .
  cp ${cwd}/path.sh .
  sed -i 's/\.\.\/\.\.\/\.\./\.\.\/\.\.\/\.\.\/\.\.\/\.\./g' path.sh

  cd ${cwd}
done

# Prepare language specific data
for l in ${all_langs}; do
  cd ${data}/${l}
  ./local/prepare_data.sh --FLP ${FLP} ${l}
  cd ${cwd}
done

# Combine all language specific training directories and generate a single
# lang directory by combining all language specific dictionaries
train_dirs=""
dev_dirs=""
eval_dirs=""
num_langs=0
for l in ${langs}; do
  num_langs=`expr $num_langs + 1`
  train_dirs="${data}/train_${l} ${train_dirs}"
  dev_dirs="${data}/dev_${l} ${dev_dirs}"
done

for l in ${recog}; do
  eval_dirs="${data}/eval_${l} ${eval_dirs}"
done

train_set="train"
dev_set="dev"
for l in ${langs}; do
  train_set="${train_set}_${l}"
  dev_set="${dev_set}_${l}"
done

./utils/combine_data.sh ${data}/${train_set} ${train_dirs}
./utils/combine_data.sh ${data}/${dev_set} ${dev_dirs}
