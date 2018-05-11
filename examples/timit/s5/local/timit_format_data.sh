#!/bin/bash

# Copyright 2013  (Author: Daniel Povey)
# Apache 2.0

# This script takes data prepared in a corpus-dependent way
# in data/local/, and converts it into the "canonical" form,
# in various subdirectories of data/, e.g. data/lang, data/train, etc.

. ./path.sh || exit 1;

echo "Preparing train, dev and test data"
srcdir=$DATA_SAVEPATH/local/data

for x in train dev test; do
  mkdir -p $DATA_SAVEPATH/$x
  cp $srcdir/${x}_wav.scp $DATA_SAVEPATH/$x/wav.scp || exit 1;
  cp $srcdir/$x.text $DATA_SAVEPATH/$x/text || exit 1;
  cp $srcdir/$x.spk2utt $DATA_SAVEPATH/$x/spk2utt || exit 1;
  cp $srcdir/$x.utt2spk $DATA_SAVEPATH/$x/utt2spk || exit 1;
  utils/filter_scp.pl $DATA_SAVEPATH/$x/spk2utt $srcdir/$x.spk2gender > $DATA_SAVEPATH/$x/spk2gender || exit 1;
  cp $srcdir/${x}.stm $DATA_SAVEPATH/$x/stm
  cp $srcdir/${x}.glm $DATA_SAVEPATH/$x/glm
  utils/validate_data_dir.sh --no-feats $DATA_SAVEPATH/$x || exit 1

  cp $srcdir/${x}.spk2gender $DATA_SAVEPATH/$x/spk2gender  # added
done

echo "Succeeded in formatting data."
