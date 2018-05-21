#!/bin/bash

# Copyright 2013  (Author: Daniel Povey)
# Apache 2.0

# This script takes data prepared in a corpus-dependent way
# in data/local/, and converts it into the "canonical" form,
# in various subdirectories of data/, e.g. data/lang, data/train, etc.

. ./path.sh || exit 1;

echo "Preparing train, dev and test data"
srcdir=$DATA/local/data

for x in train dev test; do
  mkdir -p $DATA/$x
  cp $srcdir/${x}_wav.scp $DATA/$x/wav.scp || exit 1;
  cp $srcdir/$x.text $DATA/$x/text || exit 1;
  cp $srcdir/$x.spk2utt $DATA/$x/spk2utt || exit 1;
  cp $srcdir/$x.utt2spk $DATA/$x/utt2spk || exit 1;
  utils/filter_scp.pl $DATA/$x/spk2utt $srcdir/$x.spk2gender > $DATA/$x/spk2gender || exit 1;
  cp $srcdir/${x}.stm $DATA/$x/stm
  cp $srcdir/${x}.glm $DATA/$x/glm
  utils/validate_data_dir.sh --no-feats $DATA/$x || exit 1

  cp $srcdir/${x}.spk2gender $DATA/$x/spk2gender  # added
done

echo "Succeeded in formatting data."
