#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

if [ $# -ne 3 ]; then
  echo "Usage: local/score_sclite.sh <data-dir> <decode-dir> <set>"
  exit 1;
fi

data=$1
dir=$2
set=$3  # eval2000 or rt03

hubscr=$KALDI_ROOT/tools/sctk/bin/hubscr.pl
[ ! -f $hubscr ] && echo "Cannot find scoring program at $hubscr" && exit 1;
hubdir=`dirname $hubscr`

for f in $data/$set/stm $data/$set/glm; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

# Remove some stuff we don't want to score, from the ctm.
# the big expression in parentheses contains all the things that get mapped
# by the glm file, into hesitations.
# The -$ expression removes partial words.
# the aim here is to remove all the things that appear in the reference as optionally
# deletable (inside parentheses), as if we delete these there is no loss, while
# if we get them correct there is no gain.
cat $dir/hyp.trn | sed -e 's/\[noise\]//g' | sed -e 's/\[laughter\]//g' | sed -e 's/\[vocalized-noise\]//g' | \
  sed -e 's/\s\+/ /g' | sed 's/^[ \t]*//' | sed -e 's/[ \t]*$//' \
  > $dir/hyp.trn.clean
# sed -e 's/<unk>//g' | \
# grep -i -v -E ' (uh|um|eh|mm|hm|ah|huh|ha|er|oof|hee|ach|eee|ew)$' | \
# grep -v -- '-$' | \

python local/map_acronyms_transcripts.py -i $dir/hyp.trn.clean -o $dir/hyp.trn.clean.mapped -M $data/local/dict_nosp/acronyms.map
# NOTE: inputs for map_acronyms_transcripts must be lowercase

# Convert to uppercase
awk '{ print toupper($0); }' $dir/hyp.trn.clean.mapped | sed -e 's/(SW/(sw/g' | sed -e 's/(EN/(en/g' > $dir/hyp.trn.clean.mapped.upper

# Fix stm
cat $data/$set/stm | sed -e 's/_A1/_A/g' | sed -e 's/_B1/_B/g' > $dir/stm

# Convert trn to ctm for hypothesis
trn2ctm.py $dir/hyp.trn.clean.mapped.upper --stm $dir/stm > $dir/hyp.ctm


# For eval2000 score the subsets
case "$set" in
  eval2000* )
    # Score only the, swbd part...
    grep -v '^en_' $dir/stm > $dir/stm.swbd
    grep -v '^en_' $dir/hyp.ctm > $dir/hyp.ctm.swbd
    $hubscr -p $hubdir -V -l english -h hub5 -g $data/$set/glm -r $dir/stm.swbd $dir/hyp.ctm.swbd || exit 1;

    # Score only the, callhome part...
    grep -v '^sw_' $dir/stm > $dir/stm.callhm
    grep -v '^sw_' $dir/hyp.ctm > $dir/hyp.ctm.callhm
    $hubscr -p $hubdir -V -l english -h hub5 -g $data/$set/glm -r $dir/stm.callhm $dir/hyp.ctm.callhm || exit 1;
  ;;

  rt03* )
    # Score only the swbd part...
    grep -v '^fsh_' $dir/stm > $dir/stm.swbd
    grep -v '^fsh_' $dir/hyp.ctm > $dir/hyp.ctm.swbd
    $hubscr -p $hubdir -V -l english -h hub5 -g $data/$set/glm -r $dir/stm.swbd $dir/hyp.ctm.swbd || exit 1;

    # Score only the fisher part...
    grep -v '^sw_' $dir/stm > $dir/stm.fsh
    grep -v '^sw_' $dir/hyp.ctm > $dir/hyp.ctm.fsh
    $hubscr -p $hubdir -V -l english -h hub5 -g $data/$set/glm -r $dir/stm.fsh $dir/hyp.ctm.fsh || exit 1;
  ;;
esac

grep -e Avg -e SPKR -m 2 $dir/hyp.ctm.swbd.filt.sys
grep -e Avg -e SPKR -m 2 $dir/hyp.ctm.callhm.filt.sys
