#!/usr/bin/env bash

# Prepare dict_nosp/ from lexicon.txt
# This is a simplified version of egs/csj/s5/local/csj_prepare_dict.sh

. ./path.sh
set -e # exit on error

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <lexicon> <dict_dir>"
  exit 1
fi

lexicon=$1
dir=$2

mkdir -p $dir

cat $lexicon |
  awk '{ for(n=2;n<=NF;n++){ phones[$n] = 1; }} END{for (p in phones) print p;}' |
  grep -v sp >$dir/nonsilence_phones.txt || exit 1

(
  echo sp
  echo spn
) >$dir/silence_phones.txt

echo sp >$dir/optional_silence.txt

# No "extra questions" in the input to this setup, as we don't
# have stress or tone.
echo -n >$dir/extra_questions.txt

# Add to the lexicon the silences, noises etc.
(
  echo '<sp> sp'
  echo '<unk> spn'
) | cat - $lexicon >$dir/lexicon.txt || exit 1

sort $dir/lexicon.txt -uo $dir/lexicon.txt

echo "$0: Done preparing $dir"
