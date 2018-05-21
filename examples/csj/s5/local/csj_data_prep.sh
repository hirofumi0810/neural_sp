#!/bin/bash

# Copyright  2015 Tokyo Institute of Technology (Authors: Takafumi Moriya and Takahiro Shinozaki)
#            2015 Mitsubishi Electric Research Laboratories (Author: Shinji Watanabe)
# Apache 2.0
# Acknowledgement  This work was supported by JSPS KAKENHI Grant Number 26280055.

# CSJ 291 hours training data preparation.
# "Academic lecture" 270 hours (Exclude the speakers in the evaluation set from all speech data)
# + "Other" 21 hours
# Actually, amount time of training data is 240 hours, this is excluding R-tag utterance and silence section.

# To be run from one directory above this script.

## The input is a directory that contains the CSJ corpus.
## Note: If necessary, rewrite the "cat" command used in the followings
## to locate the .wav file path.

. ./path.sh
set -e # exit on error

#check existing directories
if [ $# -ne 1 ] && [ $# -ne 2 ]; then
  echo "Usage: csj_data_prep.sh <csj-data dir> [<mode_number>]"
  echo " mode_number can be aps_other, aps, all_except_dialog, all, "
  echo "(aps_other=default using academic lecture and other data, "
  echo " aps=using academic lecture data, "
  echo " all_except_dialog=using all data except for dialog data, "
  echo " all=using all data)"
  exit 1;
fi

CSJ=$1
mode=0

if [ $# -eq 2 ]; then
  mode=$2
fi

dir=$DATA/local/train_$mode
mkdir -p $dir

# Audio data directory check
if [ ! -d $CSJ ]; then
 echo "Error: run.sh requires a directory argument"
  exit 1;
fi

# CSJ dictionary file check
# [ ! -f $dir/lexicon.txt ] && cp $CSJ/lexicon/lexicon.txt $dir || exit 1;
[ ! -f $dir/lexicon.txt ] && cp $CSJ/lexicon/lexicon.txt $dir

### Config of using wav data that relates with acoustic model training ###
if [ $mode = 'all' ]; then
  cat $CSJ/*/*/*-wav.list 2>/dev/null | sort > $dir/wav.flist # Using All data
elif [ $mode = 'all_except_dialog' ]; then
  cat $CSJ/*/{A*,M*,R*,S*}/*-wav.list 2>/dev/null | sort > $dir/wav.flist # Using All data except for "dialog" data
elif [ $mode = 'aps' ]; then
  cat $CSJ/*/A*/*-wav.list 2>/dev/null | sort > $dir/wav.flist # Using "Academic lecture" data
elif [ $mode = 'aps_other' ]; then
  # cat $CSJ/*/{A*,M*}/*-wav.list 2>/dev/null | sort > $dir/wav.flist # Using "Academic lecture" and "other" data
  cat $CSJ/*/{A,M}*/*-wav.list 2>/dev/null | sort > $dir/wav.flist # Using "Academic lecture" and "other" data
else
  exit 1;
fi


n=`cat $dir/wav.flist | wc -l`

[ $n -ne 986 ] && \
  echo "Warning: expected 986 data files (Case : Using 'Academic lecture' and 'Other' data), found $n."


# (1a) Transcriptions preparation
# make basic transcription file (add segments info)

##e.g A01F0055_0172 00380.213 00385.951 => A01F0055_0380213_0385951
## for CSJ
awk '{
      spkutt_id=$1;
      split(spkutt_id,T,"[_ ]");
      name=T[1]; stime=$2; etime=$3;
      printf("%s_%07.0f_%07.0f",name, int(1000*stime), int(1000*etime));
      for(i=4;i<=NF;i++) printf(" %s", tolower($i)); printf "\n"
}' $CSJ/*/*/*-trans.text | sort > $dir/transcripts1.txt # This data is for training language models
# Except evaluation set (30 speakers)

# test if trans. file is sorted
export LC_ALL=C;
sort -c $dir/transcripts1.txt || exit 1; # check it's sorted.

# Remove Option.
# **NOTE: modified the pattern matches to make them case insensitive
cat $dir/transcripts1.txt \
  | perl -ane 's:\<s\>::gi;
               s:\<\/s\>::gi;
               print;' \
  | awk '{if(NF > 1) { print; } } ' | sort > $dir/text


# (1c) Make segments files from transcript
#segments file format is: utt-id start-time end-time, e.g.:
#A01F0055_0380213_0385951 => A01F0055_0380213_0385951 A01F0055 00380.213 00385.951
awk '{
       segment=$1;
       split(segment,S,"[_]");
       spkid=S[1]; startf=S[2]; endf=S[3];
       print segment " " spkid " " startf/1000 " " endf/1000
   }' < $dir/text > $dir/segments

sed -e 's?.*/??' -e 's?.wav??' -e 's?\-[R,L]??' $dir/wav.flist | paste - $dir/wav.flist \
  > $dir/wavflist.scp

awk '{
 printf("%s cat %s |\n", $1, $2);
}' < $dir/wavflist.scp | sort > $dir/wav.scp || exit 1;


awk '{segment=$1; split(segment,S,"[_]"); spkid=S[1]; print $1 " " spkid}' $dir/segments > $dir/utt2spk || exit 1;

sort -k 2 $dir/utt2spk | utils/utt2spk_to_spk2utt.pl > $dir/spk2utt || exit 1;

# Copy stuff into its final locations [this has been moved from the format_data script]
mkdir -p $DATA/train_$mode
for f in spk2utt utt2spk wav.scp text segments; do
  cp $DATA/local/train_$mode/$f $DATA/train_$mode || exit 1;
done

echo "CSJ data preparation succeeded."

utils/fix_data_dir.sh $DATA/train_$mode
