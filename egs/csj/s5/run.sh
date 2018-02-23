#!/bin/bash

# Copyright  2015 Tokyo Institute of Technology
#                 (Authors: Takafumi Moriya, Tomohiro Tanaka and Takahiro Shinozaki)
#            2015 Mitsubishi Electric Research Laboratories (Author: Shinji Watanabe)
# Apache 2.0
# Acknowledgement  This work was supported by JSPS KAKENHI Grant Number 26280055.

# This recipe is based on the Switchboard corpus recipe, by Arnab Ghoshal,
# in the egs/swbd/s5c/ directory.

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.
# Caution: some of the graph creation steps use quite a bit of memory, so you
# should run this on a machine that has sufficient memory.

. ./cmd.sh
. ./path.sh
set -e # exit on error

#: << '#SKIP'

use_dev=true # Use the first 4k sentences from training data as dev set. (39 speakers.)

CSJDATATOP="/n/rd25/mimura/corpus/CSJ"
#CSJDATATOP=/db/laputa1/$CSJDATA_SAVEPATH/processed/public/CSJ ## CSJ database top directory.
CSJVER=dvd  ## Set your CSJ format (dvd or usb).
            ## Usage    :
            ## Case DVD : We assume CSJ DVDs are copied in this directory with the names dvd1, dvd2,...,dvd17.
            ##            Neccesary directory is dvd3 - dvd17.
            ##            e.g. $ ls $CSJDATATOP(DVD) => 00README.txt dvd1 dvd2 ... dvd17
            ##
            ## Case USB : Neccesary directory is MORPH/SDB and WAV
            ##            e.g. $ ls $CSJDATATOP(USB) => 00README.txt DOC MORPH ... WAV fileList.csv
            ## Case merl :MERL setup. Neccesary directory is WAV and sdb

DATASIZE=subset
# DATASIZE=aps
# DATASIZE=fullset
# DATASIZE=all
export CSJDATA_SAVEPATH=/home/lab5/inaguma/corpus/csj/kaldi/$DATASIZE

if [ ! -e $CSJDATA_SAVEPATH/csj-data/.done_make_all ]; then
 echo "CSJ transcription file does not exist"
 #local/csj_make_trans/csj_autorun.sh <RESOUCE_DIR> <MAKING_PLACE(no change)> || exit 1;
 local/csj_make_trans/csj_autorun.sh $CSJDATATOP $CSJDATA_SAVEPATH/csj-data $CSJVER
fi
wait

[ ! -e $CSJDATA_SAVEPATH/csj-data/.done_make_all ]\
    && echo "Not finished processing CSJ data" && exit 1;

# Prepare Corpus of Spontaneous Japanese (CSJ) data.
# Processing CSJ data to KALDI format based on switchboard recipe.
# local/csj_data_prep.sh <SPEECH_and_TRANSCRIPTION_DATA_DIRECTORY> [ <mode_number> ]
# mode_number can be 0, 1, 2, 3 (0=default using "Academic lecture" and "other" data,
#                                1=using "Academic lecture" data,
#                                2=using All data except for "dialog" data, 3=using All data )
if [ $DATASIZE = 'subset' ]; then
  local/csj_data_prep.sh $CSJDATA_SAVEPATH/csj-data  # subset (240h)
elif [ $DATASIZE = 'aps' ]; then
  local/csj_data_prep.sh $CSJDATA_SAVEPATH/csj-data 1  # aps
elif [ $DATASIZE = 'fullset' ]; then
  local/csj_data_prep.sh $CSJDATA_SAVEPATH/csj-data 2  # fullset (586h)
elif [ $DATASIZE = 'all' ]; then
  local/csj_data_prep.sh $CSJDATA_SAVEPATH/csj-data 3  # all
fi

local/csj_prepare_dict.sh

# utils/prepare_lang.sh --num-sil-states 4 $CSJDATA_SAVEPATH/local/dict_nosp "<unk>" $CSJDATA_SAVEPATH/local/lang_nosp $CSJDATA_SAVEPATH/lang_nosp

# Now train the language models.
# local/csj_train_lms.sh $CSJDATA_SAVEPATH/local/train/text $CSJDATA_SAVEPATH/local/dict_nosp/lexicon.txt $CSJDATA_SAVEPATH/local/lm

# We don't really need all these options for SRILM, since the LM training script
# does some of the same processing (e.g. -subset -tolower)
# srilm_opts="-subset -prune-lowprobs -unk -tolower -order 3"
# LM=$CSJDATA_SAVEPATH/local/lm/csj.o3g.kn.gz
# utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
#   $CSJDATA_SAVEPATH/lang_nosp $LM $CSJDATA_SAVEPATH/local/dict_nosp/lexicon.txt $CSJDATA_SAVEPATH/lang_nosp_csj_tg

# Data preparation and formatting for evaluation set.
# CSJ has 3 types of evaluation data
#local/csj_eval_data_prep.sh <SPEECH_and_TRANSCRIPTION_DATA_DIRECTORY_ABOUT_EVALUATION_DATA> <EVAL_NUM>
for eval_num in eval1 eval2 eval3 ; do
    local/csj_eval_data_prep.sh $CSJDATA_SAVEPATH/csj-data/eval $eval_num
done

echo "OK."
exit 1

# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc

for x in train eval1 eval2 eval3; do
  steps/make_mfcc.sh --nj 50 --cmd "$train_cmd" \
    $CSJDATA_SAVEPATH/$x exp/make_mfcc/$x $mfccdir
  steps/compute_cmvn_stats.sh $CSJDATA_SAVEPATH/$x exp/make_mfcc/$x $mfccdir
  utils/fix_data_dir.sh $CSJDATA_SAVEPATH/$x
done

echo "Finish creating MFCCs"

#SKIP

##### Training and Decoding steps start from here #####

# Use the first 4k sentences as dev set.  Note: when we trained the LM, we used
# the 1st 10k sentences as dev set, so the 1st 4k won't have been used in the
# LM training data.   However, they will be in the lexicon, plus speakers
# may overlap, so it's still not quite equivalent to a test set.

if $use_dev ;then
    dev_set=train_dev
    utils/subset_data_dir.sh --first $CSJDATA_SAVEPATH/train 4000 $CSJDATA_SAVEPATH/$dev_set # 6hr 31min
    n=$[`cat $CSJDATA_SAVEPATH/train/segments | wc -l` - 4000]
    utils/subset_data_dir.sh --last $CSJDATA_SAVEPATH/train $n $CSJDATA_SAVEPATH/train_nodev
else
    cp -r $CSJDATA_SAVEPATH/train $CSJDATA_SAVEPATH/train_nodev
fi

# Calculate the amount of utterance segmentations.
# perl -ne 'split; $s+=($_[3]-$_[2]); END{$h=int($s/3600); $r=($s-$h*3600); $m=int($r/60); $r-=$m*60; printf "%.1f sec -- %d:%d:%.1f\n", $s, $h, $m, $r;}' $CSJDATA_SAVEPATH/train/segments

# Now-- there are 162k utterances (240hr 8min), and we want to start the
# monophone training on relatively short utterances (easier to align), but want
# to exclude the shortest ones.
# Therefore, we first take the 100k shortest ones;
# remove most of the repeated utterances, and
# then take 10k random utterances from those (about 8hr 9mins)
utils/subset_data_dir.sh --shortest $CSJDATA_SAVEPATH/train_nodev 100000 $CSJDATA_SAVEPATH/train_100kshort
utils/subset_data_dir.sh $CSJDATA_SAVEPATH/train_100kshort 30000 $CSJDATA_SAVEPATH/train_30kshort

# Take the first 100k utterances (about half the data); we'll use
# this for later stages of training.
utils/subset_data_dir.sh --first $CSJDATA_SAVEPATH/train_nodev 100000 $CSJDATA_SAVEPATH/train_100k
utils/$CSJDATA_SAVEPATH/remove_dup_utts.sh 200 $CSJDATA_SAVEPATH/train_100k $CSJDATA_SAVEPATH/train_100k_nodup  # 147hr 6min

# Finally, the full training set:
utils/$CSJDATA_SAVEPATH/remove_dup_utts.sh 300 $CSJDATA_SAVEPATH/train_nodev $CSJDATA_SAVEPATH/train_nodup  # 233hr 36min

## Starting basic training on MFCC features
steps/train_mono.sh --nj 50 --cmd "$train_cmd" \
  $CSJDATA_SAVEPATH/train_30kshort $CSJDATA_SAVEPATH/lang_nosp exp/mono

steps/align_si.sh --nj 50 --cmd "$train_cmd" \
  $CSJDATA_SAVEPATH/train_100k_nodup $CSJDATA_SAVEPATH/lang_nosp exp/mono exp/mono_ali

steps/train_deltas.sh --cmd "$train_cmd" \
  3200 30000 $CSJDATA_SAVEPATH/train_100k_nodup $CSJDATA_SAVEPATH/lang_nosp exp/mono_ali exp/tri1

graph_dir=exp/tri1/graph_csj_tg
$train_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh $CSJDATA_SAVEPATH/lang_nosp_csj_tg exp/tri1 $graph_dir
for eval_num in eval1 eval2 eval3 $dev_set ; do
    steps/decode_si.sh --nj 10 --cmd "$decode_cmd" --config conf/decode.config \
	$graph_dir $CSJDATA_SAVEPATH/$eval_num exp/tri1/decode_${eval_num}_csj
done

steps/align_si.sh --nj 50 --cmd "$train_cmd" \
  $CSJDATA_SAVEPATH/train_100k_nodup $CSJDATA_SAVEPATH/lang_nosp exp/tri1 exp/tri1_ali

steps/train_deltas.sh --cmd "$train_cmd" \
  4000 70000 $CSJDATA_SAVEPATH/train_100k_nodup $CSJDATA_SAVEPATH/lang_nosp exp/tri1_ali exp/tri2

# The previous mkgraph might be writing to this file.  If the previous mkgraph
# is not running, you can remove this loop and this mkgraph will create it.
while [ ! -s $CSJDATA_SAVEPATH/lang_nosp_csj_tg/tmp/CLG_3_1.fst ]; do sleep 60; done
sleep 20; # in case still writing.
graph_dir=exp/tri2/graph_csj_tg
$train_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh $CSJDATA_SAVEPATH/lang_nosp_csj_tg exp/tri2 $graph_dir
for eval_num in eval1 eval2 eval3 $dev_set ; do
    steps/decode.sh --nj 10 --cmd "$decode_cmd" --config conf/decode.config \
	$graph_dir $CSJDATA_SAVEPATH/$eval_num exp/tri2/decode_${eval_num}_csj
done

# From now, we start with the LDA+MLLT system
steps/align_si.sh --nj 50 --cmd "$train_cmd" \
  $CSJDATA_SAVEPATH/train_100k_nodup $CSJDATA_SAVEPATH/lang_nosp exp/tri2 exp/tri2_ali_100k_nodup

# From now, we start using all of the data (except some duplicates of common
# utterances, which don't really contribute much).
steps/align_si.sh --nj 50 --cmd "$train_cmd" \
  $CSJDATA_SAVEPATH/train_nodup $CSJDATA_SAVEPATH/lang_nosp exp/tri2 exp/tri2_ali_nodup

# Do another iteration of LDA+MLLT training, on all the data.
steps/train_lda_mllt.sh --cmd "$train_cmd" \
  6000 140000 $CSJDATA_SAVEPATH/train_nodup $CSJDATA_SAVEPATH/lang_nosp exp/tri2_ali_nodup exp/tri3

graph_dir=exp/tri3/graph_csj_tg
$train_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh $CSJDATA_SAVEPATH/lang_nosp_csj_tg exp/tri3 $graph_dir
for eval_num in eval1 eval2 eval3 $dev_set ; do
    steps/decode.sh --nj 10 --cmd "$decode_cmd" --config conf/decode.config \
	$graph_dir $CSJDATA_SAVEPATH/$eval_num exp/tri3/decode_${eval_num}_csj_nosp
done

# Now we compute the pronunciation and silence probabilities from training data,
# and re-create the lang directory.
steps/get_prons.sh --cmd "$train_cmd" $CSJDATA_SAVEPATH/train_nodup $CSJDATA_SAVEPATH/lang_nosp exp/tri3
utils/dict_dir_add_pronprobs.sh --max-normalize true \
  $CSJDATA_SAVEPATH/local/dict_nosp exp/tri3/pron_counts_nowb.txt exp/tri3/sil_counts_nowb.txt \
  exp/tri3/pron_bigram_counts_nowb.txt $CSJDATA_SAVEPATH/local/dict

utils/prepare_lang.sh $CSJDATA_SAVEPATH/local/dict "<unk>" $CSJDATA_SAVEPATH/local/lang $CSJDATA_SAVEPATH/lang
LM=$CSJDATA_SAVEPATH/local/lm/csj.o3g.kn.gz
srilm_opts="-subset -prune-lowprobs -unk -tolower -order 3"
utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
  $CSJDATA_SAVEPATH/lang $LM $CSJDATA_SAVEPATH/local/dict/lexicon.txt $CSJDATA_SAVEPATH/lang_csj_tg

graph_dir=exp/tri3/graph_csj_tg
$train_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh $CSJDATA_SAVEPATH/lang_csj_tg exp/tri3 $graph_dir
for eval_num in eval1 eval2 eval3 $dev_set ; do
    steps/decode.sh --nj 10 --cmd "$decode_cmd" --config conf/decode.config \
        $graph_dir $CSJDATA_SAVEPATH/$eval_num exp/tri3/decode_${eval_num}_csj
done


# Train tri4, which is LDA+MLLT+SAT, on all the (nodup) data.
steps/align_fmllr.sh --nj 50 --cmd "$train_cmd" \
  $CSJDATA_SAVEPATH/train_nodup $CSJDATA_SAVEPATH/lang exp/tri3 exp/tri3_ali_nodup

steps/train_sat.sh  --cmd "$train_cmd" \
  11500 200000 $CSJDATA_SAVEPATH/train_nodup $CSJDATA_SAVEPATH/lang exp/tri3_ali_nodup exp/tri4

graph_dir=exp/tri4/graph_csj_tg
$train_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh $CSJDATA_SAVEPATH/lang_csj_tg exp/tri4 $graph_dir
for eval_num in eval1 eval2 eval3 $dev_set ; do
    steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" --config conf/decode.config \
	$graph_dir $CSJDATA_SAVEPATH/$eval_num exp/tri4/decode_${eval_num}_csj
done

steps/align_fmllr.sh --nj 50 --cmd "$train_cmd" \
  $CSJDATA_SAVEPATH/train_nodup $CSJDATA_SAVEPATH/lang exp/tri4 exp/tri4_ali_nodup || exit 1

# You can execute DNN training script [e.g. local/chain/run_dnn.sh] from here.

# MMI training
# local/run_mmi.sh

# this will help find issues with the lexicon.
# steps/cleanup/debug_lexicon.sh --nj 300 --cmd "$train_cmd" $CSJDATA_SAVEPATH/train_nodev $CSJDATA_SAVEPATH/lang exp/tri4 $CSJDATA_SAVEPATH/local/dict/lexicon.txt exp/debug_lexicon

# SGMM system
# local/run_sgmm2.sh

#SKIP

##### Start DNN training #####
# Karel's DNN recipe on top of fMLLR features
# local/nnet/run_dnn.sh

# nnet3 TDNN+Chain
local/chain/run_tdnn.sh

# nnet3 TDNN recipe
# local/nnet3/run_tdnn.sh

##### Start RNN-LM training for rescoring #####
# local/csj_run_rnnlm.sh

# getting results (see RESULTS file)
# for eval_num in eval1 eval2 eval3 $dev_set ; do
#     echo "=== evaluation set $eval_num ===" ;
#     for x in exp/{tri,dnn}*/decode_${eval_num}*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done ;
# done
