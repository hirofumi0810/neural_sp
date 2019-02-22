#!/bin/bash

# Copyright 2017 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

cmd=run.pl
add_deltas=false
nj=1

. utils/parse_options.sh

feat_scp=$1
cvmn_ark=$2
log_dir=$3
dump_dir=$4

if [ $# != 4 ]; then
    echo "Usage: $0 <feat_scp> <cmvn_ark> <log_dir> <dump_dir>"
    exit 1;
fi

mkdir -p $log_dir
mkdir -p $dump_dir

dump_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' ${dump_dir} ${PWD}`

# split scp file
split_feet_scps=""
for n in $(seq $nj); do
  split_feet_scps="$split_feet_scps $log_dir/feats.$n.scp"
done

utils/split_scp.pl $feat_scp $split_feet_scps || exit 1;

# dump features
if ${add_deltas};then
  $cmd JOB=1:$nj $log_dir/dump_feature.JOB.log \
    apply-cmvn --norm-vars=true $cvmn_ark scp:$log_dir/feats.JOB.scp ark:- \| \
    add-deltas ark:- ark:- \| \
    copy-feats --compress=true --compression-method=2 ark:- ark,scp:${dump_dir}/feats.JOB.ark,${dump_dir}/feats.JOB.scp || exit 1;
else
  $cmd JOB=1:$nj $log_dir/dump_feature.JOB.log \
    apply-cmvn --norm-vars=true $cvmn_ark scp:$log_dir/feats.JOB.scp ark:- \| \
    copy-feats --compress=true --compression-method=2 ark:- ark,scp:${dump_dir}/feats.JOB.ark,${dump_dir}/feats.JOB.scp || exit 1;
fi

# concatenate scp files
for n in $(seq $nj); do
  cat $dump_dir/feats.$n.scp || exit 1;
done > $dump_dir/feats.scp

# remove temp scps
rm $log_dir/feats.*.scp 2>/dev/null
echo "Succeeded dumping features for " `dirname $feat_scp`
