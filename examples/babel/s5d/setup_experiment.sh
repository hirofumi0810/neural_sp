#!/bin/bash

if [ $# -ne 1 ]; then
  echo >&2 "Usage: ./setup_experiment.sh <expname>"
  echo >&2 ""
  echo >&2 "Sets up an experiment to be run in the babel directory with the "
  echo >&2 "provided name."
  exit 1;
fi

expname=$1
cd ..
mkdir ${expname}
cd ${expname}

cp ../s5d/{cmd,path,run}.sh .
cp -P ../s5d/steps .
cp -P ../s5d/utils .
ln -s ../s5d/local .
ln -s ../s5d/conf .
