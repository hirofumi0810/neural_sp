#!/bin/bash

. ./path.sh
set -e

### Select GPU
if [ $# -ne 1 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./show_profilig.sh path_to_saved_model" 1>&2
  exit 1
fi

### Set path to save dataset
DATA="/n/sd8/inaguma/corpus/csj/kaldi"

saved_model_path=$1

$PYTHON exp/evaluation/profiling.py --model_path $saved_model_path
