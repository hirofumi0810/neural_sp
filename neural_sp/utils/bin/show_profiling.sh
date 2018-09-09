#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
set -e

### Select GPU
if [ $# -ne 1 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./show_profilig.sh path_to_saved_model" 1>&2
  exit 1
fi

saved_model_path=$1

$PYTHON exp/evaluation/profiling.py --model_path $saved_model_path
