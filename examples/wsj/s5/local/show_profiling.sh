#!/bin/bash

. ./path.sh
set -e

### Select GPU
if [ $# -ne 1 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./show_profilig.sh path_to_saved_model" 1>&2
  exit 1
fi

${PYTHON} exp/evaluation/profiling.py --model_path $1
