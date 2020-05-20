#!/usr/bin/env bash

TOOL=/home/inaguma/tool/neural_sp_v2
CONDA=$TOOL/miniconda
source $CONDA/etc/profile.d/conda.sh && conda deactivate && conda activate

# pip install pytest
pytest test_encoder.py
