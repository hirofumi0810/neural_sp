#!/usr/bin/env bash

source tools/neural_sp/miniconda/bin/activate

modules="neural_sp test utils setup.py"
pycodestyle -r ${modules} --show-source --show-pep8 --ignore="E501"

pytest