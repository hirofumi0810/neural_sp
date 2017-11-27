#!/bin/bash

current_dir=`pwd`
cd ~/tool
git clone --recursive https://github.com/ryanleary/pytorch-ctc.git
cd pytorch-ctc

pip install -r requirements.txt

# build the extension and install python package (requires gcc-5 or later)
# python setup.py install
CC=/usr/bin/gcc-5 CXX=/usr/bin/g++-5 python setup.py install

# If you do NOT require kenlm, the `--recursive` flag is not required on git clone
# and `--exclude-kenlm` should be appended to the `python setup.py install` command

cd $current_dir
