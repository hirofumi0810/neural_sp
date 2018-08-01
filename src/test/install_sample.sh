#! /bin/bash

# install TIMIT samples (public available)
mkdir -p sample
cd sample
wget https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.phn
wget https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.txt
wget https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.wav
wget https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.wrd
