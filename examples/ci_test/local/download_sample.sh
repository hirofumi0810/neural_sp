#! /bin/bash

# install TIMIT samples (publicly available)
mkdir -p ./sample
wget --no-check-certificate -P ./sample https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.phn
wget --no-check-certificate -P ./sample https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.txt
wget --no-check-certificate -P ./sample https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.wav
wget --no-check-certificate -P ./sample https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.wrd
