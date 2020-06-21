#! /bin/bash

# install TIMIT samples (publicly available)
mkdir -p ./sample
wget -P ./sample https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.phn
wget -P ./sample https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.txt
wget -P ./sample https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.wav
wget -P ./sample https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.wrd
