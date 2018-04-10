#!/bin/bash
#

if [ -f path.sh ]; then . ./path.sh; fi

if [ $# -ne 1 ]; then
  echo 'Usage: $0 <arpa-lm>'
  exit
fi

silprob=0.5
arpa_lm=$1

[ ! -f $arpa_lm ] && echo No such file $arpa_lm && exit 1;

cp -r $DATA_SAVEPATH/lang $DATA_SAVEPATH/lang_test

gunzip -c "$arpa_lm" | \
  arpa2fst --disambig-symbol=#0 \
           --read-symbol-table=$DATA_SAVEPATH/lang_test/words.txt - $DATA_SAVEPATH/lang_test/G.fst

echo  "Checking how stochastic G is (the first of these numbers should be small):"
fstisstochastic $DATA_SAVEPATH/lang_test/G.fst

## Check lexicon.
## just have a look and make sure it seems sane.
echo "First few lines of lexicon FST:"
fstprint   --isymbols=$DATA_SAVEPATH/lang/phones.txt --osymbols=$DATA_SAVEPATH/lang/words.txt $DATA_SAVEPATH/lang/L.fst  | head

echo Performing further checks

# Checking that G.fst is determinizable.
fstdeterminize $DATA_SAVEPATH/lang_test/G.fst /dev/null || echo Error determinizing G.

# Checking that L_disambig.fst is determinizable.
fstdeterminize $DATA_SAVEPATH/lang_test/L_disambig.fst /dev/null || echo Error determinizing L.

# Checking that disambiguated lexicon times G is determinizable
# Note: we do this with fstdeterminizestar not fstdeterminize, as
# fstdeterminize was taking forever (presumbaly relates to a bug
# in this version of OpenFst that makes determinization slow for
# some case).
fsttablecompose $DATA_SAVEPATH/lang_test/L_disambig.fst $DATA_SAVEPATH/lang_test/G.fst | \
   fstdeterminizestar >/dev/null || echo Error

# Checking that LG is stochastic:
fsttablecompose $DATA_SAVEPATH/lang/L_disambig.fst $DATA_SAVEPATH/lang_test/G.fst | \
   fstisstochastic || echo LG is not stochastic

echo AMI_format_data succeeded.
