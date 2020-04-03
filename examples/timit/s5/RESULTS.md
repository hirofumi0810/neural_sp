# Use caution when comparing these results with other published results.
Training Set   : 3696 sentences

Dev Set        : 400 sentences

Test Set       : 192 sentences Core Test Set (different from Full 1680 sent. set)

Language Model : no

Phone mapping  : Training with 61 phonemes, for testing mapped to 39 phonemes


### BLSTM -CTC
#### dev
|SPKR|# Snt|# Wrd|Corr|Sub|Del|Ins|Err|S.Err|
| -- | -- | -- | -- | -- | -- | -- | -- | -- |
|Sum/Avg|400|15334|80.7|15.0|4.3|2.3|21.7|99.3|
#### test
|SPKR|# Snt|# Wrd|Corr|Sub|Del|Ins|Err|S.Err|
| -- | -- | -- | -- | -- | -- | -- | -- | -- |
|Sum/Avg|192|7333|79.6|15.4|5.0|2.4|22.8|99.5|

### Transformer + SpecAugment
#### dev
|SPKR|# Snt|# Wrd|Corr|Sub|Del|Ins|Err|S.Err|
| -- | -- | -- | -- | -- | -- | -- | -- | -- |
|Sum/Avg|400|15334|81.3|15.4|3.3|2.9|21.7|99.8|
#### test
|SPKR|# Snt|# Wrd|Corr|Sub|Del|Ins|Err|S.Err|
| -- | -- | -- | -- | -- | -- | -- | -- | -- |
|Sum/Avg|192|7333|80.2|15.9|4.0|3.3|23.1|100.0|
