# Use caution when comparing these results with other published results.
Training Set   : 3696 sentences
Dev Set        : 400 sentences
Test Set       : 192 sentences Core Test Set (different from Full 1680 sent. set)
Language Model : no
Phone mapping  : Training with 61 phonemes, for testing mapped to 39 phonemes


### BLSTM-CTC
| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev|Sum/Avg|400|15334|80.7|15.0|4.3|2.3|21.7|99.3|
|test|Sum/Avg|192|7333|79.6|15.4|5.0|2.4|22.8|99.5|

### Transformer + SpecAugment
| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev|Sum/Avg|400|15334|81.3|15.4|3.3|2.9|**21.7**|99.8|
|test|Sum/Avg|192|7333|80.2|15.9|4.0|3.3|**23.1**|100.0|

### Transformer + SpecAugment + relative positional encoding (encoder)
| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev|Sum/Avg|400|15334|82.3|14.6|3.0|2.7|**20.4**|99.5|
|test|Sum/Avg|192|7333|81.7|15.0|3.3|3.1|**21.4**|98.4|
