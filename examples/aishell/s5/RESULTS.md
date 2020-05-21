### Transformer
  - conf: `conf/asr/transformer.yaml`
  - lm_conf: `conf/lm/rnnlm.yaml`
  - decoding parameters
    - n_average: 10
    - beam width: 5

|Eval Set|# Snt|# Wrd|Corr|Sub|Del|Ins|Err|S.Err|
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
|dev|14326|616023|95.1|4.6|0.2|0.2|**5.1**|39.0|
|test|7176|314295|94.7|5.0|0.3|0.2|**5.5**|41.2|

### Transformer + SpecAugment (no LM)
  - conf: `conf/asr/transformer.yaml`
  - decoding parameters
    - n_average: 10
    - beam width: 5

|Eval Set|# Snt|# Wrd|Corr|Sub|Del|Ins|Err|S.Err|
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
|dev|14326|616023|95.6|4.3|0.2|0.2|**4.6**|36.0|
|test|7176|314295|95.2|4.5|0.3|0.2|**5.0**|37.6|

### Transformer, subsample1/8
  - conf: `conf/asr/transformer_subsample8.yaml`
  - lm_conf: `conf/lm/rnnlm.yaml`
  - decoding parameters
    - n_average: 10
    - beam width: 5

|Eval Set|# Snt|# Wrd|Corr|Sub|Del|Ins|Err|S.Err|
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
|dev|14326|616023|94.9|4.8|0.3|0.2|**5.3**|39.9|
|test|7176|314295|94.3|5.3|0.4|0.2|**5.9**|42.3|

### Transformer + SpecAugment (no LM), subsample1/8
  - conf: `conf/asr/transformer.yaml`
  - decoding parameters
    - n_average: 10
    - beam width: 5

|Eval Set|# Snt|# Wrd|Corr|Sub|Del|Ins|Err|S.Err|
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
|dev|14326|616023|94.8|4.8|0.4|0.2|**5.4**|38.4|
|test|7176|314295|94.4|5.0|0.6|0.2|**5.8**|39.8|

### Offline Transformer-MMA, subsample1/8
  - conf: `conf/asr/transformer_mocha_mono4H_chunk4H_chunk16_from4L_headdrop0.5_subsample8.yaml`
  - lm_conf: `conf/lm/rnnlm.yaml`
  - decoding parameters
    - n_average: 10
    - beam width: 10
    - length_penalty: 2.0
    - mma_delay_threshold: 8

|Eval Set|# Snt|# Wrd|Corr|Sub|Del|Ins|Err|S.Err|
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
|dev|14326|616023|94.7|5.1|0.2|0.2|**5.5**|41.0|
|test|7176|314295|94.2|5.4|0.4|0.2|**6.0**|42.6|

### Streaming Transformer-MMA, subsample1/8, 96/64/32
  - conf: `conf/asr/lc_transformer_mocha_mono4H_chunk4H_chunk16_from4L_headdrop0.5_subsample8_96_64_32.yaml`
  - lm_conf: `conf/lm/rnnlm.yaml`
  - decoding parameters
    - n_average: 10
    - beam width: 10
    - length_penalty: 2.0
    - mma_delay_threshold: 8

|Eval Set|# Snt|# Wrd|Corr|Sub|Del|Ins|Err|S.Err|
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
|dev|14326|616023|94.2|5.3|0.5|0.2|**6.0**|42.0|
|test|7176|314295|93.3|5.6|1.1|0.2|**6.9**|45.3|

### Streaming Transformer-MMA, subsample1/8, 64/128/64
  - conf: `conf/asr/lc_transformer_mocha_mono4H_chunk4H_chunk16_from4L_headdrop0.5_subsample8_64_128_64.yaml`
  - lm_conf: `conf/lm/rnnlm.yaml`
  - decoding parameters
    - n_average: 10
    - beam width: 10
    - length_penalty: 2.0
    - mma_delay_threshold: 8

|Eval Set|# Snt|# Wrd|Corr|Sub|Del|Ins|Err|S.Err|
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
|dev|14326|616023|94.6|5.1|0.2|0.2|**5.6**|41.1|
|test|7176|314295|94.2|5.5|0.3|0.2|**6.1**|43.2|
