### Transformer
  - conf: `conf/asr/transformer.yaml`
  - lm_conf: `conf/lm/rnnlm.yaml`
  - decoding parameters
    - n_average: 10
    - beam width: 5

|Eval Set|# Snt|# Wrd|Corr|Sub|Del|Ins|Err|S.Err|
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
|dev|14326|205341|94.6|5.2|0.2|0.1|**5.5**|39.0|
|test|7176|104765|94.1|5.6|0.2|0.1|**6.0**|41.2|

### Transformer + SpecAugment (no LM)
  - conf: `conf/asr/transformer.yaml`
  - decoding parameters
    - n_average: 10
    - beam width: 5

|Eval Set|# Snt|# Wrd|Corr|Sub|Del|Ins|Err|S.Err|
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
|dev|14326|205341|95.1|4.8|0.1|0.1|**5.0**|36.0|
|test|7176|104765|94.7|5.1|0.2|0.1|**5.4**|37.7|

### Transformer, subsample1/8
  - conf: `conf/asr/transformer_subsample8.yaml`
  - lm_conf: `conf/lm/rnnlm.yaml`
  - decoding parameters
    - n_average: 10
    - beam width: 5

|Eval Set|# Snt|# Wrd|Corr|Sub|Del|Ins|Err|S.Err|
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
|dev|14326|205341|94.4|5.5|0.2|0.1|**5.7**|39.9|
|test|7176|104765|93.7|6.0|0.3|0.1|**6.4**|42.3|

### Transformer + SpecAugment (no LM), subsample1/8
  - conf: `conf/asr/transformer.yaml`
  - decoding parameters
    - n_average: 10
    - beam width: 5

|Eval Set|# Snt|# Wrd|Corr|Sub|Del|Ins|Err|S.Err|
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
|dev|14326|205341|94.8|5.1|0.1|0.1|**5.3**|37.3|
|test|7176|104765|94.4|5.4|0.2|0.1|**5.7**|39.0|

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
|dev|14326|205341|94.1|5.7|0.2|0.1|**6.0**|41.0|
|test|7176|104765|93.6|6.1|0.3|0.1|**6.5**|42.6|

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
|dev|14326|205341|93.6|6.0|0.4|0.1|**6.5**|42.0|
|test|7176|104765|92.7|6.3|1.0|0.1|**7.5**|45.3|

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
|dev|14326|205341|94.1|5.8|0.2|0.1|**6.1**|41.1|
|test|7176|104765|93.6|6.2|0.2|0.1|**6.6**|43.2|
