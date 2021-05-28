### Conformer-LAS + SpecAugment (no LM), hierarchical subsample1/8
- conf: `conf/asr/conformer_kernel15_clamp10_hie_subsample8_las_ln_large.yaml`
- decoding parameters
  - epoch: 30
  - n_average: 10
  - beam width: 10
  - lm_weight: 0.0

| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|test_android|5000|49532|94.0|5.8|0.2|0.1|**6.1**|36.7|
|test_ios|5000|49532|94.6|5.2|0.2|0.1|**5.5**|34.1|
|test_mic|5000|49532|94.3|5.6|0.2|0.1|**5.9**|35.7|
