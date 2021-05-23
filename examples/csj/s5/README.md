#### Conformer LAS large + speed perturb + SpecAugment
- conf: `conf/asr/conformer_kernel15_clamp10_hie_subsample8_las_ln_large.yaml`
- lm_conf: `conf/lm/rnnlm.yaml`
- decoding parameters
  - epoch 30
  - beam width: 10
  - lm_weight: 0.3
  - length norm: true

##### WER
| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|eval1|1272|26028|95.3|3.8|0.9|1.0|**5.7**|48.0|
|eval2|1292|26661|96.3|3.2|0.6|0.7|**4.4**|47.4|
|eval3|1385|17189|95.9|3.4|0.7|0.8|**4.9**|33.2|

##### CER
| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|eval1|1272|43897|96.7|2.0|1.3|0.8|**4.2**|47.0|
|eval2|1292|43623|97.3|1.8|0.9|0.6|**3.3**|46.6|
|eval3|1385|28225|97.1|1.9|1.0|0.9|**3.7**|32.4|


#### Conformer LAS + speed perturb + SpecAugment
- conf: `conf/asr/conformer_kernel15_clamp10_hie_subsample8_las_ln.yaml`
- lm_conf: `conf/lm/rnnlm.yaml`
- decoding parameters
  - epoch 30
  - beam width: 10
  - lm_weight: 0.3
  - length norm: true

##### WER
| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|eval1|1272|26028|95.0|4.0|1.0|1.0|**6.0**|49.1|
|eval2|1292|26661|96.1|3.4|0.6|0.7|**4.7**|48.9|
|eval3|1385|17189|95.6|3.6|0.7|0.9|**5.2**|34.7|

##### CER
| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|eval1|1272|43897|96.5|2.1|1.4|0.9|**4.4**|48.4|
|eval2|1292|43623|97.1|2.0|0.9|0.7|**3.6**|48.0|
|eval3|1385|28225|96.9|2.0|1.1|0.9|**4.0**|33.9|
