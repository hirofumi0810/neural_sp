#### Conformer LAS large + SpecAugment
- conf: `conf/asr/conformer_kernel15_clamp10_hie_subsample8_las_ln_large.yaml`
- decoding parameters
  - epoch 40
  - beam width: 10
  - lm_weight: 0.0
  - length norm: true

##### WER
| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev_4k|4000|57637|93.6|4.7|1.7|3.2|**9.7**|48.6|
|dev|12000|153743|91.5|6.4|2.0|4.0|**12.5**|53.5|

##### CER
| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev_4k|4000|101224|95.3|3.0|1.7|3.1|**7.8**|46.2|
|dev|12000|273004|93.8|4.0|2.2|3.9|**10.1**|50.9|
|tedx-jp-10k|10000|191708|90.2|5.0|4.8|2.6|**12.4**|64.8|
