# LAS
#### Conformer LAS large + speed perturb + SpecAugment
- conf: `conf/asr/conformer_kernel15_clamp10_hie_subsample8_las_ln_large.yaml`
- lm_conf: `conf/lm/rnnlm.yaml`
- decoding parameters
  - beam width: 10
  - lm_weight: 0.5
  - length norm: true

| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev-clean|2703|54402|98.2|1.6|0.2|0.2|**2.0**|25.6|
|dev-other|2864|50948|95.8|3.9|0.3|0.6|**4.8**|40.2|
|test-clean|2620|52576|98.1|1.7|0.1|0.3|**2.1**|27.0|
|test-other|2939|52343|95.5|4.0|0.4|0.8|**5.2**|44.5|

###### + iLM estimation (ilm_weight=0.18)
| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev-clean|2703|54402|98.3|1.6|0.1|0.2|**1.9**|24.9|
|dev-other|2864|50948|95.9|3.8|0.3|0.5|**4.6**|39.2|
|test-clean|2620|52576|98.2|1.7|0.1|0.3|**2.1**|26.0|
|test-other|2939|52343|95.8|3.9|0.4|0.7|**4.9**|42.9|


#### Conformer LAS + speed perturb + SpecAugment
- conf: `conf/asr/conformer_kernel15_clamp10_hie_subsample8_las_ln.yaml`
- lm_conf: `conf/lm/rnnlm.yaml`
- decoding parameters
  - beam width: 10
  - lm_weight: 0.5
  - length norm: true

| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev-clean|2703|54402|98.1|1.8|0.2|0.2|**2.2**|26.6|
|dev-other|2864|50948|95.3|4.3|0.4|0.8|**5.5**|42.9|
|test-clean|2620|52576|98.0|1.9|0.2|0.4|**2.4**|28.9|
|test-other|2939|52343|95.0|4.5|0.5|0.8|**5.8**|46.0|

###### + iLM estimation (ilm_weight=0.18)
| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev-clean|2703|54402|98.2|1.6|0.1|0.2|**2.0**|25.8|
|dev-other|2864|50948|95.6|4.1|0.3|0.6|**5.1**|41.7|
|test-clean|2620|52576|98.1|1.8|0.2|0.3|**2.2**|27.7|
|test-other|2939|52343|95.4|4.2|0.4|0.7|**5.3**|44.9|


#### BLSTM LAS + speed perturb + SpecAugment
- conf: `conf/asr/blstm_las.yaml`
- conf2: `conf/data/spec_augment_speed_perturb_pretrain_F27_T100.yaml`
- lm_conf: `conf/lm/rnnlm_6L.yaml`
- decoding parameters
  - beam width: 10
  - lm_weight: 0.5
  - length norm: true
- NOTE: use BLSTM LAS w/o speed perturb and SpecAugment as a seed

| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev-clean|2703|54402|98.0|1.8|0.2|0.2|**2.2**|27.6|
|dev-other|2864|50948|94.1|5.3|0.5|1.3|**7.1**|46.8|
|test-clean|2620|52576|97.8|2.0|0.2|0.3|**2.5**|29.5|
|test-other|2939|52343|94.1|5.3|0.6|0.9|**6.8**|49.8|


#### BLSTM LAS
- conf: `conf/asr/blstm_las.yaml`
- lm_conf: `conf/lm/rnnlm.yaml`
- decoding parameters
  - beam width: 10
  - lm_weight: 0.5
  - length norm: true

| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev-clean|2703|54402|97.6|2.2|0.2|0.3|**2.7**|31.9|
|dev-other|2864|50948|91.9|7.2|0.9|1.3|**9.3**|55.8|
|test-clean|2620|52576|97.4|2.4|0.2|0.5|**3.1**|32.5|
|test-other|2939|52343|91.8|7.3|0.8|1.4|**9.5**|58.4|


# Transformer
#### Transformer large + speed perturb + SpecAugment
- conf: `conf/asr/transformer/transformer_dec_attn_type768dmodel_3072dff_8H.yaml`
- lm_conf: `conf/lm/rnnlm_6L.yaml`
- decoding parameters
  - epoch: 35
  - n_average: 10
  - beam width: 10
  - lm_weight: 0.5
  - ctc_weight: 0.2

| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev-clean|2703|54402|98.1|1.7|0.2|0.2|**2.1**|26.6|
|dev-other|2864|50948|95.4|4.2|0.5|0.7|**5.3**|42.5|
|test-clean|2620|52576|98.0|1.8|0.2|0.3|**2.4**|28.4|
|test-other|2939|52343|95.1|4.4|0.5|0.8|**5.7**|46.5|


#### Transformer medium + speed perturb + SpecAugment
- conf: `conf/asr/transformer/transformer_dec_attn_type512dmodel_2048dff_8H.yaml`
- lm_conf: `conf/lm/rnnlm.yaml`
- decoding parameters
  - epoch: 45
  - n_average: 10
  - beam width: 10
  - lm_weight: 0.5
  - ctc_weight: 0.2

| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev-clean|2703|54402|98.0|1.8|0.2|0.3|**2.3**|27.4|
|dev-other|2864|50948|95.1|4.4|0.5|0.8|**5.7**|43.9|
|test-clean|2620|52576|97.9|1.9|0.2|0.3|**2.4**|28.7|
|test-other|2939|52343|94.9|4.6|0.5|0.8|**5.9**|47.4|


#### Transformer + speed perturb + SpecAugment
- conf: `conf/asr/transformer/transformer.yaml`
- lm_conf: `conf/lm/rnnlm.yaml`
- decoding parameters
  - epoch: 45
  - n_average: 10
  - beam width: 10
  - lm_weight: 0.5
  - ctc_weight: 0.2

| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev-clean|2703|54402|97.6|2.0|0.3|0.3|**2.7**|30.5|
|dev-other|2864|50948|94.4|5.0|0.6|1.0|**6.6**|47.2|
|test-clean|2620|52576|97.6|2.1|0.3|0.4|**2.8**|31.5|
|test-other|2939|52343|94.2|5.1|0.7|1.0|**6.8**|50.1|


#### Transformer large + speed perturb + SpecAugment, subsample1/8
- conf: `conf/asr/transformer/transformer_subsample8_768dmodel_3072dff_8H.yaml`
- lm_conf: `conf/lm/rnnlm.yaml`
- decoding parameters
  - epoch: 35
  - n_average: 10
  - beam width: 10
  - lm_weight: 0.5
  - ctc_weight: 0.2

| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev-clean|2703|54402|98.0|1.8|0.2|0.2|**2.3**|28.5|
|dev-other|2864|50948|94.9|4.6|0.5|0.7|**5.8**|45.8|
|test-clean|2620|52576|97.8|1.9|0.3|0.3|**2.5**|29.0|
|test-other|2939|52343|94.7|4.7|0.6|0.8|**6.1**|48.3|


#### Transformer medium + speed perturb + SpecAugment, subsample1/8
- conf: `conf/asr/transformer/transformer_subsample8_512dmodel_8H.yaml`
- lm_conf: `conf/lm/rnnlm.yaml`
- decoding parameters
  - epoch: 50
  - n_average: 10
  - beam width: 10
  - lm_weight: 0.5
  - ctc_weight: 0.2

| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev-clean|2703|54402|97.9|1.9|0.2|0.3|**2.4**|29.0|
|dev-other|2864|50948|94.8|4.8|0.5|0.8|**6.1**|46.5|
|test-clean|2620|52576|97.7|2.0|0.3|0.3|**2.6**|29.5|
|test-other|2939|52343|94.5|4.9|0.6|0.9|**6.4**|48.6|


#### Transformer + speed perturb + SpecAugment, subsample1/8
- conf: `conf/asr/transformer/transformer_subsample8.yaml`
- lm_conf: `conf/lm/rnnlm.yaml`
- decoding parameters
  - epoch: 50
  - n_average: 10
  - beam width: 10
  - lm_weight: 0.5
  - ctc_weight: 0.2

| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev-clean|2703|54402|97.7|2.1|0.2|0.3|**2.7**|31.4|
|dev-other|2864|50948|94.3|5.2|0.5|0.9|**6.6**|48.0|
|test-clean|2620|52576|97.6|2.1|0.3|0.4|**2.8**|31.5|
|test-other|2939|52343|94.0|5.3|0.7|1.0|**7.0**|51.9|
