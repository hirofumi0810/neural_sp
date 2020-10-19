### BLSTM LAS
| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev-clean|2703|54402|97.5|2.2|0.3|0.3|**2.8**|32.8|
|dev-other|2864|50948|91.7|7.0|1.2|1.1|**9.4**|55.7|
|test-clean|2620|52576|97.3|2.4|0.3|0.4|**3.1**|33.4|
|test-other|2939|52343|91.6|7.3|1.2|1.1|**9.5**|59.0|

  - conf: `conf/asr/blstm_las.yaml`
  - lm_conf: `conf/lm/rnnlm.yaml`
  - decoding parameters
    - beam width: 10


### BLSTM LAS + SpecAugment
| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev-clean|2703|54402|97.7|2.0|0.2|0.3|**2.5**|30.7|
|dev-other|2864|50948|93.6|5.6|0.8|0.8|**7.2**|49.1|
|test-clean|2620|52576|97.6|2.2|0.3|0.3|**2.8**|31.8|
|test-other|2939|52343|93.3|5.9|0.8|0.9|**7.6**|53.1|

  - conf: `conf/asr/blstm_las.yaml`
  - conf2: `conf/data/spec_augment_pt.yaml`
  - lm_conf: `conf/lm/rnnlm.yaml`
  - decoding parameters
    - beam width: 10
  - NOTE: use the above BLSTM LAS as a seed


### Transformer
| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev-clean|2703|54402|97.5|2.2|0.3|0.5|**3.0**|32.5|
|dev-other|2864|50948|92.9|6.5|0.6|1.2|**8.3**|54.3|
|test-clean|2620|52576|97.4|2.3|0.4|0.4|**3.1**|33.0|
|test-other|2939|52343|92.6|6.6|0.8|1.4|**8.8**|56.8|

  - conf: `conf/asr/transformer/transformer.yaml`
  - lm_conf: `conf/lm/rnnlm.yaml`
  - decoding parameters
    - epoch: 40
    - n_average: 10
    - beam width: 10
    - ctc_weight: 0.2


### Transformer + speed perturb + SpecAugment
| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev-clean|2703|54402|97.6|2.0|0.3|0.3|**2.7**|30.5|
|dev-other|2864|50948|94.4|5.0|0.6|1.0|**6.6**|47.2|
|test-clean|2620|52576|97.6|2.1|0.3|0.4|**2.8**|31.5|
|test-other|2939|52343|94.2|5.1|0.7|1.0|**6.8**|50.1|

  - conf: `conf/asr/transformer/transformer.yaml`
  - conf2: `conf/data/spec_augment_speed_perturb_transformer.yaml`
  - lm_conf: `conf/lm/rnnlm.yaml`
  - decoding parameters
    - epoch: 45
    - n_average: 10
    - beam width: 10
    - ctc_weight: 0.2


### Transformer + speed perturb + SpecAugment, large size
| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev-clean|2703|54402|98.1|1.7|0.2|0.2|**2.1**|26.6|
|dev-other|2864|50948|95.4|4.2|0.5|0.7|**5.3**|42.5|
|test-clean|2620|52576|98.0|1.8|0.2|0.3|**2.4**|28.4|
|test-other|2939|52343|95.1|4.4|0.5|0.8|**5.7**|46.5|

  - conf: `conf/asr/transformer/transformer_768dmodel_3072dff_8H.yaml`
  - conf2: `conf/data/spec_augment_speed_perturb_transformer_subsample8.yaml`
  - lm_conf: `conf/lm/rnnlm_6L.yaml`
  - decoding parameters
    - epoch: 35
    - n_average: 10
    - beam width: 10
    - ctc_weight: 0.2


### Transformer, subsample1/8
| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev-clean|2703|54402|97.2|2.4|0.4|0.4|**3.1**|33.9|
|dev-other|2864|50948|92.7|6.6|0.8|1.3|**8.7**|54.7|
|test-clean|2620|52576|97.2|2.5|0.3|0.4|**3.3**|35.4|
|test-other|2939|52343|92.2|6.8|0.9|1.3|**9.1**|58.8|

  - conf: `conf/asr/transformer/transformer_subsample8.yaml`
  - lm_conf: `conf/lm/rnnlm.yaml`
  - decoding parameters
    - epoch: 40
    - n_average: 10
    - beam width: 10
    - ctc_weight: 0.2


### Transformer + speed perturb + SpecAugment, subsample1/8
| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev-clean|2703|54402|97.7|2.1|0.2|0.3|**2.7**|31.4|
|dev-other|2864|50948|94.3|5.2|0.5|0.9|**6.6**|48.0|
|test-clean|2620|52576|97.6|2.1|0.3|0.4|**2.8**|31.5|
|test-other|2939|52343|94.0|5.3|0.7|1.0|**7.0**|51.9|

  - conf: `conf/asr/transformer/transformer_subsample8.yaml`
  - conf2: `conf/data/spec_augment_speed_perturb_transformer_subsample8.yaml`
  - lm_conf: `conf/lm/rnnlm.yaml`
  - decoding parameters
    - epoch: 50
    - n_average: 10
    - beam width: 10
    - ctc_weight: 0.2


### Transformer + speed perturb + SpecAugment, subsample1/8, medium size
| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev-clean|2703|54402|97.9|1.9|0.2|0.3|**2.4**|29.0|
|dev-other|2864|50948|94.8|4.8|0.5|0.8|**6.1**|46.5|
|test-clean|2620|52576|97.7|2.0|0.3|0.3|**2.6**|29.5|
|test-other|2939|52343|94.5|4.9|0.6|0.9|**6.4**|48.6|

  - conf: `conf/asr/transformer/transformer_subsample8_512dmodel_8H.yaml`
  - conf2: `conf/data/spec_augment_speed_perturb_transformer_subsample8.yaml`
  - lm_conf: `conf/lm/rnnlm.yaml`
  - decoding parameters
    - epoch: 50
    - n_average: 10
    - beam width: 10
    - ctc_weight: 0.2


### Transformer + speed perturb + SpecAugment, subsample1/8, large size
| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev-clean|2703|54402|98.0|1.8|0.2|0.2|**2.3**|28.5|
|dev-other|2864|50948|94.9|4.6|0.5|0.7|**5.8**|45.8|
|test-clean|2620|52576|97.8|1.9|0.3|0.3|**2.5**|29.0|
|test-other|2939|52343|94.7|4.7|0.6|0.8|**6.1**|48.3|

  - conf: `conf/asr/transformer/transformer_subsample8_768dmodel_3072dff_8H.yaml`
  - conf2: `conf/data/spec_augment_speed_perturb_transformer_subsample8.yaml`
  - lm_conf: `conf/lm/rnnlm.yaml`
  - decoding parameters
    - epoch: 35
    - n_average: 10
    - beam width: 10
    - ctc_weight: 0.2


<!-- | Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev-clean|
|dev-other|
|test-clean|
|test-other| -->
