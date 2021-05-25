### Conformer-LAS + SpecAugment, hierarchical subsample1/8
- conf: `conf/asr/transformer/conformer_kernel15_clamp10_hie_subsample8_las_ln.yaml`
- decoding parameters
  - epoch: 35
  - n_average: 10
  - beam width: 10
  - lm_weight: 0.3
  - length_norm: true
  - softmax_smoothing: 0.7

| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev|507|17783|94.0|3.8|2.2|1.1|**7.1**|69.0|
|test|1155|27500|94.1|3.5|2.4|1.2|**7.1**|60.9|

##### + iLM estimation
| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev|507|17783|94.1|3.8|2.1|1.1|**7.0**|67.9|
|test|1155|27500|94.2|3.3|2.5|0.9|**6.8**|59.7|


### BLSTM-LAS + SpecAugment
- conf: `conf/asr/las/blstm_las.yaml`
- conf2: `conf/data/spec_augment_speed_perturb.yaml`
- decoding parameters
  - epoch: 40
  - beam width: 10
  - lm_weight: 0.3
  - ctc_weight: 0.2
  - length_norm: true
  - softmax_smoothing: 0.8

| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev|507|17783|93.3|4.5|2.2|1.4|**8.1**|73.4|
|test|1155|27500|93.6|4.0|2.4|1.1|**7.5**|63.9|


### LC-BLSTM-MoChA + SpecAugment
- conf: `conf/asr/mocha/lcblstm_mocha_chunk4040_ctc_sync.yaml`
- conf2: `conf/data/spec_augment_speed_perturb_pretrain_F27_T50.yaml`
- decoding parameters
  - epoch: 30
  - beam width: 10
  - lm_weight: 0.3
  - length_norm: true
  - softmax_smoothing: 0.8
- NOTE* pretrain by `conf/asr/mocha/blstm_mocha.yaml` w/o SpecAugment

| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev|507|17783|91.0|4.8|4.2|1.4|**10.3**|76.3|
|test|1155|27500|92.5|4.4|3.1|1.1|**8.6**|68.5|


### UniLSTM-MoChA + SpecAugment
- conf: `conf/asr/mocha/lstm_mocha_ctc_sync.yaml`
- conf2: `conf/data/spec_augment_speed_perturb_pretrain_F13_T50.yaml`
- decoding parameters
  - epoch: 30
  - beam width: 10
  - lm_weight: 0.3
  - length_norm: true
  - softmax_smoothing: 0.7
- NOTE* pretrain by `conf/asr/mocha/lstm_mocha.yaml` w/o SpecAugment

| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev|507|17783|88.3|6.3|5.4|1.8|**13.5**|83.0|
|test|1155|27500|89.7|6.1|4.2|1.4|**11.6**|77.2|


# UniConformer RNN-T
- conf: `conf/asr/transducer/uni_conformer_kernel7_clamp10_hie_subsample8_rnnt_long_ln_bpe1k.yaml`
- decoding parameters
  - epoch: 40
  - beam width: 10
  - lm_weight: 0.3
  - softmax_smoothing: 0.7

| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev|507|17783|92.7|4.7|2.6|1.2|**8.5**|76.5|
|test|1155|27500|92.8|4.3|2.8|1.0|**8.2**|66.2|


# LC-BLSTM RNN-T
- conf: `conf/asr/transducer/lcblstm_rnnt_40_40_bpe1k.yaml`
- conf: `conf/asr/data/spec_augment_speed_perturb_pretrain_F27_T100.yaml`
- decoding parameters
  - epoch: 30
  - beam width: 10
  - lm_weight: 0.3
  - softmax_smoothing: 0.7
- NOTE* pretrain by `conf/asr/transducer/blstm_rnnt_bpe1k.yaml` w/o SpecAugment

| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev|507|17783|93.4|4.5|2.0|1.4|**8.0**|73.0|
|test|1155|27500|93.4|4.3|2.2|1.1|**7.7**|65.6|


# UniLSTM RNN-T
- conf: `conf/asr/transducer/lstm_rnnt_bpe1k.yaml`
- conf: `conf/asr/data/spec_augment_speed_perturb_pretrain_F13_T50.yaml`
- decoding parameters
  - epoch: 30
  - beam width: 10
  - lm_weight: 0.3
  - softmax_smoothing: 0.7
- NOTE* pretrain by `conf/asr/transducer/lstm_rnnt_bpe1k.yaml` w/o SpecAugment

| Eval Set | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |
| -------- | ----- | ----- | ---- | --- | --- | --- | --- | ----- |
|dev|507|17783|90.9|6.4|2.7|1.6|**10.7**|82.6|
|test|1155|27500|90.8|6.2|3.0|1.4|**10.7**|74.8|
