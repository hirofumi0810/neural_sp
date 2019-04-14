# NeuralSP: Neural network based Speech Processing

## How to install

## Data preparation

## Features
### Connectionist Temporal Classification (CTC)
  - beam search
  - Shallow fusion [link]

### Attention-based sequence-to-sequence
#### Encoder
  - CNN encoder
  - (bidirectional/unidirectional) LSTM encoder
  - CNN+(bidirectional/unidirectional) LSTM encoder
  - self-attention (Transformer) encoder [[link](https://arxiv.org/abs/1706.03762)]
  - Time-Depth Seprarabel (TDS) convolutional encoder [[link](https://arxiv.org/abs/1904.02619)] (<font color="Red">NEW!</font>)

#### Decoder
  - RNN decoder
    - Beam search
    - Shallow fusion [link]
    - Cold fusion [[link](https://arxiv.org/abs/1708.06426)]
    - Deep fusion [[link](https://arxiv.org/abs/1503.03535)]
    - Forward-backward attention decoding [[link](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1160.pdf)]
    <!-- - cache -->
  - Transformer decoder

#### Attention
  - RNN decoder
    - location
    - additive
    - dot-product
    - Luong's dot/general/concat [[link](https://aclweb.org/anthology/D15-1166)]
    - Multi-headed dor-product [[link](https://arxiv.org/abs/1706.03762)]
  - Transformer decoder
    - Multi-headed dor-product [[link](https://arxiv.org/abs/1706.03762)]

### Language model (LM)
  - RNNLM (recurrent neural network language model)
  - Gated convolutional LM [[link](https://arxiv.org/abs/1612.08083)]

### Output units
  - phoneme (TIMIT, Switchboard)
  - grapheme
  - wordpiece (BPE, sentencepiece)
  - word
  - word-char mix

### Multi-task learning (MTL)
Multi-task learning (MTL) with different units are supported to alleviate data sparseness.
  - Hybrid CTC/attention [[link](https://www.merl.com/publications/docs/TR2017-190.pdf)]
  - Hierarchical Attention (e.g., word attention + character CTC) [[link](http://sap.ist.i.kyoto-u.ac.jp/lab/bib/intl/INA-SLT18.pdf)]
  - Hierarchical CTC (e.g., word CTC + character CTC) [[link](https://arxiv.org/abs/1711.10136)]
  - Hierarchical CTC+Attention (e.g., word attention + character CTC) [[link](http://www.sap.ist.i.kyoto-u.ac.jp/lab/bib/intl/UEN-ICASSP18.pdf)]
  - Forward-backward attention [[link](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1160.pdf)]
  - RNNLM objective [link]

## Performance (word error rate)
### WSJ
| model | test_dev93 | test_eval92 |
| --- | --- | --- |
| Char attn | 17.1 | 14.1 |
| + RNNLM | N/A | N/A |
| BPE1k attn | 15.1 | 12.4 |
| + RNNLM | 11.8 | 10.3 |

### CSJ
| model | eva1l | eval2 | eval3 |
| --- | --- | --- | --- |
| Char attn | N/A | N/A | N/A |
| + RNNLM  | N/A | N/A | N/A |
| BPE30k attn | 8.8 | 6.3 | 6.9 |
| + RNNLM | 8.2 | 6.0 | 6.7 |
| Word30k attn | 9.3 | 7.0 | 7.9 |
| + RNNLM | 8.9 | 6.9 | 7.6 |
| + Char attn | 8.8 | 6.8 | 7.6 |
| + OOV resolution | 8.3 | 6.1 | 6.7 |

### Switchboard
| model | SWB | CH |
| --- | --- | --- |
| Char attn | N/A | N/A |
| BPE10k attn | 11.8 | 23.5 |
| + RNNLM | 11.0 | 23.3 |
| Word10k attn | N/A | N/A |

### Librispeech
| model | dev-clean | dev-other | test-clean | test-other |
| --- | --- | --- | --- | --- |
| Char attn | N/A | N/A | N/A | N/A |
| BPE30k attn | N/A | N/A | N/A | N/A |
| Word30k attn | N/A | N/A | N/A | N/A |


## Reference
- https://github.com/kaldi-asr/kaldi
- https://github.com/espnet/espnet
- https://github.com/awni/speech

<!-- ## TODO
- WFST decoder
- Minimum WER training
- Convolutional decoder
- Speech Translation
- Tacotron2 -->
