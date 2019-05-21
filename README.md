# NeuralSP: Neural network based Speech Processing

## How to install
```
# Set path to CUDA, NCCL
CUDAROOT=/usr/local/cuda
NCCL_ROOT=/usr/local/nccl

export CPATH=$NCCL_ROOT/include:$CPATH
export LD_LIBRARY_PATH=$NCCL_ROOT/lib/:$CUDAROOT/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$NCCL_ROOT/lib/:$LIBRARY_PATH
export CUDA_HOME=$CUDAROOT
export CUDA_PATH=$CUDAROOT

# Install miniconda, python libraries, and other tools
cd tools
make KALDI=/path/to/kaldi
```


## Data preparation


## Features
### Corpus
#### ASR
- AMI
- CSJ
- Librispeech
- Switchboard (+ Fisher)
- TEDLIUM2
- TEDLIUM3
- TIMIT
- WSJ

#### LM
- Penn Tree Bank
- WikiText2

### Front-end
- Sequence summary network [[link](https://www.isca-speech.org/archive/Interspeech_2018/abstracts/1438.html)]

### Encoder
- CNN encoder
- (Bidirectional/unidirectional) LSTM encoder
- CNN+(bidirectional/unidirectional) LSTM encoder
- Self-attention (Transformer) encoder [[link](https://arxiv.org/abs/1706.03762)]
- Time-Depth Seprarabel (TDS) convolutional encoder [[link](https://arxiv.org/abs/1904.02619)]
- Gated CNN encoder (GLU) [link]

### Connectionist Temporal Classification (CTC) decoder
- beam search
- Shallow fusion [link]

### Attention-based decoder
- RNN decoder
  - Beam search
  - Shallow fusion [link]
  - Cold fusion [[link](https://arxiv.org/abs/1708.06426)]
  - Deep fusion [[link](https://arxiv.org/abs/1503.03535)]
  - Forward-backward attention decoding [[link](https://www.isca-speech.org/archive/Interspeech_2018/abstracts/1160.html)]
  - Ensemble decoding
  - Adaptive softmax [[link](https://arxiv.org/abs/1609.04309)]
- Transformer decoder

### Language model (LM)
- RNNLM (recurrent neural network language model)
- Gated convolutional LM [[link](https://arxiv.org/abs/1612.08083)]

### Output units
- Phoneme
- Grapheme
- Wordpiece (BPE, sentencepiece)
- Word
- Word-char mix

### Multi-task learning (MTL)
Multi-task learning (MTL) with different units are supported to alleviate data sparseness.
- Hybrid CTC/attention [[link](https://www.merl.com/publications/docs/TR2017-190.pdf)]
- Hierarchical Attention (e.g., word attention + character attention) [[link](http://sap.ist.i.kyoto-u.ac.jp/lab/bib/intl/INA-SLT18.pdf)]
- Hierarchical CTC (e.g., word CTC + character CTC) [[link](https://arxiv.org/abs/1711.10136)]
- Hierarchical CTC+Attention (e.g., word attention + character CTC) [[link](http://www.sap.ist.i.kyoto-u.ac.jp/lab/bib/intl/UEN-ICASSP18.pdf)]
- Forward-backward attention [[link](https://www.isca-speech.org/archive/Interspeech_2018/abstracts/1160.html)]
- RNNLM objective [link]


## ASR Performance
### WSJ (WER)
| model                            | test_dev93 | test_eval92 |
| -------------------------------- | ---------- | ----------- |
| BPE1k attn + conv + CTC + RNNLM  | 9.6        | 7.5         |

### CSJ (WER/CER)
| model                                         | eval1   | eval2   | eval3   |
| --------------------------------------------- | ------- | ------- | ------- |
| BPE10k attn + char CTC init. + add 2L + RNNLM | 7.4/5.8 | 5.7/4.5 | 6.0/4.5 |

### Switchboard (WER)
| model                | SWB  | CH   |
| -------------------- | ---- | ---- |
| BPE10k attn + RNNLM  | 11.4 | 23.0 |

### Librispeech (WER)
| model                | dev-clean | dev-other | test-clean | test-other |
| -------------------- | --------- | --------- | ---------- | ---------- |
| BPE30k attn + RNNLM  | 3.6       | 11.2      | 3.9        | 12.2       |

### TEDLIUM2 (WER)
| model                       | dev  | test |
| --------------------------- | ---- | ---- |
| BPE10k attn + conv + RNNLM  | 11.4 | 11.0 |


## LM Performance
### Penn Tree Bank (PPL)
| model       | valid | test  |
| ------------| ----- | ----- |
| RNNLM       | 87.99 | 86.06 |
| + cache=100 | 79.58 | 79.12 |
| + cache=500 | 77.36 | 76.94 |

### WikiText2 (PPL)
| model        | valid  | test  |
| ------------ | ------ | ----- |
| RNNLM        | 104.53 | 98.73 |
| + cache=100  | 90.86  | 85.87 |
| + cache=2000 | 76.10  | 72.77 |


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
