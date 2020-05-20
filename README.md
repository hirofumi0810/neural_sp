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
export CPATH=$CUDA_PATH/include:$CPATH  # for warp-rnnt

# Install miniconda, python libraries, and other tools
cd tools
make KALDI=/path/to/kaldi
```


## Data preparation


## Features
### Corpus
#### ASR
- AISHELL-1
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
- Frame stacking
- Sequence summary network [[link](https://www.isca-speech.org/archive/Interspeech_2018/abstracts/1438.html)]
- SpecAugment [[link](https://arxiv.org/abs/1904.08779)]

### Encoder
- CNN encoder
- LSTM encoder
- CNN+LSTM encoder
- Transformer encoder [[link](https://arxiv.org/abs/1706.03762)]
- Time-Depth Seprarabel (TDS) convolutional encoder [[link](https://arxiv.org/abs/1904.02619)]
- Gated CNN encoder (GLU) [[link](https://openreview.net/forum?id=Hyig0zb0Z)]

### Connectionist Temporal Classification (CTC) decoder
- Shallow fusion

### Attention-based decoder
- RNN decoder
  - Shallow fusion
  - Cold fusion [[link](https://arxiv.org/abs/1708.06426)]
  - Deep fusion [[link](https://arxiv.org/abs/1503.03535)]
  - Forward-backward attention decoding [[link](https://www.isca-speech.org/archive/Interspeech_2018/abstracts/1160.html)]
  - Ensemble decoding
- Streaming RNN decoder
  - Hard monotonic attention [[link](https://arxiv.org/abs/1704.00784)]
  - Monotonic chunkwise attention (MoChA) [[link](https://arxiv.org/abs/1712.05382)]
- Transformer decoder [[link](https://arxiv.org/abs/1706.03762)]
- RNN transducer [[link](https://arxiv.org/abs/1211.3711)]
<!-- - Transformer transducer [[link]()] -->
<!-- - Monotonic Multihead Attention [[link]()] -->

### Language model (LM)
- RNNLM (recurrent neural network language model)
- Gated convolutional LM [[link](https://arxiv.org/abs/1612.08083)]
- Transformer LM
- Transformer-XL LM [[link](https://arxiv.org/abs/1901.02860)]
- Adaptive softmax [[link](https://arxiv.org/abs/1609.04309)]

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
### AISHELL-1 (CER)
| model       | dev | test |
| ----------- | --- | ---- |
| Transformer | 5.1 | 5.3  |

### CSJ (WER)
| model                                        | eval1   | eval2   | eval3   |
| -------------------------------------------- | ------- | ------- | ------- |
| BPE10k LAS + RNNLM | 7.9 | 5.8 | 6.4 |
|   + SpecAugment    | 6.5 | 5.1 | 5.6 |

### Switchboard 300h (WER)
| model               | SWB  | CH   |
| ------------------- | ---- | ---- |
| BPE10k LAS + RNNLM  | 10.9 | 22.6 |
|   + SpecAugment     | 9.1  | 18.8 |

### Switchboard+Fisher 2000h (WER)
| model               | SWB  | CH   |
| ------------------- | ---- | ---- |
| BPE34k LAS          | 7.8  | 13.8 |

### Librispeech (WER)
| model               | dev-clean | dev-other | test-clean | test-other |
| ------------------- | --------- | --------- | ---------- | ---------- |
| BPE30k LAS + RNNLM  | 3.4       | 10.7      | 3.4        | 11.3       |

### TEDLIUM2 (WER)
| model               | dev  | test |
| ------------------- | ---- | ---- |
| BPE10k LAS + RNNLM  | 10.9 | 11.2 |

### WSJ (WER)
| model                    | test_dev93 | test_eval92 |
| ------------------------ | ---------- | ----------- |
| BPE1k LAS + CTC + RNNLM  | 8.8        | 6.2         |

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
- https://github.com/HawkAaron/E2E-ASR

## Dependency
- https://github.com/SeanNaren/warp-ctc
- https://github.com/HawkAaron/warp-transducer

<!-- ## TODO
- WFST decoder
- Minimum WER training
- Convolutional decoder
- Speech Translation
- Tacotron2 -->
