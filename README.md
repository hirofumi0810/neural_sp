[![Build Status](https://travis-ci.org/hirofumi0810/neural_sp.svg?branch=master)](https://travis-ci.org/hirofumi0810/neural_sp)

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

## Key features
### Corpus
  - ASR
    - AISHELL-1
    - CSJ
    - Librispeech
    - Switchboard (+ Fisher)
    - TEDLIUM2/TEDLIUM3
    - TIMIT
    - WSJ

  - LM
    - Penn Tree Bank
    - WikiText2

### Front-end
  - Frame stacking
  - Sequence summary network [[link](https://www.isca-speech.org/archive/Interspeech_2018/abstracts/1438.html)]
  - SpecAugment [[link](https://arxiv.org/abs/1904.08779)]
  - Adaptive SpecAugment [[link](https://arxiv.org/abs/1912.05533)]

### Encoder
  - RNN encoder
    - (CNN-)BLSTM, (CNN-)LSTM, (CNN-)BLGRU, (CNN-)LGRU
    - Latency-controlled BLSTM [[link](https://arxiv.org/abs/1510.08983)]
  - Transformer encoder [[link](https://arxiv.org/abs/1706.03762)]
    - (CNN-)Transformer
    - Chunk hopping mechanism [[link](https://arxiv.org/abs/1902.06450)]
    - Relative positional encoding [[link](https://arxiv.org/abs/1901.02860)]
  - Time-depth separable (TDS) convolution encoder [[link](https://arxiv.org/abs/1904.02619)] [[line](https://arxiv.org/abs/2001.09727)]
  - Gated CNN encoder (GLU) [[link](https://openreview.net/forum?id=Hyig0zb0Z)]
  - Conformer encoder [[link](https://arxiv.org/abs/2005.08100)]

### Connectionist Temporal Classification (CTC) decoder
  - Forced alignment
  - Beam search
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
    - CTC-synchronous training (CTC-ST) [[link](https://arxiv.org/abs/2005.04712)]
  - RNN transducer [[link](https://arxiv.org/abs/1211.3711)]
  - Transformer decoder [[link](https://arxiv.org/abs/1706.03762)]
  - Streaming Transformer decoder
    - Monotonic Multihead Attention [[link](https://arxiv.org/abs/1909.12406)] [[link](https://arxiv.org/abs/2005.09394)]

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
  - LM objective


## ASR Performance
### AISHELL-1 (CER)
| model         | dev | test |
| -----------   | --- | ---- |
| Transformer   | 5.0 | 5.4  |
| Streaming MMA | 6.1 | 6.6  |

### CSJ (WER)
| model | eval1 | eval2 | eval3 |
| ----- | ----- | ----- | ----- |
| LAS   | 6.5   | 5.1   | 5.6   |

### Switchboard 300h (WER)
| model | SWB  | CH   |
| ----- | ---- | ---- |
| LAS   | 9.1  | 18.8 |

### Switchboard+Fisher 2000h (WER)
| model | SWB  | CH   |
| ----- | ---- | ---- |
| LAS   | 7.8  | 13.8 |

### Librispeech (WER)
| model         | dev-clean | dev-other | test-clean | test-other |
| -----------   | --------- | --------- | ---------- | ---------- |
| Transformer   | 2.2       | 5.4       | 2.4        | 5.7        |
| Streaming MMA | 2.5       | 6.9       | 2.7        | 7.1        |

### TEDLIUM2 (WER)
| model | dev  | test |
| ----- | ---- | ---- |
| LAS   | 10.9 | 11.2 |

### WSJ (WER)
| model | test_dev93 | test_eval92 |
| ----- | ---------- | ----------- |
| LAS   | 8.8        | 6.2         |

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
- https://github.com/1ytic/warp-rnnt

<!-- ## TODO
- WFST decoder
- Minimum WER training
- Convolutional decoder
- Speech Translation
- Tacotron2 -->
