#!/usr/bin/env bash

# pip install pytest
pytest ./test/test_encoder.py
pytest ./test/test_las_decoder.py
pytest ./test/test_transformer_decoder.py
pytest ./test/test_rnn_transducer_decoder.py
pytest ./test/test_rnnlm.py
