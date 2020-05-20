#!/usr/bin/env bash

# pip install pytest
pytest ./test/test_encoder.py
pytest ./test/test_rnn_decoder.py
pytest ./test/test_transformer_decoder.py
