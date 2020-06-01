#!/usr/bin/env bash

# pip install pytest

# encoder
pytest ./test/encoders/test_conv_encoder.py
pytest ./test/encoders/test_rnn_encoder.py
pytest ./test/encoders/test_transformer_encoder.py
# decoder
pytest ./test/decoders/test_las_decoder.py
pytest ./test/decoders/test_transformer_decoder.py
pytest ./test/decoders/test_rnn_transducer_decoder.py
# LM
pytest ./test/lm/test_rnnlm.py
pytest ./test/lm/test_transformerlm.py
pytest ./test/lm/test_transformer_xl_lm.py
# modules
pytest ./test/modules/test_attention.py
pytest ./test/modules/test_gmm_attention.py
pytest ./test/modules/test_multihead_attention.py
pytest ./test/modules/test_mocha.py
pytest ./test/modules/test_pointwise_feed_forward.py
