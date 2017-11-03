#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as

from models.pytorch.base import ModelBase
from models.pytorch.encoders.load_encoder import load
from models.pytorch.attention.attention_layer import AttentionMechanism


class MultitaskAttentionSeq2seq(object):

    def __init__(self,
                 input_size,
                 encoder_type,
                 encoder_bidirectional,
                 encoder_num_units,
                 #  encoder_num_proj,
                 encoder_num_layers,
                 encoder_num_layers_sub,
                 encoder_dropout,
                 attention_type,
                 attention_type_sub,
                 attention_dim,
                 attention_dim_sub,
                 decoder_type,
                 decoder_type_sub,
                 decoder_num_units,
                 decoder_num_units_sub,
                 decoder_num_proj,
                 decoder_num_proj_sub,
                 #   decdoder_num_layers,
                 #   decdoder_num_layers_sub,
                 decoder_dropout,
                 decoder_dropout_sub,
                 embedding_dim,
                 embedding_dim_sub,
                 #  embedding_dropout,
                 #  embedding_dropout_sub,
                 num_classes,
                 num_classes_sub,
                 eos_index,
                 eos_index_sub,
                 max_decode_length=100,
                 max_decode_length_sub=100,
                 splice=1,
                 parameter_init=0.1,
                 downsample_list=[],
                 init_dec_state_with_enc_state=True,
                 sharpening_factor=1.,
                 sharpening_factor_sub=1.,
                 logits_temperature=1,
                 logits_temperature_sub=1):

        super(AttentionSeq2seq, self).__init__(
            encoder_type=encoder_type,
            encoder_bidirectional=encoder_bidirectional,
            encoder_num_units=encoder_num_units,
            #  encoder_num_proj=encoder_num_proj,
            encoder_num_layers=encoder_num_layers,
            encoder_dropout=encoder_dropout,
            attention_type=attention_type,
            attention_dim=attention_dim,
            decoder_type=decoder_type,
            decoder_num_units=decoder_num_units,
            decoder_num_proj=decoder_num_units,
            #   decdoder_num_layers=decdoder_num_layers,
            decoder_dropout=decoder_dropout,
            embedding_dim=embedding_dim,
            #  embedding_dropout=embedding_dropout,
            num_classes=num_classes,
            eos_index=eos_index,
            max_decode_length=max_decode_length,
            splice=splice,
            parameter_init=parameter_init,
            downsample_list=downsample_list,
            init_dec_state_with_enc_state=init_dec_state_with_enc_state,
            sharpening_factor=sharpening_factor,
            logits_temperature=logits_temperature)

        # Setting for the encoder
        self.encoder_num_layers_sub = encoder_num_layers_sub

        # Setting for the decoder
        self.attention_type_sub = attention_type_sub
        self.attention_dim_sub = attention_dim_sub
        self.decoder_type_sub = decoder_type_sub
        self.decoder_num_units_sub = decoder_num_units_sub
        self.decoder_num_proj_sub = decoder_num_proj_sub
        # self.decdoder_num_layers_sub = decdoder_num_layers_sub
        self.decoder_dropout_sub = decoder_dropout_sub
        self.embedding_dim_sub = embedding_dim_sub
        # self.embedding_dropout_sub = embedding_dropout_sub
        self.num_classes_sub = num_classes_sub + 2
        # NOTE: add <SOS> and <EOS>
        self.eos_index_sub = eos_index_sub
        self.max_decode_length_sub = max_decode_length_sub
        self.init_dec_state_with_enc_state_sub = init_dec_state_with_enc_state_sub
        self.sharpening_factor_sub = sharpening_factor_sub
        self.logits_temperature_sub = logits_temperature_sub

    def forward(self, inputs, labels, labels_sub):
        """
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            labels ():
            labels_sub ():
        Returns:
            outputs (FloatTensor): A tensor of size
                `[T_out, B, num_classes (including <SOS> and <EOS>)]`
            att_weights (FloatTensor): A tensor of size `[B, T_out, T_in]`
            outputs_sub (FloatTensor):
            att_weights_sub (FloatTensor):
        """
        encoder_states, encoder_final_state, encoder_states_sub, encoder_final_state_sub = self.encoder(
            inputs)

        # main task
        outputs, att_weights = self.decode_train(
            encoder_states, labels, encoder_final_state)

        # sub task
        outputs_sub, att_weights_sub = self.decode_train(
            encoder_states_sub, labels_sub, encoder_final_state_sub)

        return outputs, att_weights, outputs_sub, att_weights_sub
