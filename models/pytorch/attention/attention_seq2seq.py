#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pytorch.base import ModelBase
from models.pytorch.encoders.load_encoder import load
from models.pytorch.attention.attention_layer import AttentionMechanism


class AttentionSeq2seq(ModelBase):
    """The Attention-besed model.
    Args:
        input_size (int): the dimension of input features
        encoder_type (string): the type of the encoder. Set lstm or gru or rnn.
        encoder_bidirectional (bool): if True, create a bidirectional encoder
        encoder_num_units (int): the number of units in each layer
        encoder_num_proj (int, optional): the number of nodes in recurrent
            projection layer of the encoder
        encoder_num_layers (int): the number of layers of the encoder
        encoder_dropout (float, optional): the probability to drop nodes

        attention_type (string):
        attention_dim (int):

        decoder_type (string): lstm or gru or rnn
        decoder_num_units (int):
        # decdoder_num_layers (int):
        embedding_dim (int):
        num_classes (int): the number of classes of target labels
            (except for a blank label)
        decoder_dropout (float, optional):
        max_decode_length (int):

        splice (int, optional): frames to splice. Default is 1 frame.
        parameter_init (float, optional): Range of uniform distribution to
            initialize weight parameters
        att_softmax_temperature (float):
        logits_softmax_temperature (float):
        clip_grad (float, optional): Range of gradient clipping (> 0)
    """

    def __init__(self,
                 input_size,
                 encoder_type,
                 encoder_bidirectional,
                 encoder_num_units,
                 #  encoder_num_proj,
                 encoder_num_layers,
                 encoder_dropout,
                 attention_type,
                 attention_dim,
                 decoder_type,
                 decoder_num_units,
                 #   decdoder_num_layers,
                 embedding_dim,
                 num_classes,
                 decoder_dropout,
                 max_decode_length=100,
                 splice=1,
                 parameter_init=0.1,
                 att_softmax_temperature=1.,
                 logits_softmax_temperature=1,
                 clip_grad=None):

        super(ModelBase, self).__init__()

        # Setting for the encoder
        self.input_size = input_size
        self.splice = splice
        self.encoder_type = encoder_type
        self.encoder_bidirectional = encoder_bidirectional
        self.encoder_num_directions = 2 if encoder_bidirectional else 1
        self.encoder_num_units = encoder_num_units
        # self.encoder_num_proj = encoder_num_proj
        self.encoder_num_layers = encoder_num_layers
        self.encoder_dropout = encoder_dropout

        # Setting for the decoder
        self.attention_type = attention_type
        self.attention_dim = attention_dim
        self.decoder_type = decoder_type
        self.decoder_num_units = decoder_num_units
        # self.decdoder_num_layers = decdoder_num_layers
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes + 2
        # NOTE: add <SOS> and <EOS>
        self.decoder_dropout = decoder_dropout
        self.max_decode_length = max_decode_length
        self.att_softmax_temperature = att_softmax_temperature
        self.logits_softmax_temperature = logits_softmax_temperature

        # Common setting
        self.parameter_init = parameter_init
        self.clip_grad = clip_grad

        ####################
        # Encoder
        ####################
        encoder = load(encoder_type=encoder_type)
        if encoder_type in ['lstm', 'gru', 'rnn']:
            self.encoder = encoder(input_size=input_size,
                                   num_units=encoder_num_units,
                                   num_layers=encoder_num_layers,
                                   rnn_type=encoder_type,
                                   bidirectional=encoder_bidirectional,
                                   dropout=encoder_dropout,
                                   parameter_init=parameter_init,
                                   use_cuda=self.use_cuda,
                                   batch_first=True)
        else:
            raise NotImplementedError

        ####################
        # Decoder
        ####################
        if decoder_type == 'lstm':
            self.decoder = nn.LSTM(embedding_dim,
                                   hidden_size=decoder_num_units,
                                   num_layers=1,
                                   bias=True,
                                   batch_first=True,
                                   dropout=decoder_dropout,
                                   bidirectional=False)
        elif decoder_type == 'gru':
            self.decoder = nn.GRU(embedding_dim,
                                  hidden_size=decoder_num_units,
                                  num_layers=1,
                                  bias=True,
                                  batch_first=True,
                                  dropout=decoder_dropout,
                                  bidirectional=False)
        else:
            raise TypeError
        # NOTE: decoder is unidirectional and only 1 layer now

        ##############################
        # Attention layer
        ##############################
        self.attend = AttentionMechanism(
            encoder_num_units=encoder_num_units * self.encoder_num_directions,
            decoder_num_units=decoder_num_units,
            attention_type=attention_type,
            attention_dim=attention_dim,
            att_softmax_temperature=att_softmax_temperature)

        self.embedding = nn.Embedding(num_classes + 2, embedding_dim)
        self.fc = nn.Linear(decoder_num_units, num_classes + 2)
        # NOTE: <SOS> is removed because the decoder never predict <SOS> class

    def forward(self, inputs, labels):
        """
        Args:
            inputs (Variable): A tensor of size `[B, T, input_size]`
        Returns:
            outputs (Variable): A tensor of size `[T, B, num_classes + 1]`
        """
        encoder_outputs, final_state = self.encoder(inputs)
        outputs, _ = self.decode_train(encoder_outputs, labels)
        return outputs

    def compute_loss(self, outputs, labels):
        """
        Args:
            outputs ():
            labels (torch.LongTensor):
        Returns:
            loss ():
        """
        batch_size, _, num_classes = outputs.size()
        outputs = outputs.view((-1, num_classes))
        labels = labels[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(outputs, labels,
                               size_average=False)
        loss = loss / batch_size
        return loss

    def compute_coverage(self):
        raise NotImplementedError

    def decode_train(self, encoder_outputs, labels):
        """
        Args:
            encoder_outputs (torch.FloatTensor): A tensor of size
                `[B, T (inputs), encoder_num_units]`
            labels (torch.LongTensor): A tensor of size
                `[B, T (labels)]`
        Returns:
            outputs ():
            attention_weights ():
        """
        decoder_inputs = self.embedding(labels[:, :-1])
        labels_max_seq_len = labels.size(1)

        outputs = []
        attention_weights = []
        attention_weights_step = None
        decoder_state_step = None if self.decoder_type == 'gru' else (
            None, None)
        for t in range(labels_max_seq_len - 1):
            decoder_inputs_step = decoder_inputs[:, t:t + 1, :]
            if self.decoder_type == 'lstm':
                decoder_outputs_step, decoder_state_step = self.decoder(
                    decoder_inputs_step,
                    hx=decoder_state_step)
                # TODO; fix bug
            elif self.decoder_type == 'gru':
                decoder_outputs_step, decoder_state_step = self.decoder(
                    decoder_inputs_step,
                    hx=decoder_state_step)

            # decoder_outputs_step: `[B, 1, decoder_num_units]`
            content_vector, attention_weights_step = self.attend(encoder_outputs,
                                                                 decoder_outputs_step,
                                                                 attention_weights_step)
            attention_weights.append(attention_weights_step)
            outputs.append(decoder_outputs_step + content_vector)
            # TODO: add additional MLP

        outputs = torch.cat(outputs, dim=1)
        outputs = self.fc(outputs)

        attention_weights = torch.stack(attention_weights, dim=1)

        return outputs, attention_weights
        # NOTE: return attention_weights for visualization

    def decoder_infer(self):
        # if state is None:
        #     hx, ax = None, None
        # else:
        #     hx, ax = state
        #
        # ix = self.embedding(y)
        # ox, hx = self.dec_rnn(ix, hx=hx)
        # sx, ax = self.attend(x, ox, ax=ax)
        # out = ox + sx
        # out = self.fc(out.squeeze(dim=1))
        # return out, (hx, ax)

        raise NotImplementedError

    def decoder_infer_step(self):
        raise NotImplementedError
