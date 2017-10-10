#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
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
        # encoder_num_proj (int): the number of nodes in recurrent
            projection layer of the encoder
        encoder_num_layers (int): the number of layers of the encoder
        encoder_dropout (float): the probability to drop nodes

        attention_type (string):
        attention_dim (int):

        decoder_type (string): lstm or gru
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
                 decoder_dropout,
                 embedding_dim,
                 num_classes,
                 eos_index,
                 max_decode_length=100,
                 splice=1,
                 parameter_init=0.1,
                 att_softmax_temperature=1.,
                 logits_softmax_temperature=1):

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
        self.decoder_dropout = decoder_dropout
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes + 2
        # NOTE: add <SOS> and <EOS>
        self.eos_index = eos_index
        self.max_decode_length = max_decode_length
        self.att_softmax_temperature = att_softmax_temperature
        self.logits_softmax_temperature = logits_softmax_temperature

        # Common setting
        self.parameter_init = parameter_init

        ####################
        # Encoder
        ####################
        encoder = load(encoder_type=encoder_type)
        if encoder_type in ['lstm', 'gru', 'rnn']:
            self.encoder = encoder(input_size=input_size,
                                   rnn_type=encoder_type,
                                   bidirectional=encoder_bidirectional,
                                   num_units=encoder_num_units,
                                   num_layers=encoder_num_layers,
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
        # self.embedding_dropout = nn.Dropout(decoder_dropout)
        self.fc = nn.Linear(decoder_num_units, num_classes + 2)
        # NOTE: <SOS> is removed because the decoder never predict <SOS> class

        # GPU setting
        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.attend = self.attend.cuda()
            self.decoder = self.decoder.cuda()
            # TODO: Remove this??

    def forward(self, inputs, labels):
        """
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
        Returns:
            outputs (FloatTensor): A tensor of size
                `[T_out, B, num_classes (including <SOS> and <EOS>)]`
        """
        encoder_outputs, encoder_final_state = self.encoder(inputs)
        # TODO: start decoder state from encoder final state
        outputs, _ = self.decode_train(encoder_outputs, labels)
        return outputs

    def compute_loss(self, outputs, labels):
        """
        Args:
            outputs (FloatTensor):
            labels (LongTensor):
        Returns:
            loss ():
        """
        batch_size, _, num_classes = outputs.size()
        outputs = outputs.view((-1, num_classes))
        labels = labels[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(outputs, labels,
                               size_average=False)
        loss /= batch_size
        return loss

    def compute_coverage(self):
        raise NotImplementedError

    def decode_train(self, encoder_outputs, labels):
        """
        Args:
            encoder_outputs (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            labels (LongTensor): A tensor of size `[B, T_out]`
        Returns:
            outputs (FloatTensor):
            att_weights (np.ndarray):
        """
        ys = self.embedding(labels[:, :-1])
        # ys = self.embedding_dropout(ys)
        labels_max_seq_len = labels.size(1)

        outputs = []
        att_weights = []
        att_weight_vec = None
        dec_state = None if self.decoder_type == 'gru' else (None, None)
        for t in range(labels_max_seq_len - 1):
            y = ys[:, t:t + 1, :]

            dec_output, dec_state, context_vec, att_weight_vec = self.decode_step(
                encoder_outputs,
                y,
                dec_state,
                att_weight_vec)

            output = dec_output + context_vec

            att_weights.append(att_weight_vec)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        outputs = self.fc(outputs)
        # TODO: add additional MLP
        att_weights = torch.stack(att_weights, dim=1)
        att_weights = att_weights.cpu().data.numpy()

        return outputs, att_weights

    def decode_step(self, encoder_outputs, y, dec_state, att_weight_vec):
        """
        Args:
            encoder_outputs (torch.FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            y (FloatTensor): A tensor of size `[B, 1, embedding_dim]`
            dec_state (FloatTensor): A tensor of size
                `[1, B, decoder_num_units]`
            att_weight_vec (FloatTensor): A tensor of size `[B, T_in]`
        Returns:
            dec_output (FloatTensor): A tensor of size
                `[B, 1, decoder_num_units]`
            dec_state (FloatTensor): A tensor of size
                `[1, B, decoder_num_units]`
            content_vector (FloatTensor): A tensor of size
                `[B, 1, encoder_num_units]`
            att_weight_vec (FloatTensor): A tensor of size `[B, T_in]`
        """
        if self.decoder_type == 'lstm':
            dec_output, dec_state = self.decoder(
                y, hx=dec_state)
            # TODO; fix bug

        elif self.decoder_type == 'gru':
            dec_output, dec_state = self.decoder(
                y, hx=dec_state)

        # dec_output: `[B, 1, decoder_num_units]`
        context_vec, att_weight_vec = self.attend(encoder_outputs,
                                                  dec_output,
                                                  att_weight_vec)

        return dec_output, dec_state, context_vec, att_weight_vec

    def decode_infer(self, inputs, labels, beam_width=1):
        """
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            labels (LongTensor): A tensor of size `[B, T_out]`
            beam_width (int, optional):
        Returns:

        """
        if beam_width == 1:
            return self._decode_infer_greedy(inputs, labels)
        else:
            return self._decode_infer_beam_search(inputs, labels, beam_width)

    def _decode_infer_greedy(self, inputs, labels):
        """
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            labels (LongTensor): A tensor of size `[B, T_out]`
        Returns:
            outputs (np.ndarray):
            att_weights (np.ndarray):
        """
        encoder_outputs, encoder_final_state = self.encoder(inputs)

        # Start from <SOS>
        y = labels[:, 0:1]

        outputs = []
        att_weights = []
        att_weight_vec = None
        dec_state = None if self.decoder_type == 'gru' else (None, None)
        for _ in range(self.max_decode_length):

            y = self.embedding(y)
            # y = self.embedding_dropout(y)

            dec_output, dec_state, context_vec, att_weight_vec = self.decode_step(
                encoder_outputs,
                y,
                dec_state,
                att_weight_vec)

            output = dec_output + context_vec
            output = self.fc(output.squeeze(dim=1))
            # TODO: add additional MLP

            # Pick up 1-best
            y = torch.max(output, dim=1)[1]
            y = y.unsqueeze(dim=1)
            output = y.cpu().data.numpy()

            outputs.append(output)
            att_weights.append(att_weight_vec.cpu().data.numpy())

            # Break if <EOS> is outputed
            if np.all(output == self.eos_index):
                break

        outputs = np.concatenate(outputs, axis=1)
        return outputs, att_weights

    def _decode_infer_beam_search(self, inputs, labels, beam_width):
        """
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            labels (LongTensor): A tensor of size `[B, T_out]`
        Returns:

        """
        encoder_outputs, encoder_final_state = self.encoder(inputs)

        # Start from <SOS>
        y = labels[:, 0:1]

        outputs = []
        att_weights = []
        att_weight_vec = None
        dec_state = None if self.decoder_type == 'gru' else (None, None)

        raise NotImplementedError
