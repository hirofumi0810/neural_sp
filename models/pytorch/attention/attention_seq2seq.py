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
        decoder_dropout (float, optional):
        embedding_dim (int):
        # embedding_dropout (int):
        num_classes (int): the number of classes of target labels
            (except for a blank label)


        max_decode_length (int):
        splice (int, optional): frames to splice. Default is 1 frame.
        parameter_init (float, optional): Range of uniform distribution to
            initialize weight parameters
        downsample_list (list, optional):
        init_dec_state_with_enc_state (bool, optional):
        sharpening_factor (float):
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
                 decoder_num_proj,
                 #   decdoder_num_layers,
                 decoder_dropout,
                 embedding_dim,
                 #  embedding_dropout,
                 num_classes,
                 eos_index,
                 max_decode_length=100,
                 splice=1,
                 parameter_init=0.1,
                 downsample_list=[],
                 init_dec_state_with_enc_state=True,
                 sharpening_factor=1.,
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
        self.downsample_list = downsample_list
        self.encoder_dropout = encoder_dropout

        # Setting for the decoder
        self.attention_type = attention_type
        self.attention_dim = attention_dim
        self.decoder_type = decoder_type
        self.decoder_num_units = decoder_num_units
        self.decoder_num_proj = decoder_num_proj
        # self.decdoder_num_layers = decdoder_num_layers
        self.decoder_dropout = decoder_dropout
        self.embedding_dim = embedding_dim
        # self.embedding_dropout = embedding_dropout
        self.num_classes = num_classes + 2
        # NOTE: add <SOS> and <EOS>
        self.eos_index = eos_index
        self.max_decode_length = max_decode_length
        self.init_dec_state_with_enc_state = init_dec_state_with_enc_state
        self.sharpening_factor = sharpening_factor
        self.logits_softmax_temperature = logits_softmax_temperature

        # Common setting
        self.parameter_init = parameter_init

        ####################
        # Encoder
        ####################
        if len(downsample_list) == 0:
            encoder = load(encoder_type=encoder_type)
        else:
            encoder = load(encoder_type='p' + encoder_type)
        if encoder_type in ['lstm', 'gru', 'rnn']:
            if len(downsample_list) == 0:
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
                self.encoder = encoder(input_size=input_size,
                                       rnn_type=encoder_type,
                                       bidirectional=encoder_bidirectional,
                                       num_units=encoder_num_units,
                                       num_layers=encoder_num_layers,
                                       downsample_list=downsample_list,
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
            sharpening_factor=sharpening_factor)

        self.embedding = nn.Embedding(num_classes + 2, embedding_dim)
        # self.embedding_dropout = nn.Dropout(decoder_dropout)
        self.dec_proj_dec_state = nn.Linear(
            decoder_num_units, decoder_num_proj)
        self.dec_proj_context = nn.Linear(
            encoder_num_units * self.encoder_num_directions, decoder_num_proj)
        self.fc = nn.Linear(decoder_num_proj, num_classes + 2)
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
            att_weights (FloatTensor): A tensor of size `[B, T_out, T_in]`
        """
        encoder_states, encoder_final_state = self.encoder(inputs)
        # TODO: start decoder state from encoder final state
        outputs, att_weights = self.decode_train(
            encoder_states, labels, encoder_final_state)
        return outputs, att_weights

    def compute_loss(self, outputs, labels,
                     att_weights=None, coverage_weight=0):
        """
        Args:
            outputs (FloatTensor):
            labels (LongTensor):
            att_weights (FloatTensor): A tensor of size `[B, T_out, T_in]`
            coverage_weight (float, optional):
        Returns:
            loss ():
        """
        batch_size, _, num_classes = outputs.size()
        outputs = outputs.view((-1, num_classes))
        labels = labels[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(outputs, labels,
                               size_average=False)
        loss /= batch_size

        if coverage_weight != 0:
            pass
            # coverage = self.compute_coverage(att_weights)
            # loss += coverage_weight * coverage
            # raise NotImplementedError

        return loss

    def compute_coverage(self, att_weights):
        batch_size, max_time_outputs, max_time_inputs = att_weights.size()
        raise NotImplementedError

    def decode_train(self, encoder_states, labels, encoder_final_state):
        """Decoding when training.
        Args:
            encoder_states (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            labels (LongTensor): A tensor of size `[B, T_out]`
            encoder_final_state (FloatTensor): A tensor of size
                `[]`
        Returns:
            outputs (FloatTensor): A tensor of size `[B,]`
            att_weights (FloatTensor): A tensor of size `[B, T_out, T_in]`
        """
        ys = self.embedding(labels[:, :-1])
        # ys = self.embedding_dropout(ys)
        labels_max_seq_len = labels.size(1)

        outputs = []
        att_weights = []
        att_weight_vec = None

        if self.init_dec_state_with_enc_state:
            # Initialize decoder state with the final state of the top layer of
            # the encoder
            if self.encoder_bidirectional:
                final_enc_state_fw = encoder_final_state[-2]
                final_enc_state_bw = encoder_final_state[-1]
                dec_state = torch.cat(
                    (final_enc_state_fw, final_enc_state_bw), dim=1).unsqueeze(0)
            else:
                dec_state = encoder_final_state[-1].unsqueeze(0)
        else:
            dec_state = None if self.decoder_type == 'gru' else (None, None)

        for t in range(labels_max_seq_len - 1):
            y = ys[:, t:t + 1, :]

            dec_state, dec_state, context_vec, att_weight_vec = self.decode_step(
                encoder_states,
                y,
                dec_state,
                att_weight_vec)

            # Map to the projection layer
            output = self.dec_proj_dec_state(
                dec_state) + self.dec_proj_context(context_vec)

            att_weights.append(att_weight_vec)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        outputs = self.fc(F.tanh(outputs))
        att_weights = torch.stack(att_weights, dim=1)
        # NOTE; att_weights in the training stage may be used for computing the
        # coverage, so do not convert to numpy yet.

        return outputs, att_weights

    def decode_step(self, encoder_states, y, dec_state, att_weight_vec):
        """
        Args:
            encoder_states (torch.FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            y (FloatTensor): A tensor of size `[B, 1, embedding_dim]`
            dec_state (FloatTensor): A tensor of size
                `[1, B, decoder_num_units]`
            att_weight_vec (FloatTensor): A tensor of size `[B, T_in]`
        Returns:
            dec_state (FloatTensor): A tensor of size
                `[B, 1, decoder_num_units]`
            dec_state (FloatTensor): A tensor of size
                `[1, B, decoder_num_units]`
            content_vector (FloatTensor): A tensor of size
                `[B, 1, encoder_num_units]`
            att_weight_vec (FloatTensor): A tensor of size `[B, T_in]`
        """
        if self.decoder_type == 'lstm':
            dec_state, dec_state = self.decoder(
                y, hx=dec_state)
            # TODO; fix bug

        elif self.decoder_type == 'gru':
            dec_state, dec_state = self.decoder(
                y, hx=dec_state)

        # dec_state: `[B, 1, decoder_num_units]`
        context_vec, att_weight_vec = self.attend(encoder_states,
                                                  dec_state,
                                                  att_weight_vec)

        return dec_state, dec_state, context_vec, att_weight_vec

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
        """Greedy decoding when inference.
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            labels (LongTensor): A tensor of size `[B, T_out]`
        Returns:
            outputs (np.ndarray): A tensor of size `[]`
            att_weights (np.ndarray): A tensor of size `[]`
        """
        encoder_states, encoder_final_state = self.encoder(inputs)

        # Start from <SOS>
        y = labels[:, 0:1]

        outputs = []
        att_weights = []
        att_weight_vec = None

        if self.init_dec_state_with_enc_state:
            # Initialize decoder state with the final state of the top layer of
            # the encoder
            if self.encoder_bidirectional:
                final_enc_state_fw = encoder_final_state[-2]
                final_enc_state_bw = encoder_final_state[-1]
                dec_state = torch.cat(
                    (final_enc_state_fw, final_enc_state_bw), dim=1).unsqueeze(0)
            else:
                dec_state = encoder_final_state[-1].unsqueeze(0)
        else:
            dec_state = None if self.decoder_type == 'gru' else (None, None)

        for _ in range(self.max_decode_length):

            y = self.embedding(y)
            # y = self.embedding_dropout(y)

            dec_state, dec_state, context_vec, att_weight_vec = self.decode_step(
                encoder_states,
                y,
                dec_state,
                att_weight_vec)

            # Map to the projection layer
            output = self.dec_proj_dec_state(
                dec_state) + self.dec_proj_context(context_vec)

            # Map to the outpu layer
            output = self.fc(F.tanh(output.squeeze(dim=1)))

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
        """Beam search decoding when inference.
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            labels (LongTensor): A tensor of size `[B, T_out]`
        Returns:

        """
        encoder_states, encoder_final_state = self.encoder(inputs)

        # Start from <SOS>
        y = labels[:, 0:1]

        outputs = []
        att_weights = []
        att_weight_vec = None

        if self.init_dec_state_with_enc_state:
            # Initialize decoder state with the final state of the top layer of
            # the encoder
            if self.encoder_bidirectional:
                final_enc_state_fw = encoder_final_state[-2]
                final_enc_state_bw = encoder_final_state[-1]
                dec_state = torch.cat(
                    (final_enc_state_fw, final_enc_state_bw), dim=1).unsqueeze(0)
            else:
                dec_state = encoder_final_state[-1].unsqueeze(0)
        else:
            dec_state = None if self.decoder_type == 'gru' else (None, None)

        raise NotImplementedError
