#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention sequence-to-sequence model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.pytorch.base import ModelBase
from models.pytorch.encoders.load_encoder import load
from models.pytorch.attention.attention_layer import AttentionMechanism


class AttentionSeq2seq(ModelBase):
    """The Attention-besed model.
    Args:
        input_size (int): the dimension of input features
        encoder_type (string): the type of the encoder. Set lstm or gru or rnn.
        encoder_bidirectional (bool): if True, create a bidirectional encoder
        encoder_num_units (int): the number of units in each layer of the
            encoder
        encoder_num_proj (int): the number of nodes in the projection layer of
            the encoder.
        encoder_num_layers (int): the number of layers of the encoder
        encoder_dropout (float): the probability to drop nodes of the encoder
        attention_type (string): the type of attention
        attention_dim: (int) the dimension of the attention layer
        decoder_type (string): lstm or gru
        decoder_num_units (int): the number of units in each layer of the
            decoder
        decoder_num_proj (int): the number of nodes in the projection layer of
            the decoder.
        decoder_num_layers (int): the number of layers of the decoder
        decoder_dropout (float): the probability to drop nodes of the decoder
        embedding_dim (int): the dimension of the embedding in target spaces
        embedding_dropout (int): the probability to drop nodes of the
            embedding layer
        num_classes (int): the number of nodes in softmax layer
        sos_index (int): index of the start of sentence tag (<SOS>)
        eos_index (int): index of the end of sentence tag (<EOS>)
        num_stack (int, optional): the number of frames to stack
        max_decode_length (int): the length of output sequences to stop
            prediction when EOS token have not been emitted
        splice (int, optional): the number of frames to splice. This is used
            when using CNN-like encoder. Default is 1 frame.
        parameter_init (float, optional): the range of uniform distribution to
            initialize weight parameters (>= 0)
        downsample_list (list, optional):
        init_dec_state_with_enc_state (bool, optional):
        sharpening_factor (float, optional): a sharpening factor in the
            softmax layer for computing attention weights
        logits_temperature (float, optional): a parameter for smoothing the
            softmax layer in outputing probabilities
        sigmoid_smoothing (bool, optional): if True, replace softmax function
            in computing attention weights with sigmoid function for smoothing
        input_feeding_approach (bool, optional): if True,
    """

    def __init__(self,
                 input_size,
                 encoder_type,
                 encoder_bidirectional,
                 encoder_num_units,
                 encoder_num_proj,
                 encoder_num_layers,
                 encoder_dropout,
                 attention_type,
                 attention_dim,
                 decoder_type,
                 decoder_num_units,
                 decoder_num_proj,
                 decdoder_num_layers,
                 decoder_dropout,
                 embedding_dim,
                 embedding_dropout,
                 num_classes,
                 sos_index,
                 eos_index,
                 num_stack=1,
                 max_decode_length=100,
                 splice=1,
                 parameter_init=0.1,
                 downsample_list=[],
                 init_dec_state_with_enc_state=True,
                 sharpening_factor=1,
                 logits_temperature=1,
                 sigmoid_smoothing=False,
                 input_feeding_approach=False):

        super(ModelBase, self).__init__()

        # TODO:
        # clip_activation
        # time_major

        assert input_size % 3 == 0, 'input_size must be divisible by 3 (+ delta, double delta features).'
        # NOTE: input features are expected to including Δ and ΔΔ features
        assert splice % 2 == 1, 'splice must be the odd number'

        # Setting for the encoder
        self.input_size = input_size
        self.num_stack = num_stack
        self.splice = splice
        self.encoder_type = encoder_type
        self.encoder_bidirectional = encoder_bidirectional
        self.encoder_num_directions = 2 if encoder_bidirectional else 1
        self.encoder_num_units = encoder_num_units
        self.encoder_num_proj = encoder_num_proj
        self.encoder_num_layers = encoder_num_layers
        self.downsample_list = downsample_list
        self.encoder_dropout = encoder_dropout

        # Setting for the attention decoder
        self.attention_type = attention_type
        self.attention_dim = attention_dim
        self.decoder_type = decoder_type
        self.decoder_num_units = decoder_num_units
        self.decoder_num_proj = decoder_num_proj
        self.decdoder_num_layers = decdoder_num_layers
        self.decoder_dropout = decoder_dropout
        self.embedding_dim = embedding_dim
        self.embedding_dropout = embedding_dropout
        self.num_classes = num_classes + 2
        # NOTE: add <SOS> and <EOS>
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.max_decode_length = max_decode_length
        self.init_dec_state_with_enc_state = init_dec_state_with_enc_state
        self.sharpening_factor = sharpening_factor
        self.logits_temperature = logits_temperature
        self.sigmoid_smoothing = sigmoid_smoothing
        self.input_feeding_approach = input_feeding_approach

        # Common setting
        self.parameter_init = parameter_init
        self.name = 'pytorch_attention_seq2seq'

        ####################
        # Encoder
        ####################
        # Load an instance
        if len(downsample_list) == 0:
            encoder = load(encoder_type=encoder_type)
        else:
            encoder = load(encoder_type='p' + encoder_type)

        # Call the encoder function
        if encoder_type in ['lstm', 'gru', 'rnn']:
            if len(downsample_list) == 0:
                self.encoder = encoder(
                    input_size=input_size,
                    rnn_type=encoder_type,
                    bidirectional=encoder_bidirectional,
                    num_units=encoder_num_units,
                    num_proj=encoder_num_proj,
                    num_layers=encoder_num_layers,
                    dropout=encoder_dropout,
                    parameter_init=parameter_init,
                    use_cuda=self.use_cuda,
                    batch_first=True)
            else:
                # Pyramidal encoder
                self.encoder = encoder(
                    input_size=input_size,
                    rnn_type=encoder_type,
                    bidirectional=encoder_bidirectional,
                    num_units=encoder_num_units,
                    num_proj=encoder_num_proj,
                    num_layers=encoder_num_layers,
                    dropout=encoder_dropout,
                    parameter_init=parameter_init,
                    downsample_list=downsample_list,
                    downsample_type='drop',
                    use_cuda=self.use_cuda,
                    batch_first=True)
        else:
            raise NotImplementedError

        ####################
        # Decoder
        ####################
        if decoder_type == 'lstm':
            self.decoder = nn.LSTM(
                embedding_dim,
                hidden_size=decoder_num_units,
                num_layers=1,
                bias=True,
                batch_first=True,
                dropout=decoder_dropout,
                bidirectional=False)
        elif decoder_type == 'gru':
            self.decoder = nn.GRU(
                embedding_dim,
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
            encoder_num_units=encoder_num_units,
            decoder_num_units=decoder_num_units,
            attention_type=attention_type,
            attention_dim=attention_dim,
            sharpening_factor=sharpening_factor)

        ##################################################
        # Bridge layer between the encoder and decoder
        ##################################################
        if encoder_num_units != decoder_num_units:
            self.bridge = nn.Linear(
                encoder_num_units, decoder_num_units)
        else:
            self.bridge = None

        self.embedding = nn.Embedding(self.num_classes, embedding_dim)
        self.embedding_dropout = nn.Dropout(decoder_dropout)

        if input_feeding_approach:
            self.decoder_proj_layer = nn.Linear(
                decoder_num_units * 2, decoder_num_proj)
            # NOTE: input-feeding approach
            self.fc = nn.Linear(decoder_num_proj, self.num_classes)
        else:
            self.fc = nn.Linear(decoder_num_units, self.num_classes)
        # NOTE: <SOS> is removed because the decoder never predict <SOS> class
        # TODO: self.num_classes - 1

    def forward(self, inputs, labels, volatile=False):
        """
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            volatile (bool, optional):
        Returns:
            outputs (FloatTensor): A tensor of size
                `[T_out, B, num_classes (including <SOS> and <EOS>)]`
            attention_weights (FloatTensor): A tensor of size `[B, T_out, T_in]`
        """
        encoder_outputs, encoder_final_state = self._encode(inputs, volatile)

        outputs, attention_weights = self.decode_train(
            encoder_outputs, labels, encoder_final_state)
        return outputs, attention_weights

    def _encode(self, inputs, volatile):
        """
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            volatile (bool):
        Returns:
            encoder_outputs (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            encoder_final_state (FloatTensor): A tensor of size
                `[1, B, encoder_num_units]`
        """
        encoder_outputs, encoder_final_state = self.encoder(inputs, volatile)
        # NOTE: encoder_outputs:
        # `[B, T_in, encoder_num_units * encoder_num_directions]`
        # encoder_final_state:
        # `[encoder_num_layers * encoder_num_directions, B, encoder_num_units]`

        # Sum bidirectional outputs
        if self.encoder_bidirectional:
            encoder_outputs = encoder_outputs[:, :, :self.encoder_num_units] + \
                encoder_outputs[:, :, self.encoder_num_units:]
            # NOTE: encoder_outputs: `[B, T_in, encoder_num_units]`

            # Pick up the final state of the top layer of the encoder (forward)
            encoder_final_state = encoder_final_state[-2:-1, :, :]
            # NOTE: encoder_final_state: `[1, B, encoder_num_units]`
            # TODO: check source code
        else:
            encoder_final_state = encoder_final_state[-1, :, :].unsqueeze(0)

        if self.encoder_num_units != self.decoder_num_units:
            # Bridge between the encoder and decoder
            encoder_outputs = self.bridge(encoder_outputs)
            encoder_final_state = encoder_final_state
            encoder_final_state = self.bridge(
                encoder_final_state.transpose(0, 1)).transpose(0, 1)

        return encoder_outputs, encoder_final_state

    def compute_loss(self, outputs, labels,
                     attention_weights=None, coverage_weight=0):
        """
        Args:
            outputs (FloatTensor): A tensor of size `[B, ]`
            labels (LongTensor): A tensor of size `[B, ]`
            attention_weights (FloatTensor): A tensor of size
                `[B, T_out, T_in]`
            coverage_weight (float, optional):
        Returns:
            loss (FloatTensor): A tensor of size `[]`
        """
        batch_size, _, num_classes = outputs.size()
        outputs = outputs.view((-1, num_classes))
        labels = labels[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(outputs, labels,
                               size_average=False)
        loss /= batch_size

        if coverage_weight != 0:
            pass
            # coverage = self._compute_coverage(attention_weights)
            # loss += coverage_weight * coverage

        return loss

    def _compute_coverage(self, attention_weights):
        batch_size, max_time_outputs, max_time_inputs = attention_weights.size()
        raise NotImplementedError

    def decode_train(self, encoder_outputs, labels, encoder_final_state=None):
        """Decoding when training.
        Args:
            encoder_outputs (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            labels (LongTensor): A tensor of size `[B, T_out]`
            encoder_final_state (FloatTensor, optional): A tensor of size
                `[1, B, encoder_num_units]`
        Returns:
            outputs (FloatTensor): A tensor of size `[B, T_out]`
            attention_weights (FloatTensor): A tensor of size
                `[B, T_out, T_in]`
        """
        ys = self.embedding(labels[:, :-1])
        # NOTE: remove <EOS>
        ys = self.embedding_dropout(ys)
        labels_max_seq_len = labels.size(1)

        outputs = []
        attention_weights = []
        attention_weights_step = None

        decoder_state = self._init_decoder_state(encoder_final_state)

        for t in range(labels_max_seq_len - 1):
            y = ys[:, t:t + 1, :]

            decoder_outputs, decoder_state, context_vector, attention_weights_step = self._decode_step(
                encoder_outputs,
                y,
                decoder_state,
                attention_weights_step)

            if self.input_feeding_approach:
                # Input-feeding approach
                output = self.decoder_proj_layer(
                    torch.cat([decoder_outputs, context_vector], dim=-1))
                output = self.fc(F.tanh(output))
            else:
                output = self.fc(decoder_outputs + context_vector)

            attention_weights.append(attention_weights_step)
            outputs.append(output)

        # Concatenate in T_out-dimension
        outputs = torch.cat(outputs, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)
        # NOTE; attention_weights in the training stage may be used for computing the
        # coverage, so do not convert to numpy yet.

        return outputs, attention_weights

    def _init_decoder_state(self, encoder_final_state, volatile=False):
        """
        Args:
            encoder_final_state (FloatTensor): A tensor of size
                `[1, B, encoder_num_units]`
            volatile (bool, optional):
        Returns:
            decoder_state (FloatTensor): A tensor of size
                `[1, B, decoder_num_units]`
        """
        if self.init_dec_state_with_enc_state and encoder_final_state is None:
            raise ValueError('Set the final state of the encoder.')

        batch_size = encoder_final_state.size()[1]

        if self.decoder_type == 'lstm':
            c_0 = Variable(torch.zeros(1, batch_size, self.decoder_num_units))

            if volatile:
                c_0.volatile = True

            if self.use_cuda:
                c_0 = c_0.cuda()

            if self.init_dec_state_with_enc_state and self.encoder_type == self.decoder_type:
                # Initialize decoder state with
                # the final state of the top layer of the encoder (forward)
                decoder_state = (encoder_final_state, c_0)
                # TODO: LSTMの場合はメモリセルもencoderのラストで初期化？？
            else:
                h_0 = Variable(torch.zeros(
                    1, batch_size, self.decoder_num_units))

                if volatile:
                    h_0.volatile = True

                if self.use_cuda:
                    h_0 = h_0.cuda()

                decoder_state = (h_0, c_0)
        else:
            # gru decoder
            if self.init_dec_state_with_enc_state and self.encoder_type == self.decoder_type:
                # Initialize decoder state with
                # the final state of the top layer of the encoder (forward)
                decoder_state = encoder_final_state
            else:
                h_0 = Variable(torch.zeros(
                    1, batch_size, self.decoder_num_units))

                if volatile:
                    h_0.volatile = True

                if self.use_cuda:
                    h_0 = h_0.cuda()

                decoder_state = h_0

        return decoder_state

    def _decode_step(self, encoder_outputs, y, decoder_state,
                     attention_weights_step):
        """
        Args:
            encoder_outputs (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            y (FloatTensor): A tensor of size `[B, 1, embedding_dim]`
            decoder_state (FloatTensor): A tensor of size
                `[decoder_num_layers, B, decoder_num_units]`
            attention_weights_step (FloatTensor): A tensor of size `[B, T_in]`
        Returns:
            decoder_outputs (FloatTensor): A tensor of size
                `[B, 1, decoder_num_units]`
            decoder_state (FloatTensor): A tensor of size
                `[decoder_num_layers, B, decoder_num_units]`
            content_vector (FloatTensor): A tensor of size
                `[B, 1, encoder_num_units]`
            attention_weights_step (FloatTensor): A tensor of size `[B, T_in]`
        """
        if self.decoder_type == 'lstm':
            decoder_outputs, decoder_state = self.decoder(
                y, hx=decoder_state)

        elif self.decoder_type == 'gru':
            decoder_outputs, decoder_state = self.decoder(
                y, hx=decoder_state)

        # decoder_outputs: `[B, 1, decoder_num_units]`
        context_vector, attention_weights_step = self.attend(
            encoder_outputs,
            decoder_outputs,
            attention_weights_step)

        return decoder_outputs, decoder_state, context_vector, attention_weights_step

    def decode_infer(self, inputs, beam_width=1):
        """
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            beam_width (int, optional): the size of beam
        Returns:

        """
        if beam_width == 1:
            return self._decode_infer_greedy(inputs)
        else:
            return self._decode_infer_beam_search(inputs, beam_width)

    def _decode_infer_greedy(self, inputs):
        """Greedy decoding when inference.
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
        Returns:
            argmaxs (np.ndarray): A tensor of size `[B, ]`
            attention_weights (np.ndarray): A tensor of size `[B, ]`
        """
        encoder_states, encoder_final_state = self._encode(inputs,
                                                           volatile=True)

        # Start from <SOS>
        batch_size = inputs.size()[0]
        y = np.full((batch_size, 1),
                    fill_value=self.sos_index, dtype=np.int64)
        y = torch.from_numpy(y)
        y = Variable(y, requires_grad=False)
        y.volatile = True
        if self.use_cuda:
            y = y.cuda()
        # NOTE: y: `[B, 1]`

        argmaxs = []
        attention_weights = []
        attention_weights_step = None

        decoder_state = self._init_decoder_state(encoder_final_state,
                                                 volatile=True)

        for _ in range(self.max_decode_length):
            y = self.embedding(y)
            y = self.embedding_dropout(y)

            decoder_outputs, decoder_state, context_vector, attention_weights_step = self._decode_step(
                encoder_states,
                y,
                decoder_state,
                attention_weights_step)

            if self.input_feeding_approach:
                # Input-feeding approach
                output = self.decoder_proj_layer(
                    torch.cat([decoder_outputs, context_vector], dim=-1))
                output = self.fc(F.tanh(output))
            else:
                output = self.fc(decoder_outputs + context_vector)

            # TODO: check this
            output = output.squeeze(dim=1)

            # Pick up 1-best
            y = torch.max(output, dim=1)[1]
            y = y.unsqueeze(dim=1)
            argmaxs.append(y)
            attention_weights.append(attention_weights_step)

            # Break if <EOS> is outputed in all mini-batch
            if torch.sum(y.data == self.eos_index) == y.numel():
                break

        # Concatenate in T_out-dimension
        argmaxs = torch.cat(argmaxs, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)

        # Convert to numpy
        argmaxs = argmaxs.cpu().data.numpy()
        attention_weights = attention_weights.cpu().data.numpy()

        return argmaxs, attention_weights

    def _decode_infer_beam_search(self, inputs, beam_width):
        """Beam search decoding when inference.
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            beam_width (int): the size of beam
        Returns:

        """
        encoder_states, encoder_final_state = self._encode(inputs,
                                                           volatile=True)

        raise NotImplementedError
