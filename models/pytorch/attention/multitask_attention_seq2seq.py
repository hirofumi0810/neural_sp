#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.pytorch.base import ModelBase
from models.pytorch.attention.attention_seq2seq import AttentionSeq2seq
from models.pytorch.encoders.load_encoder import load
from models.pytorch.attention.attention_layer import AttentionMechanism
from utils.io.variable import var2np


class MultitaskAttentionSeq2seq(AttentionSeq2seq):

    def __init__(self,
                 input_size,
                 encoder_type,
                 encoder_bidirectional,
                 encoder_num_units,
                 encoder_num_proj,
                 encoder_num_layers,
                 encoder_num_layers_sub,  # ***
                 encoder_dropout,
                 attention_type,
                 attention_dim,
                 decoder_type,
                 decoder_num_units,
                 decoder_num_proj,
                 decdoder_num_layers,
                 decoder_num_units_sub,  # ***
                 decoder_num_proj_sub,  # ***
                 decdoder_num_layers_sub,  # ***
                 decoder_dropout,
                 embedding_dim,
                 embedding_dim_sub,  # ***
                 embedding_dropout,
                 num_classes,
                 sos_index,
                 eos_index,
                 num_classes_sub,  # ***
                 sos_index_sub,  # ***
                 eos_index_sub,  # ***
                 max_decode_length=50,
                 max_decode_length_sub=100,  # ***
                 num_stack=1,
                 splice=1,
                 parameter_init=0.1,
                 downsample_list=[],
                 init_dec_state_with_enc_state=True,
                 sharpening_factor=1,
                 logits_temperature=1,
                 sigmoid_smoothing=False,
                 input_feeding_approach=False):

        super(MultitaskAttentionSeq2seq, self).__init__(
            input_size=input_size,
            encoder_type=encoder_type,
            encoder_bidirectional=encoder_bidirectional,
            encoder_num_units=encoder_num_units,
            encoder_num_proj=encoder_num_proj,
            encoder_num_layers=encoder_num_layers,
            encoder_dropout=encoder_dropout,
            attention_type=attention_type,
            attention_dim=attention_dim,
            decoder_type=decoder_type,
            decoder_num_units=decoder_num_units,
            decoder_num_proj=decoder_num_units,
            decdoder_num_layers=decdoder_num_layers,
            decoder_dropout=decoder_dropout,
            embedding_dim=embedding_dim,
            embedding_dropout=embedding_dropout,
            num_classes=num_classes,
            sos_index=sos_index,
            eos_index=eos_index,
            max_decode_length=max_decode_length,
            num_stack=num_stack,
            splice=splice,
            parameter_init=parameter_init,
            downsample_list=downsample_list,
            init_dec_state_with_enc_state=init_dec_state_with_enc_state,
            sharpening_factor=sharpening_factor,
            logits_temperature=logits_temperature,
            sigmoid_smoothing=sigmoid_smoothing,
            input_feeding_approach=input_feeding_approach)

        # Setting for the encoder
        self.encoder_num_layers_sub = encoder_num_layers_sub

        # Setting for the decoder
        self.decoder_num_units_sub = decoder_num_units_sub
        self.decoder_num_proj_sub = decoder_num_proj_sub
        self.decdoder_num_layers_sub = decdoder_num_layers_sub
        self.embedding_dim_sub = embedding_dim_sub
        self.num_classes_sub = num_classes_sub + 2
        # NOTE: add <SOS> and <EOS>
        self.sos_index_sub = sos_index_sub
        self.eos_index_sub = eos_index_sub
        self.max_decode_length_sub = max_decode_length_sub

        # Common setting
        self.name = 'pt_multitask_attention_seq2seq'

        #########################
        # Encoder
        # NOTE: overide encoder
        #########################
        # Load an instance
        encoder = load(encoder_type=encoder_type + '_multitask')

        # Call the encoder function
        if encoder_type in ['lstm', 'gru', 'rnn']:
            if len(downsample_list) == 0:
                self.encoder = encoder(
                    input_size=self.input_size,
                    rnn_type=encoder_type,
                    bidirectional=encoder_bidirectional,
                    num_units=encoder_num_units,
                    num_proj=encoder_num_proj,
                    num_layers_main=encoder_num_layers,
                    num_layers_sub=encoder_num_layers_sub,
                    dropout=encoder_dropout,
                    parameter_init=parameter_init,
                    use_cuda=self.use_cuda,
                    batch_first=True)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        ##############################
        # Decoder in the sub task
        ##############################
        if decoder_type == 'lstm':
            self.decoder_sub = nn.LSTM(
                embedding_dim_sub,
                hidden_size=decoder_num_units_sub,
                num_layers=1,
                bias=True,
                batch_first=True,
                dropout=decoder_dropout,
                bidirectional=False)
        elif decoder_type == 'gru':
            self.decoder_sub = nn.GRU(
                embedding_dim_sub,
                hidden_size=decoder_num_units_sub,
                num_layers=1,
                bias=True,
                batch_first=True,
                dropout=decoder_dropout,
                bidirectional=False)
        else:
            raise TypeError
        # NOTE: decoder is unidirectional and only 1 layer now

        ###################################
        # Attention layer in the sub task
        ###################################
        self.attend_sub = AttentionMechanism(
            encoder_num_units=encoder_num_units,
            decoder_num_units=decoder_num_units_sub,
            attention_type=attention_type,
            attention_dim=attention_dim,
            sharpening_factor=sharpening_factor)

        ##################################################
        # Bridge layer between the encoder and decoder
        ##################################################
        if encoder_num_units != decoder_num_units_sub:
            self.bridge_sub = nn.Linear(
                encoder_num_units, decoder_num_units_sub)
        else:
            self.bridge_sub = None

        self.embedding_sub = nn.Embedding(
            self.num_classes_sub, embedding_dim_sub)
        self.embedding_dropout_sub = nn.Dropout(embedding_dropout)
        # TODO: dropoutは別に用意する必要ある（実装確認）？

        if input_feeding_approach:
            self.decoder_proj_layer_sub = nn.Linear(
                decoder_num_units_sub * 2, decoder_num_proj_sub)
            # NOTE: input-feeding approach
            self.fc_sub = nn.Linear(decoder_num_proj_sub, self.num_classes_sub)
        else:
            self.fc_sub = nn.Linear(
                decoder_num_units_sub, self.num_classes_sub)
        # NOTE: <SOS> is removed because the decoder never predict <SOS> class
        # TODO: self.num_classes_sub - 1

    def forward(self, inputs, labels, labels_sub, volatile=False):
        """
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            labels (LongTensor): labels in the main task.
                A tensor of size `[B, T_out]`
            labels_sub (LongTensor): labels in the sub task.
                A tensor of size `[B, T_out_sub]`
            volatile (bool, optional): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            outputs (FloatTensor): outputs in the main task.
                A tensor of size
                `[B, T_out, num_classes (including <SOS> and <EOS>)]`
            attention_weights (FloatTensor): attention weights in the main task.
                A tensor of size `[B, T_out, T_in]`
            outputs_sub (FloatTensor): outputs in the sub task.
                A tensor of size
                `[B, T_out_sub, num_classes_sub (including <SOS> and <EOS>)]`
            attention_weights_sub (FloatTensor): attention weights in the sub task.
                A tensor of size `[B, T_out_sub, T_in]`
        """
        encoder_outputs, encoder_final_state, encoder_outputs_sub, encoder_final_state_sub = self._encode(
            inputs, volatile)

        # main task
        outputs, attention_weights = self._decode_train(
            encoder_outputs, labels, encoder_final_state)

        # sub task
        outputs_sub, attention_weights_sub = self._decode_train_sub(
            encoder_outputs_sub, labels_sub, encoder_final_state_sub)

        return outputs, attention_weights, outputs_sub, attention_weights_sub

    def _encode(self, inputs, volatile):
        """Encode acoustic features.
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            volatile (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            encoder_outputs (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            encoder_final_state (FloatTensor): A tensor of size
                `[1, B, decoder_num_units (may be equal to encoder_num_units)]`
            encoder_outputs_sub (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            encoder_final_state_sub (FloatTensor): A tensor of size
                `[1, B, decoder_num_units_sub (may be equal to encoder_num_units)]`
        """
        encoder_outputs, encoder_final_state, encoder_outputs_sub, encoder_final_state_sub = self.encoder(
            inputs, volatile)
        # NOTE: encoder_outputs:
        # `[B, T_in, encoder_num_units * encoder_num_directions]`
        # encoder_final_state: `[1, B, encoder_num_units]`
        # encoder_outputs_sub: `[B, T_in, encoder_num_units * encoder_num_directions]`
        # encoder_final_state_sub: `[1, B, encoder_num_units]`

        batch_size, max_time, encoder_num_units = encoder_outputs.size()

        # Sum bidirectional outputs
        if self.encoder_bidirectional:
            encoder_outputs = encoder_outputs[:, :, :encoder_num_units // 2] + \
                encoder_outputs[:, :, encoder_num_units // 2:]
            # NOTE: encoder_outputs: `[B, T_in, encoder_num_units]`

            encoder_outputs_sub = encoder_outputs_sub[:, :, :encoder_num_units // 2] + \
                encoder_outputs_sub[:, :, encoder_num_units // 2:]
            # NOTE: encoder_outputs_sub: `[B, T_in, encoder_num_units]`

        # Bridge between the encoder and decoder in the main task
        if self.encoder_num_units != self.decoder_num_units:
            # Bridge between the encoder and decoder
            encoder_outputs = self.bridge(encoder_outputs)
            encoder_final_state = self.bridge(
                encoder_final_state.view(-1, encoder_num_units))
            encoder_final_state = encoder_final_state.view(1, batch_size, -1)

        # Bridge between the encoder and decoder in the sub task
        if self.encoder_num_units != self.decoder_num_units_sub:
            # Bridge between the encoder and decoder
            encoder_outputs_sub = self.bridge_sub(encoder_outputs_sub)
            encoder_final_state_sub = self.bridge_sub(
                encoder_final_state_sub.view(-1, encoder_num_units))
            encoder_final_state_sub = encoder_final_state_sub.view(
                1, batch_size, -1)

        return encoder_outputs, encoder_final_state, encoder_outputs_sub, encoder_final_state_sub

    def compute_loss(self, outputs, labels, outputs_sub, labels_sub,
                     attention_weights=None, attention_weights_sub=None,
                     coverage_weight=0):
        """Compute multitask loss.
        Args:
            outputs (FloatTensor): A tensor of size `[B, T_out, num_classes]`
            outputs_sub (FloatTensor): A tensor of size
                `[B, T_out_sub, num_classes_sub]`
            labels (LongTensor): A tensor of size `[B, T_out]`
            labels_sub (LongTensor): labels in the sub task.
                A tensor of size `[B, T_out_sub]`
            attention_weights (FloatTensor): A tensor of size
                `[B, T_out, T_in]`
            attention_weights_sub (FloatTensor): A tensor of size
                `[B, T_out_sub, T_in]`
            coverage_weight (float, optional):
        Returns:
            loss (FloatTensor): A tensor of size `[]`
        """
        batch_size, _, num_classes = outputs.size()
        outputs = outputs.view((-1, num_classes))
        labels = labels[:, 1:].contiguous().view(-1)

        _, _, num_classes_sub = outputs_sub.size()
        outputs_sub = outputs_sub.view((-1, num_classes_sub))
        labels_sub = labels_sub[:, 1:].contiguous().view(-1)

        loss = F.cross_entropy(outputs, labels,
                               size_average=False)
        loss += F.cross_entropy(outputs_sub, labels_sub,
                                size_average=False)

        # Average the loss by mini-batch
        loss /= batch_size

        if coverage_weight != 0:
            pass
            # coverage = self._compute_coverage(attention_weights)
            # loss += coverage_weight * coverage

        return loss

    def _decode_train_sub(self, encoder_outputs, labels,
                          encoder_final_state=None):
        """Decoding when training in the sub task.
        Args:
            encoder_outputs (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            labels (LongTensor): A tensor of size `[B, T_out_sub]`
            encoder_final_state (FloatTensor, optional): A tensor of size
                `[1, B, encoder_num_units]`
        Returns:
            outputs (FloatTensor): A tensor of size `[B, T_out_sub, num_classes]`
            attention_weights_sub (FloatTensor): A tensor of size
                `[B, T_out_sub, T_in]`
        """
        ys = self.embedding_sub(labels[:, :-1])
        # NOTE: remove <EOS>
        ys = self.embedding_dropout_sub(ys)
        labels_max_seq_len = labels.size(1)

        decoder_state = self._init_decoder_state(
            self.init_dec_state_with_enc_state,
            encoder_final_state)

        outputs = []
        attention_weights = []
        attention_weights_step = None

        for t in range(labels_max_seq_len - 1):
            y = ys[:, t:t + 1, :]

            decoder_outputs, decoder_state, context_vector, attention_weights_step = self._decode_step_sub(
                encoder_outputs,
                y,
                decoder_state,
                attention_weights_step)

            if self.input_feeding_approach:
                # Input-feeding approach
                output = self.decoder_proj_layer_sub(
                    torch.cat([decoder_outputs, context_vector], dim=-1))
                output = self.fc_sub(F.tanh(output))
            else:
                output = self.fc_sub(decoder_outputs + context_vector)

            attention_weights.append(attention_weights_step)
            outputs.append(output)

        # Concatenate in T_out-dimension
        outputs = torch.cat(outputs, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)
        # NOTE; attention_weights in the training stage may be used for computing the
        # coverage, so do not convert to numpy yet.

        return outputs, attention_weights

    def _decode_step_sub(self, encoder_outputs, y, decoder_state,
                         attention_weights_step):
        """
        Args:
            encoder_outputs (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            y (FloatTensor): A tensor of size `[B, 1, embedding_dim]`
            decoder_state (FloatTensor): A tensor of size
                `[decoder_num_layers_sub, B, decoder_num_units_sub]`
            attention_weights_step (FloatTensor): A tensor of size `[B, T_in]`
        Returns:
            decoder_outputs (FloatTensor): A tensor of size
                `[B, 1, decoder_num_units_sub]`
            decoder_state (FloatTensor): A tensor of size
                `[decoder_num_layers_sub, B, decoder_num_units_sub]`
            content_vector (FloatTensor): A tensor of size
                `[B, 1, encoder_num_units]`
            attention_weights_step (FloatTensor): A tensor of size `[B, T_in]`
        """
        if self.decoder_type == 'lstm':
            decoder_outputs, decoder_state = self.decoder_sub(
                y, hx=decoder_state)
        elif self.decoder_type == 'gru':
            decoder_outputs, decoder_state = self.decoder_sub(
                y, hx=decoder_state)

        # decoder_outputs: `[B, 1, decoder_num_units]`
        context_vector, attention_weights_step = self.attend_sub(
            encoder_outputs,
            decoder_outputs,
            attention_weights_step)

        return decoder_outputs, decoder_state, context_vector, attention_weights_step

    def decode_infer_sub(self, inputs, beam_width=1):
        """
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            beam_width (int, optional): the size of beam
        Returns:

        """
        if beam_width == 1:
            return self._decode_infer_greedy_sub(inputs)
        else:
            return self._decode_infer_beam_sub(inputs, beam_width=beam_width)

    def _decode_infer_greedy_sub(self, inputs):
        """Greedy decoding when inference.
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
        Returns:
            argmaxs (np.ndarray): A tensor of size `[B, T_out_sub]`
            attention_weights (np.ndarray): A tensor of size
                `[B, T_out_sub, T_in]`
        """
        encoder_outputs, encoder_final_state = self._encode(
            inputs, volatile=True)[2:]

        batch_size = inputs.size(0)

        # Start from <SOS>
        y = self._create_token(value=self.sos_index_sub, batch_size=batch_size)

        # Initialize decoder state
        decoder_state = self._init_decoder_state(
            self.init_dec_state_with_enc_state,
            encoder_final_state,
            volatile=True)

        argmaxs = []
        attention_weights = []
        attention_weights_step = None

        for _ in range(self.max_decode_length_sub):
            y = self.embedding_sub(y)
            y = self.embedding_dropout_sub(y)
            # TODO: remove dropout??

            decoder_outputs, decoder_state, context_vector, attention_weights_step = self._decode_step_sub(
                encoder_outputs,
                y,
                decoder_state,
                attention_weights_step)

            if self.input_feeding_approach:
                # Input-feeding approach
                output = self.decoder_proj_layer_sub(
                    torch.cat([decoder_outputs, context_vector], dim=-1))
                logits = self.fc_sub(F.tanh(output))
            else:
                logits = self.fc_sub(decoder_outputs + context_vector)

            logits = logits.squeeze(dim=1)
            # NOTE: `[B, 1, num_classes]` -> `[B, num_classes]`

            # Path through the softmax layer & convert to log-scale
            log_probs = self.log_softmax(logits)

            # Pick up 1-best
            y = torch.max(log_probs, dim=1)[1]
            y = y.unsqueeze(dim=1)
            argmaxs.append(y)
            attention_weights.append(attention_weights_step)

            # Break if <EOS> is outputed in all mini-batch
            if torch.sum(y.data == self.eos_index_sub) == y.numel():
                break

        # Concatenate in T_out-dimension
        argmaxs = torch.cat(argmaxs, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)

        # Convert to numpy
        argmaxs = var2np(argmaxs)
        attention_weights = var2np(attention_weights)

        return argmaxs, attention_weights

    def _decode_infer_beam_sub(self, inputs, beam_width):
        raise NotImplementedError
