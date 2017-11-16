#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Joint CTC-Attention model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from warpctc_pytorch import CTCLoss
except:
    raise ImportError('Install warpctc_pytorch.')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.pytorch.attention.attention_seq2seq import AttentionSeq2seq
from models.pytorch.encoders.load_encoder import load


class JointCTCAttention(AttentionSeq2seq):
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
            (except for <SOS> and <EOS> calsses)
        sos_index (int): index of the start of sentence tag (<SOS>)
        eos_index (int): index of the end of sentence tag (<EOS>)
        max_decode_length (int): the length of output sequences to stop
            prediction when EOS token have not been emitted
        ctc_num_layers (int): index of the layer to attatch a CTC decoder
        ctc_loss_weight (float): A weight parameter for auxiliary CTC loss
        num_stack (int, optional): the number of frames to stack
        splice (int, optional): the number of frames to splice. This is used
            when using CNN-like encoder. Default is 1 frame.
        parameter_init (float, optional): the range of uniform distribution to
            initialize weight parameters (>= 0)
        downsample_list (list, optional):
        init_dec_state_with_enc_state (bool, optional): if True, initialize
            decoder state with the final encoder state.
        sharpening_factor (float, optional): a sharpening factor in the
            softmax layer for computing attention weights
        logits_temperature (float, optional): a parameter for smoothing the
            softmax layer in outputing probabilities
        sigmoid_smoothing (bool, optional): if True, replace softmax function
            in computing attention weights with sigmoid function for smoothing
        input_feeding_approach (bool, optional): See detail in
            Luong, Minh-Thang, Hieu Pham, and Christopher D. Manning.
            "Effective approaches to attention-based neural machine translation."
                arXiv preprint arXiv:1508.04025 (2015).
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
                 decoder_num_layers,
                 decoder_dropout,
                 embedding_dim,
                 embedding_dropout,
                 num_classes,
                 sos_index,
                 eos_index,
                 ctc_num_layers,  # ***
                 ctc_loss_weight,  # ***
                 max_decode_length=100,
                 num_stack=1,
                 splice=1,
                 parameter_init=0.1,
                 downsample_list=[],
                 init_dec_state_with_enc_state=True,
                 sharpening_factor=1,
                 logits_temperature=1,
                 sigmoid_smoothing=False,
                 input_feeding_approach=False):

        super(JointCTCAttention, self).__init__(
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
            decoder_num_layers=decoder_num_layers,
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

        # TODO:
        # clip_activation
        # time_major

        assert encoder_num_layers >= ctc_num_layers, 'ctc_num_layers must be equal to or less than encoder_num_layers'

        # Common setting
        self.name = 'pt_joint_ctc_attn'

        # Setting for MTL
        self.ctc_num_layers = ctc_num_layers
        self.ctc_loss_weight = ctc_loss_weight
        self.ctc_num_classes = num_classes + 1

        #########################
        # Encoder
        # NOTE: overide encoder
        #########################
        # Load an instance
        if len(downsample_list) == 0:
            encoder = load(encoder_type=encoder_type + '_hierarchical')
        else:
            raise NotImplementedError

        # Call the encoder function
        if encoder_type in ['lstm', 'gru', 'rnn']:
            if len(downsample_list) == 0:
                self.encoder = encoder(
                    input_size=self.input_size,
                    rnn_type=encoder_type,
                    bidirectional=encoder_bidirectional,
                    num_units=encoder_num_units,
                    num_proj=encoder_num_proj,
                    num_layers=encoder_num_layers,
                    num_layers_sub=ctc_num_layers,
                    dropout=encoder_dropout,
                    parameter_init=parameter_init,
                    use_cuda=self.use_cuda,
                    batch_first=True)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        self.fc_ctc = nn.Linear(
            encoder_num_units * self.encoder_num_directions, self.ctc_num_classes)

    def forward(self, inputs, inputs_seq_len, labels, volatile=False):
        """Forward computation.
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            inputs_seq_len (IntTensor): A tensor of size `[B]`
            labels (LongTensor): A tensor of size `[B, T_out]`
            volatile (bool, optional): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            logits (FloatTensor): A tensor of size
                `[B, T_out, num_classes (including <SOS> and <EOS>)]`
            attention_weights (FloatTensor): A tensor of size
                `[B, T_out, T_in]`
            logits_ctc (FloatTensor): A tensor of size
                `[B, T_in, ctc_num_classes (including blank)]`
            perm_indices ():
        """
        encoder_outputs, encoder_final_state, logits_ctc, perm_indices = self._encode(
            inputs, inputs_seq_len, volatile)

        logits, attention_weights = self._decode_train(
            encoder_outputs, labels[perm_indices], encoder_final_state)
        return logits, attention_weights, logits_ctc, perm_indices

    def _encode(self, inputs, inputs_seq_len, volatile):
        """Encode acoustic features.
        Args:
            inputs (FloatTensor): A tensor of size `[B, T_in, input_size]`
            inputs_seq_len (IntTensor): A tensor of size `[B]`
            volatile (bool): if True, the history will not be saved.
                This should be used in inference model for memory efficiency.
        Returns:
            encoder_outputs (FloatTensor): A tensor of size
                `[B, T_in, encoder_num_units]`
            encoder_final_state (FloatTensor): A tensor of size
                `[1, B, decoder_num_units (may be equal to encoder_num_units)]`
            logits_ctc (FloatTensor): A tensor of size
                `[B, T_in, ctc_num_classes (including blank)]`
            perm_indices ():
        """
        encoder_outputs, encoder_final_state,  encoder_outputs_sub, _, perm_indices = self.encoder(
            inputs, inputs_seq_len, volatile, mask_sequence=True)
        # NOTE: encoder_outputs:
        # `[B, T_in, encoder_num_units * encoder_num_directions]`
        # encoder_final_state: `[1, B, encoder_num_units]`
        # encoder_outputs_sub: `[B, T_in, encoder_num_units *
        # encoder_num_directions]`

        batch_size, max_time, encoder_num_units = encoder_outputs.size()

        ####################
        # Attention
        ####################
        # Sum bidirectional outputs
        if self.encoder_bidirectional:
            encoder_outputs = encoder_outputs[:, :, :encoder_num_units // 2] + \
                encoder_outputs[:, :, encoder_num_units // 2:]
            # NOTE: encoder_outputs: `[B, T_in, encoder_num_units]`

        # Bridge between the encoder and decoder
        if self.encoder_num_units != self.decoder_num_units:
            # Bridge between the encoder and decoder
            encoder_outputs = self.bridge(encoder_outputs)
            encoder_final_state = self.bridge(
                encoder_final_state.view(-1, encoder_num_units))
            encoder_final_state = encoder_final_state.view(1, batch_size, -1)

        ####################
        # CTC
        ####################
        # Convert to 2D tensor
        encoder_outputs_sub = encoder_outputs_sub.contiguous()
        encoder_outputs_sub = encoder_outputs_sub.view(
            batch_size, max_time, -1)

        logits_ctc = self.fc_ctc(encoder_outputs_sub)

        # Reshape back to 3D tensor
        logits_ctc = logits_ctc.view(batch_size, max_time, -1)

        return encoder_outputs, encoder_final_state, logits_ctc, perm_indices

    def compute_loss(self, logits, inputs_seq_len, labels, labels_seq_len,
                     logits_ctc, labels_ctc, labels_seq_len_ctc,
                     attention_weights=None, coverage_weight=0):
        """Compute loss.
        Args:
            logits (FloatTensor): A tensor of size `[B, T_out, num_classes]`
            inputs_seq_len (IntTensor): A tensor of size `[B]`
            labels (LongTensor): A tensor of size `[B, T_out]`
            labels_seq_len (IntTensor): A tensor of size `[B]`
            logits_ctc (FloatTensor): A tensor of size
                `[B, T_in, ctc_num_classes]`
            labels_ctc (IntTensor): A tensor of size `[B, T_out_ctc]`
            labels_seq_len_ctc (IntTensor): A tensor of size `[B]`
            attention_weights (FloatTensor): A tensor of size
                `[B, T_out, T_in]`
            coverage_weight (float, optional):
        Returns:
            loss (FloatTensor): A tensor of size `[1]`
        """
        batch_size, _, num_classes = logits.size()

        if self.logits_temperature != 1:
            logits /= self.logits_temperature
            logits_ctc /= self.logits_temperature

        ####################
        # Attention
        ####################
        logits = logits.view((-1, num_classes))
        labels = labels[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(logits, labels,
                               ignore_index=self.sos_index,
                               size_average=False)
        # NOTE: labels are padded by sos_index

        if coverage_weight != 0:
            pass
            # coverage = self._compute_coverage(attention_weights)
            # loss += coverage_weight * coverage

        ####################
        # CTC
        ####################
        ctc_loss_fn = CTCLoss()

        # Concatenate all labels for warpctc_pytorch
        # `[B, T_out]` -> `[1,]`
        total_lables_seq_len = labels_seq_len_ctc.data.sum()
        concatenated_labels = Variable(
            torch.zeros(total_lables_seq_len)).int()
        label_counter = 0
        for i_batch in range(batch_size):
            concatenated_labels[label_counter:label_counter +
                                labels_seq_len_ctc.data[i_batch]] = labels_ctc[i_batch][:labels_seq_len_ctc.data[i_batch]]
            label_counter += labels_seq_len_ctc.data[i_batch]
        labels_ctc = labels_ctc.view(-1,)

        ctc_loss = ctc_loss_fn(logits_ctc, concatenated_labels.cpu(),
                               inputs_seq_len.cpu(), labels_seq_len_ctc.cpu())

        # Linearly interpolate XE sequence loss and CTC loss
        loss = self.ctc_loss_weight * ctc_loss + \
            (1 - self.ctc_loss_weight) * loss

        # Average the loss by mini-batch
        loss /= batch_size

        return loss
