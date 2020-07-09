#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Streaming encoding interface."""

import torch


class Streaming(object):
    """Streaming encoding interface."""

    def __init__(self, x_whole, params, encoder, idx2token):
        """
        Args:
            x_whole (FloatTensor): `[T, input_dim]`

        """
        super(Streaming, self).__init__()

        self.x_whole = x_whole
        self.encoder = encoder
        if self.encoder.conv is not None:
            self.encoder.turn_off_ceil_mode(self.encoder)
        self.idx2token = idx2token

        # latency
        self.factor = encoder.subsampling_factor
        self.N_l = encoder.chunk_size_left
        # self.N_c = getattr(encoder, 'chunk_size_current', -1)  # for Transformer
        self.N_c = encoder.chunk_size_left  # for Transformer
        self.N_r = encoder.chunk_size_right
        if self.N_l == 0 and self.N_r == 0:
            self.N_l = 40  # for unidirectional encoder
            # TODO(hirofumi0810): make this hyper-parameters

        # threshold for CTC-VAD
        self.blank = 0
        self.is_ctc_vad = params['recog_ctc_vad']
        self.BLANK_THRESHOLD = params['recog_ctc_vad_blank_threshold']
        self.SPIKE_THRESHOLD = params['recog_ctc_vad_spike_threshold']
        self.MAX_N_ACCUM_FRAMES = params['recog_ctc_vad_n_accum_frames']
        assert params['recog_ctc_vad_blank_threshold'] % self.factor == 0
        assert params['recog_ctc_vad_n_accum_frames'] % self.factor == 0
        # NOTE: these parameters are based on 10ms/frame

        self.offset = 0  # global time offset in the session
        self.n_blanks = 0  # number of blank frames
        self.n_accum_frames = 0
        self.bd_offset = -1  # boudnary offset in each chunk (AFTER subsampling)

        # for CNN
        self.conv_lookback_n_frames = encoder.conv.n_frames_context if encoder.conv is not None else 0
        self.conv_lookahead_n_frames = encoder.conv.n_frames_context if encoder.conv is not None else 0

        # for test
        self.eout_chunks = []

    def reset(self):
        self.eout_chunks = []
        self.n_blanks = 0
        self.n_accum_frames = 0

    def register(self):
        pass

    def next_chunk(self):
        self.offset += self.N_l

    def extract_feature(self):
        j = self.offset
        l = self.N_l
        r = self.N_r

        # Encode input features chunk by chunk
        if getattr(self.encoder, 'conv', None) is not None:
            context = self.encoder.conv.n_frames_context
            x_chunk = self.x_whole[max(0, j - context):j + (l + r) + context]
        else:
            x_chunk = self.x_whole[j:j + (l + r)]

        is_last_chunk = (j + l - 1) >= len(self.x_whole) - 1
        self.bd_offset = -1  # reset
        self.n_accum_frames += x_chunk.shape[1]

        start = j - self.conv_lookback_n_frames
        end = j + (l + r) + self.conv_lookahead_n_frames
        lookback = start >= 0
        lookahead = end <= self.x_whole.shape[0] - 1

        return x_chunk, is_last_chunk, lookback, lookahead

    def ctc_vad(self, ctc_probs_chunk):
        """Voice activity detection with CTC posterior probabilities.

        Args:
            ctc_probs_chunk (FloatTensor): `[1, T_chunk, vocab]`
        Returns:
            is_reset (bool): reset encoder/decoder states if successive blank
                labels are generated above the pre-defined threshold (BLANK_THRESHOLD)

        """
        is_reset = False  # detect the first boundary in the same chunk

        # Segmentation strategy 1:
        # If any segmentation points are not found in the current chunk,
        # encoder states will be carried over to the next chunk.
        # Otherwise, the current chunk is segmented at the point where
        # n_blanks surpasses the threshold.
        if self.n_accum_frames >= self.MAX_N_ACCUM_FRAMES:
            _, topk_ids_chunk = torch.topk(
                ctc_probs_chunk, k=1, dim=-1, largest=True, sorted=True)

            for j in range(ctc_probs_chunk.size(1)):
                if topk_ids_chunk[0, j, 0] == self.blank:
                    self.n_blanks += 1
                    # print('CTC (T:%d): <blank>' % (self.offset + j * self.factor))

                elif ctc_probs_chunk[0, j, topk_ids_chunk[0, j, 0]] < self.SPIKE_THRESHOLD:
                    self.n_blanks += 1
                    # print('CTC (T:%d): <blank>' % (self.offset + j * self.factor))

                else:
                    self.n_blanks = 0
                    # print('CTC (T:%d): %s' % (self.offset + j * self.factor,
                    #                           self.idx2token([topk_ids_chunk[0, j, 0].item()])))

                if not is_reset and (self.n_blanks * self.factor > self.BLANK_THRESHOLD):
                    self.bd_offset = j  # select the most right blank offset
                    self.next_start_offset = self.offset + j
                    is_reset = True

        return is_reset

    def backoff(self, x_chunk, decoder):
        if 0 <= self.bd_offset * self.factor < self.N_l - 1:
            # the boundary locatted in the middle of the current chunk
            self.offset -= x_chunk[(self.bd_offset + 1) * self.factor:self.N_l].shape[0]
            decoder.n_frames = 0
            # print('Back %d frames' %
            #       (x_chunk[(self.bd_offset + 1) * self.factor:self.N_l].shape[0]))
