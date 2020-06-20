#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Streaming encoding interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


class Streaming(object):
    """Streaming encoding interface."""

    def __init__(self, x_whole, params, encoder):
        """
        Args:
            x_whole (FloatTensor): `[T, input_dim]`

        """
        super(Streaming, self).__init__()

        self.blank = 0

        self.x_whole = x_whole
        self.encoder = encoder
        if self.encoder.conv is not None:
            self.encoder.turn_off_ceil_mode(self.encoder)

        # latency
        self.factor = encoder.subsampling_factor
        self.N_l = encoder.chunk_size_left
        # self.N_c = getattr(encoder, 'chunk_size_current', -1)  # for Transformer
        self.N_c = encoder.chunk_size_left  # for Transformer
        self.N_r = encoder.chunk_size_right
        if self.N_c == 0 and self.N_r == 0:
            # self.N_c = params['lc_chunk_size_left']  # for unidirectional encoder
            self.N_c = 40

        # threshold for CTC-VAD
        self.is_ctc_vad = params['recog_ctc_vad']
        self.BLANK_THRESHOLD = params['recog_ctc_vad_blank_threshold']
        self.SPIKE_THRESHOLD = params['recog_ctc_vad_spike_threshold']
        self.MAX_N_ACCUM_FRAMES = params['recog_ctc_vad_n_accum_frames']

        self.offset = 0  # global time offset in the session
        self.n_blanks = 0  # number of blank frames
        self.n_accum_frames = 0
        self.boundary_offset = -1  # boudnary offset in each chunk (after subsampling)

        # for test
        self.eout_chunks = []

    def reset(self):
        self.eout_chunks = []
        self.n_blanks = 0
        self.n_accum_frames = 0

    def register(self):
        pass

    def extract_feature(self):
        j = self.offset
        c = self.N_c
        r = self.N_r

        # Encode input features chunk by chunk
        if getattr(self.encoder, 'conv', None) is not None:
            context = self.encoder.conv.n_frames_context
            x_chunk = self.x_whole[max(0, j - context):j + (c + r) + context]
        else:
            x_chunk = self.x_whole[j:j + (c + r)]

        is_last_chunk = (j + c - 1) >= len(self.x_whole) - 1
        self.boundary_offset = -1  # reset
        self.n_accum_frames += x_chunk.shape[1]

        return x_chunk, is_last_chunk

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
                    #                           idx2token([topk_ids_chunk[0, j, 0].item()])))

                if not is_reset and self.n_blanks > self.BLANK_THRESHOLD:
                    self.boundary_offset = j  # select the most right blank offset
                    self.next_start_offset = self.offset + j
                    is_reset = True

        return is_reset
