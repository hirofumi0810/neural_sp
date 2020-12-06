# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Streaming encoding interface."""

import numpy as np
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
        self.N_c = getattr(encoder, 'chunk_size_current', 0)  # for Transformer
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
        self.conv_n_lookahead = encoder.conv.context_size if encoder.conv is not None else 0

        # for test
        self.eout_chunks = []

    def reset(self, stdout=False):
        self.eout_chunks = []
        self.n_blanks = 0
        self.n_accum_frames = 0
        if stdout:
            print('Reset')

    def register(self):
        pass

    def next_chunk(self):
        self.offset += self.N_l

    def extract_feature(self):
        j = self.offset
        N_l = self.N_l
        N_r = self.N_r

        # Encode input features chunk by chunk
        if getattr(self.encoder, 'conv', None) is not None:
            cnn_context = self.encoder.conv.context_size
            x_chunk = self.x_whole[max(0, j - cnn_context):j + (N_l + N_r) + cnn_context]
        else:
            cnn_context = 0
            x_chunk = self.x_whole[j:j + (N_l + N_r)]

        # zero paddign for the last chunk
        if j > 0 and x_chunk.shape[0] != (N_l + N_r + cnn_context * 2):
            zero_pad = np.zeros(((N_l + N_r + cnn_context * 2) - x_chunk.shape[0], x_chunk.shape[1])).astype(np.float32)
            x_chunk = np.concatenate([x_chunk, zero_pad], axis=0)

        is_last_chunk = (j + N_l - 1) >= len(self.x_whole) - 1
        self.bd_offset = -1  # reset
        self.n_accum_frames += min(self.N_l, x_chunk.shape[1])

        start = j - self.conv_n_lookahead
        end = j + (N_l + N_r) + self.conv_n_lookahead
        lookback = start >= 0
        lookahead = end <= self.x_whole.shape[0] - 1

        return x_chunk, is_last_chunk, lookback, lookahead

    def ctc_vad(self, ctc_probs_chunk, stdout=False):
        """Voice activity detection with CTC posterior probabilities.

        Args:
            ctc_probs_chunk (FloatTensor): `[1, T_chunk, vocab]`
        Returns:
            is_reset (bool): reset encoder/decoder states if successive blank
                labels are generated above the pre-defined threshold (BLANK_THRESHOLD)

        """
        is_reset = False  # detect the first boundary in the same chunk

        if self.n_accum_frames < self.MAX_N_ACCUM_FRAMES:
            return is_reset

        # Segmentation strategy 1:
        # If any segmentation points are not found in the current chunk,
        # encoder states will be carried over to the next chunk.
        # Otherwise, the current chunk is segmented at the point where
        # n_blanks surpasses the threshold.
        topk_ids_chunk = torch.topk(ctc_probs_chunk, k=1, dim=-1, largest=True, sorted=True)[1]
        topk_ids_chunk = topk_ids_chunk[0, :, 0]  # `[T_chunk]`
        bs, xmax_chunk, vocab = ctc_probs_chunk.size()

        # skip all blank segments
        if (topk_ids_chunk == self.blank).sum() == xmax_chunk:
            self.n_blanks += xmax_chunk
            if stdout:
                for j in range(xmax_chunk):
                    print('CTC (T:%d): <blank>' % (self.offset + (j + 1) * self.factor))
                print('All blank segments')
            if self.n_blanks * self.factor >= self.BLANK_THRESHOLD:
                is_reset = True
            return is_reset

        n_blanks_tmp = self.n_blanks
        for j in range(xmax_chunk):
            if topk_ids_chunk[j] == self.blank:
                self.n_blanks += 1
                if stdout:
                    print('CTC (T:%d): <blank>' % (self.offset + (j + 1) * self.factor))

            else:
                if ctc_probs_chunk[0, j, topk_ids_chunk[j]] < self.SPIKE_THRESHOLD:
                    self.n_blanks += 1
                else:
                    self.n_blanks = 0
                if stdout:
                    print('CTC (T:%d): %s' % (self.offset + (j + 1) * self.factor,
                                              self.idx2token([topk_ids_chunk[j].item()])))

            # if not is_reset and (self.n_blanks * self.factor >= self.BLANK_THRESHOLD):# NOTE: select the leftmost blank offset
            if self.n_blanks * self.factor >= self.BLANK_THRESHOLD:  # NOTE: select the rightmost blank offset
                self.bd_offset = j
                is_reset = True
                n_blanks_tmp = self.n_blanks

        if stdout and is_reset:
            print('--- Segment (%d >= %d) ---' % (n_blanks_tmp * self.factor, self.BLANK_THRESHOLD))

        return is_reset

    def backoff(self, x_chunk, decoder, stdout=False):
        if 0 <= self.bd_offset * self.factor < self.N_l - 1:
            # boundary located in the middle of the current chunk
            decoder.n_frames = 0
            offset_prev = self.offset
            self.offset = self.offset - x_chunk[(self.bd_offset + 1) * self.factor:self.N_l].shape[0]
            if stdout:
                print('Back %d frames (%d -> %d)' %
                      (x_chunk[(self.bd_offset + 1) * self.factor:self.N_l].shape[0],
                       offset_prev, self.offset))
