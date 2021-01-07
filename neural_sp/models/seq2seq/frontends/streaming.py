# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Streaming encoding interface."""

import numpy as np
import torch


class Streaming(object):
    """Streaming encoding interface."""

    def __init__(self, x_whole, params, encoder, block_size, idx2token=None):
        """
        Args:
            x_whole (FloatTensor): `[T, input_dim]`
            params (dict): decoding hyperparameters
            encoder (torch.nn.module): encoder module
            block_size (int): block size for streaming inference
            idx2token ():

        """
        super(Streaming, self).__init__()

        self.x_whole = x_whole
        # TODO: implement wav input
        self.feat_dim = x_whole.shape[1]
        self.encoder = encoder
        if self.encoder.conv is not None:
            self.encoder.turn_off_ceil_mode(self.encoder)
        self.idx2token = idx2token

        # latency
        self._factor = encoder.subsampling_factor
        self.N_l = getattr(encoder, 'chunk_size_left', 0)  # for Transformer/Conformer
        self.N_c = encoder.chunk_size_current
        self.N_r = encoder.chunk_size_right
        if self.N_c <= 0 and self.N_r <= 0:
            self.N_c = block_size  # for unidirectional encoder
            assert block_size % self._factor == 0

        # threshold for CTC-VAD
        self.blank_id = 0
        self.is_ctc_vad = params['recog_ctc_vad']
        self.BLANK_THRESHOLD = params['recog_ctc_vad_blank_threshold']
        self.SPIKE_THRESHOLD = params['recog_ctc_vad_spike_threshold']
        self.MAX_N_ACCUM_FRAMES = params['recog_ctc_vad_n_accum_frames']
        assert params['recog_ctc_vad_blank_threshold'] % self._factor == 0
        assert params['recog_ctc_vad_n_accum_frames'] % self._factor == 0
        # NOTE: these parameters are based on 10ms/frame

        self._offset = 0  # global time offset in the session
        self._n_blanks = 0  # number of blank frames
        self._n_accum_frames = 0
        self._bd_offset = -1  # boudnary offset in each block (AFTER subsampling)

        # for CNN frontend
        self.conv_context = encoder.conv.context_size if encoder.conv is not None else 0
        if not getattr(encoder, 'cnn_lookahead', True):
            self.conv_context = 0
            # NOTE: CNN lookahead surpassing the block is not allowed in Transformer/Conformer

        # for test
        self._eout_blocks = []

    @property
    def offset(self):
        return self._offset

    @property
    def n_blanks(self):
        return self._n_blanks

    @property
    def n_accum_frames(self):
        return self._n_accum_frames

    @property
    def bd_offset(self):
        return self._bd_offset

    @property
    def n_cache_block(self):
        return len(self._eout_blocks)

    def reset(self, stdout=False):
        self._eout_blocks = []
        self._n_blanks = 0
        self._n_accum_frames = 0
        if stdout:
            print('Reset')

    def cache_eout(self, eout_block):
        self._eout_blocks.append(eout_block)

    def pop_eouts(self):
        return torch.cat(self._eout_blocks, dim=1)

    def next_block(self):
        self._offset += self.N_c

    def extract_feature(self):
        """Slice acoustic features.

        Returns:
            x_block (np.array): `[T_block, input_dim]`
            is_last_block (bool): the last input block
            cnn_lookback (bool): use lookback frames in CNN
            cnn_lookahead (boo): use lookahead frames in CNN
            xlen_block (int): input length in a block (for the last block)

        """
        j = self._offset
        N_l = self.N_l
        N_c = self.N_c
        N_r = self.N_r

        # Encode input features block by block
        j_left = max(0, j - (self.conv_context + N_l))
        x_block = self.x_whole[j_left:j + (N_c + N_r + self.conv_context)]

        is_last_block = (j + N_c) >= len(self.x_whole)
        # is_last_block = (j + 1 + N_c) >= len(self.x_whole)

        # zero padding for the first blocks
        if j_left == 0:
            zero_pad = np.zeros((self.conv_context + N_l - j, self.feat_dim)).astype(np.float32)
            x_block = np.concatenate([zero_pad, x_block], axis=0)

        # zero padding for the last blocks
        # if j > 0 and x_block.shape[0] != (N_l + N_c + N_r + self.conv_context * 2):
        if x_block.shape[0] != (N_l + N_c + N_r + self.conv_context * 2):
            zero_pad = np.zeros(((N_l + N_c + N_r + self.conv_context * 2) - x_block.shape[0],
                                 self.feat_dim)).astype(np.float32)
            x_block = np.concatenate([x_block, zero_pad], axis=0)
        xlen_block = len(self.x_whole) - j if is_last_block else N_c

        self._bd_offset = -1  # reset
        self._n_accum_frames += min(self.N_c, self.feat_dim)

        start = j - (self.conv_context + N_l)
        end = j + (N_c + N_r + self.conv_context)
        cnn_lookback = start >= 0
        cnn_lookahead = end <= len(self.x_whole) - 1

        return x_block, is_last_block, cnn_lookback, cnn_lookahead, xlen_block

    def ctc_vad(self, ctc_probs_block, stdout=False):
        """Voice activity detection with CTC posterior probabilities.

        Args:
            ctc_probs_block (FloatTensor): `[1, T_block, vocab]`
        Returns:
            is_reset (bool): reset encoder/decoder states if successive blank
                labels are generated above the pre-defined threshold (BLANK_THRESHOLD)

        """
        is_reset = False  # detect the first boundary in the same block

        if self._n_accum_frames < self.MAX_N_ACCUM_FRAMES:
            return is_reset

        assert ctc_probs_block is not None

        # Segmentation strategy 1:
        # If any segmentation points are not found in the current block,
        # encoder states will be carried over to the next block.
        # Otherwise, the current block is segmented at the point where
        # _n_blanks surpasses the threshold.
        topk_ids_block = torch.topk(ctc_probs_block, k=1, dim=-1, largest=True, sorted=True)[1]
        topk_ids_block = topk_ids_block[0, :, 0]  # `[T_block]`
        bs, xmax_block, vocab = ctc_probs_block.size()

        # skip all blank segments
        if (topk_ids_block == self.blank_id).sum() == xmax_block:
            self._n_blanks += xmax_block
            if stdout:
                for j in range(xmax_block):
                    print('CTC (T:%d): <blank>' % (self._offset + (j + 1) * self._factor))
                print('All blank segments')
            if self._n_blanks * self._factor >= self.BLANK_THRESHOLD:
                is_reset = True
            return is_reset

        n_blanks_tmp = self._n_blanks
        for j in range(xmax_block):
            if topk_ids_block[j] == self.blank_id:
                self._n_blanks += 1
                if stdout:
                    print('CTC (T:%d): <blank>' % (self._offset + (j + 1) * self._factor))

            else:
                if ctc_probs_block[0, j, topk_ids_block[j]] < self.SPIKE_THRESHOLD:
                    self._n_blanks += 1
                else:
                    self._n_blanks = 0
                if stdout and self.idx2token is not None:
                    print('CTC (T:%d): %s' % (self._offset + (j + 1) * self._factor,
                                              self.idx2token([topk_ids_block[j].item()])))

            # if not is_reset and (self._n_blanks * self._factor >= self.BLANK_THRESHOLD):# NOTE: select the leftmost blank offset
            if self._n_blanks * self._factor >= self.BLANK_THRESHOLD:  # NOTE: select the rightmost blank offset
                self._bd_offset = j
                is_reset = True
                n_blanks_tmp = self._n_blanks

        if stdout and is_reset:
            print('--- Segment (%d >= %d) ---' % (n_blanks_tmp * self._factor, self.BLANK_THRESHOLD))

        return is_reset

    def backoff(self, x_block, decoder, stdout=False):
        if 0 <= self._bd_offset * self._factor < self.N_c - 1:
            # boundary located in the middle of the current block
            decoder.n_frames = 0
            offset_prev = self._offset
            self._offset = self._offset - x_block[(self._bd_offset + 1) * self._factor:self.N_c].shape[0]
            if stdout:
                print('Back %d frames (%d -> %d)' %
                      (x_block[(self._bd_offset + 1) * self._factor:self.N_c].shape[0],
                       offset_prev, self._offset))
