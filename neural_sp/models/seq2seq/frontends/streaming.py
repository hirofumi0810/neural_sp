# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Streaming encoding interface."""

import numpy as np
import torch


class Streaming(object):
    """Streaming encoding interface."""

    def __init__(self, x_whole, params, encoder, idx2token=None):
        """
        Args:
            x_whole (FloatTensor): `[T, input_dim]`
            params (dict): decoding hyperparameters
            encoder (torch.nn.module): encoder module
            idx2token (): converter from index to token

        """
        super(Streaming, self).__init__()

        self.x_whole = x_whole
        self.xmax_whole = len(x_whole)
        # TODO: implement wav input
        self.input_dim = x_whole.shape[1]
        self.enc_type = encoder.enc_type
        self.idx2token = idx2token

        if self.enc_type in ['lstm', 'conv_lstm', 'conv_uni_transformer',
                             'conv_uni_conformer', 'conv_uni_conformer_v2']:
            self.streaming_type = 'unidir'
        elif 'lstm' in self.enc_type or 'gru' in self.enc_type:
            self.streaming_type = 'lc_bidir'
        else:
            assert hasattr(encoder, 'streaming_type')
            self.streaming_type = getattr(encoder, 'streaming_type', '')  # for LC-Transformer/Conformer

        # latency related
        self._factor = encoder.subsampling_factor
        self.N_l = getattr(encoder, 'chunk_size_left', 0)  # for LC-Transformer/Conformer
        self.N_c = encoder.chunk_size_current
        self.N_r = encoder.chunk_size_right
        if self.streaming_type == 'mask':
            self.N_l = 0
            # NOTE: context in previous chunks are cached inside the encoder
        if self.N_c <= 0 and self.N_r <= 0:
            self.N_c = params['recog_block_sync_size']  # for unidirectional encoder
            assert self.N_c % self._factor == 0
        # NOTE: these lengths are the ones before subsampling

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
            # NOTE: CNN lookahead surpassing a block is not allowed in LC-Transformer/Conformer.
            # Unidirectional Transformer/Conformer can use lookahead in frontend CNN.

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
            is_last_block (bool): flag for the last input block
            cnn_lookback (bool): use lookback frames in CNN
            cnn_lookahead (bool): use lookahead frames in CNN
            xlen_block (int): input length of the cernter region in a block (for the last block)

        """
        j = self._offset
        N_l, N_c, N_r = self.N_l, self.N_c, self.N_r

        # Encode input features block by block
        start = j - (self.conv_context + N_l)
        end = j + (N_c + N_r + self.conv_context)
        x_block = self.x_whole[max(0, start):end]

        is_last_block = (j + N_c) >= self.xmax_whole
        cnn_lookback = self.streaming_type != 'reshape' and start >= 0
        cnn_lookahead = self.streaming_type != 'reshape' and end < self.xmax_whole
        N_conv = self.conv_context if j == 0 or is_last_block else self.conv_context * 2
        # TODO: implement frame stacking

        if self.streaming_type in ['reshape', 'mask']:
            xlen_block = min(self.xmax_whole - j, N_c)
        elif self.streaming_type == 'lc_bidir':
            xlen_block = min(self.xmax_whole - j + N_conv, N_c + N_conv)
        else:
            xlen_block = len(x_block)

        if self.streaming_type == 'reshape':
            # zero padding for the first blocks
            if start < 0:
                zero_pad = np.zeros((-start, self.input_dim)).astype(np.float32)
                x_block = np.concatenate([zero_pad, x_block], axis=0)
            # zero padding for the last blocks
            if len(x_block) < (N_l + N_c + N_r):
                zero_pad = np.zeros(((N_l + N_c + N_r) - len(x_block), self.input_dim)).astype(np.float32)
                x_block = np.concatenate([x_block, zero_pad], axis=0)

        self._bd_offset = -1  # reset
        self._n_accum_frames += min(self.N_c, xlen_block)

        xlen_block = max(xlen_block, self._factor)  # to avoid elen=0 after subsampling

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
