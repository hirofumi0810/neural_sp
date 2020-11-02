# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import codecs
import numpy as np
import os

import sentencepiece as spm
from collections import deque


class WordAlignmentConverter(object):
    """Class for converting word alignment into word-piece alignment.

    Args:
        dict_path (str): path to a dictionary file
        wp_model ():

    """

    def __init__(self, dict_path, wp_model):
        # Load a dictionary file
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(wp_model)

    def __call__(self, alignment_dir, speaker, utt_id, text):
        """Convert word alignment into word-piece alignment.

        Args:
            alignment_dir (str): path to word alignment directory
            speaker (str): speaker ID
            utt_id (str): utterance ID
            text (str): reference
        Returns:
            boundaries (list): token boundaries

        """
        speed_rate = 1.
        if speaker[:5] in ['sp0.9', 'sp1.0', 'sp1.1']:
            speed_rate = float(speaker[2:5])
            speaker = '-'.join(speaker.split('-')[1:])
            utt_id = '-'.join(utt_id.split('-')[1:])

        alignment_path = os.path.join(alignment_dir, speaker, utt_id + '.txt')
        if not os.path.isfile(alignment_path):
            return None
        with codecs.open(alignment_path, 'r', encoding='utf-8') as f:
            word_alignments = deque([line.strip().split(' ') for line in f])

        # Remove space before the first special symbol
        wps = self.sp.EncodeAsPieces(text)
        if wps[0] == '▁' and wps[1][0] == '<':
            wps = wps[1:]

        boundaries = []
        wps_single_word = []
        ends = []
        for i, wp in enumerate(wps):
            if wp[0] == '▁':
                if i > 0:
                    word, start, end = word_alignments.popleft()
                    ends.append(end)
                    assert ''.join(wps_single_word) == word
                    start = float(start) * 100 * speed_rate
                    end = float(end) * 100 * speed_rate
                    assert start >= 0
                    assert end >= 0
                    boundaries += [start + (end - start) * len(''.join(wps_single_word[:j + 1])) / len(word)
                                   for j in range(len(wps_single_word))]
                    wps_single_word = []  # reset
                wp = wp[1:]  # remove word boundary mark
            wps_single_word.append(wp)
        # last word
        word, start, end = word_alignments.popleft()
        ends.append(end)
        assert ''.join(wps_single_word) == word
        start = float(start) * 100 * speed_rate
        end = float(end) * 100 * speed_rate
        assert start >= 0
        assert end >= 0
        boundaries += [start + (end - start) * len(''.join(wps_single_word[:j + 1])) / len(word)
                       for j in range(len(wps_single_word))]
        if len(boundaries) > 1:
            diff = np.array(boundaries[1:], dtype=np.int32) - np.array(boundaries[:-1], dtype=np.int32)
            assert (diff < 0).sum() == 0, (diff, boundaries, utt_id, speed_rate, ends)

        return np.ceil(np.array(boundaries)).astype(np.int32)


def load_ctc_alignment(alignment_dir, speaker, utt_id):
    """Load CTC alignment.

    Args:
        alignment_dir (str): path to CTC alignment directory
        speaker (str): speaker ID
        utt_id (str): utterance ID
    Returns:
        boundaries (list): token boundaries

    """
    alignment_path = os.path.join(alignment_dir, speaker, utt_id + '.txt')
    if not os.path.isfile(alignment_path):
        return None
    with codecs.open(alignment_path, 'r', encoding='utf-8') as f:
        boundaries = [int(line.strip().split(' ')[1]) for line in f]
    return np.array(boundaries, dtype=np.int32)
