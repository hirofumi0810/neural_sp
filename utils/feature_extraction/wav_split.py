#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Split a WAV file into each utterance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import basename, join
import numpy as np
import wave
from tqdm import tqdm
import pickle

from utils.util import mkdir_join


def split_wav(wav_paths, save_path, speaker_dict):
    """Read WAV files & divide them with respect to each utterance.
    Args:
        wav_paths (list): path to WAV files
        save_path (string): path to save WAV files
        speaker_dict (dict): the dictionary of utterances of each speaker
            key => speaker
            value => the dictionary of utterance information of each speaker
                key => utterance index
                value => [start_frame, end_frame, transcript]
    """
    # Read each WAV file
    print('==> Reading WAV files...')
    for wav_path in tqdm(wav_paths):
        speaker = basename(wav_path).split('.')[0]

        # NOTE: For Switchboard
        speaker = speaker.replace('sw0', 'sw')
        speaker = speaker.replace('sw_', 'sw')
        speaker = speaker.replace('en_', 'en')

        if 'subject' in speaker:
            speaker = '_'.join(speaker.split('_')[:2]) + '_U'
        elif 'operator' in speaker:
            speaker = '_'.join(speaker.split('_')[:2]) + '_S'

        utt_dict = speaker_dict[speaker]
        wav_utt_save_path = mkdir_join(save_path, speaker)

        # Read a wav file
        audio = Audio(file_path=wav_path)
        audio_data = audio.read()

        # Split per utterance & save as wav files
        audio.split(audio_data, utt_dict, speaker,
                    save_path=wav_utt_save_path)

    # Save the frame number dictionary
    with open(join(save_path, 'frame_num.pickle'), 'wb') as f:
        pickle.dump(audio.frame_num_dict, f)


class Audio(object):
    def __init__(self, file_path):
        """Audio Class.
        Args:
            file_path: file name of a WAV file
        """
        self.file_path = file_path
        self.filename = file_path.split('/')[-1]
        self.frame_num_dict = {}

    def read(self):
        """Return audio file as array of integer.
        Returns:
            audio_data: np.ndarray, shape of (frame_num,)
        """
        # Read wav file
        with wave.open(self.file_path, "r") as wav:
            # Move to head of the audio file
            wav.rewind()

            self.frame_num = wav.getnframes()
            self.sampling_rate = wav.getframerate()  # 16,000 Hz
            self.channels = wav.getnchannels()
            self.sample_size = wav.getsampwidth()  # 2

            # Read to buffer as binary format
            buf = wav.readframes(self.frame_num)

        if self.channels == 1:
            audio_data = np.frombuffer(buf, dtype="int16")
        elif self.channels == 2:
            audio_data = np.frombuffer(buf, dtype="int32")

        return audio_data

    def split(self, audio_data, utterance_dict, speaker, save_path):
        """
        Args:
            audio_data:
            utterance_dict: the dictionary of utterance information of each speaker
                key => utterance index
                value => [start_frame, end_frame, transcript]
            speaker:
            save_path: path to save each WAV file
        """
        for utt_index, utt_info in sorted(utterance_dict.items(),
                                          key=lambda x: x[0]):
            start_frame, end_frame = utt_info[:2]
            start_frame = int((start_frame / 100) * self.sampling_rate)
            end_frame = int((end_frame / 100) * self.sampling_rate)
            audio_data_split = audio_data[start_frame:end_frame]

            self.frame_num_dict[speaker + '_' +
                                utt_index] = audio_data_split.shape[0]

            with wave.Wave_write(
                    join(save_path, speaker + '_' + str(utt_index) + ".wav")) as w:
                w.setnchannels(self.channels)
                w.setsampwidth(self.sample_size)
                w.setframerate(self.sampling_rate)
                w.writeframes(audio_data_split)
