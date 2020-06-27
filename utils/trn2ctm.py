#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"Convert a trn file to a ctm file based on a stm segmentation."

import argparse
import codecs

parser = argparse.ArgumentParser()
parser.add_argument('trn', type=str,
                    help='trn file')
parser.add_argument('--stm', type=str, default='', nargs='?',
                    help='stm file (for Switchboard corpus)')
args = parser.parse_args()


def main():

    stm_segments = {}
    if args.stm:
        with codecs.open(args.stm, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if line[0] == ';':
                    continue

                speaker = line.split()[2]
                start_t = float(line.split()[3])
                end_t = float(line.split()[4])
                if speaker not in stm_segments.keys():
                    stm_segments[speaker] = {}
                stm_segments[speaker][start_t] = end_t

    with codecs.open(args.trn, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            words = line.split()[:-1]
            speaker_utt_id = line.split()[-1].replace('(', '').replace(')', '')
            speaker = speaker_utt_id.split('-')[0]
            utt_id = '-'.join(speaker_utt_id.split('-')[1:]).replace('-', '_')
            if args.stm:
                channel = speaker.split('-')[0].split('_')[-1]  # A or B
            else:
                channel = '1'
            start_f = utt_id.split('_')[-2]
            end_f = utt_id.split('_')[-1]
            start_t = round(int(start_f) / 100, 2)

            if args.stm:
                # Fix end time based on the stm file
                if start_t in stm_segments[speaker].keys():
                    end_t = stm_segments[speaker][start_t]
                else:
                    end_t = stm_segments[speaker][round(start_t + 0.01, 2)]
            else:
                end_t = round(int(end_f) / 100, 2)

            duration_t = end_t - start_t
            if len(words) > 0:
                duration_t /= len(words)

            confidence = 1  # Nist-1 manner in the ROVER paper

            speaker_no_channel = speaker.replace('_A', '').replace('_B', '')

            for w in words:
                print('%s %s %.2f %.2f %s %.3f' % (speaker_no_channel, channel, start_t, duration_t, w, confidence))
                start_t += duration_t


if __name__ == '__main__':
    main()
