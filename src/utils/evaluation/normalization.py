#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re


def normalize(trans, remove_tokens=[]):

    if len(remove_tokens) > 0:
        for token in remove_tokens:
            trans = trans.replace(token, '')

    # Remove consecutive spaces
    trans = re.sub(r'[_]+', '_', trans)

    trans = trans.replace('_>', '').replace('>', '')

    # Remove first and last space
    if len(trans) > 0:
        if trans[0] == '_':
            trans = trans[1:]

    if len(trans) > 0:
        if trans[-1] == '_':
            trans = trans[: -1]

    return trans


def normalize_swbd(trans, glm):
    """Fix transcription for Switchboard corpus.
    Args:
        trans (string):
    Returns:
        trans (string):
    """
    # Replace OOV temporaly
    trans = trans.replace('OOV', '#')

    # Remove noisy labels
    trans = normalize(trans, remove_tokens=['N', 'L', 'V', '>'])
    # LAUGHTER = 'L'
    # NOISE = 'N'
    # VOCALIZED_NOISE = 'V'

    if len(trans) == 0:
        return ''

    # Fix abbreviation, map all hesitations into a single class (%hesitation)
    trans = glm(trans)

    # Remove partial words ending in '-'
    # and split hyphen-divided words
    words = []
    for w in trans.split('_'):
        if w[-1] == '-':
            # Remove the last hyphen
            words += [w[:-1]]
        elif w[0] == '-':
            # Remove the fist hyphen
            words += [w[1:]]
        else:
            # Divide by hyphen
            words += w.split('-')
    trans = '_'.join(words)

    # TODO: Map acronyms
    trans = trans.replace('.', '')

    # Replace back OOV
    trans = trans.replace('#', 'OOV')

    return trans


class GLM(object):
    """GLM for Switchboard corpus."""

    def __init__(self, glm_path, space='_'):
        super(GLM, self).__init__()

        self.space = space

        self.map_dict = {}
        with open(glm_path, 'r')as f:
            for line in f:
                line = line.strip().lower()
                if len(line) == 0 or line[0] in [';', '*', '\'']:
                    continue

                line = line.split(' ;;')[0]
                line = line.replace(' / [ ] __ [ ]', '')
                line = re.sub(r'[_\[\]]+', '', line)
                line = re.sub(r'[\s]+', ' ', line)

                # Remove the last space
                if line[-1] == ' ':
                    line = line[:-1]

                if '{' in line:
                    # right to left
                    line = re.sub(r'[{}]+', '', line)
                    before, after = line.split(' => ')
                    for cand in after.split(' / '):
                        if before == cand:
                            continue
                        else:
                            self.map_dict[before] = cand
                            break
                        # print(cand + ' => ' + after)
                else:
                    # left to right
                    before, after = line.split(' => ')
                    self.map_dict[before] = after
                    # print(before + ' => ' + after)

    def __call__(self, trans):
        """
        Args:
            trans (string):
        Returns:
            trans (string):
        """
        # Fix abbreviation, hesitation based on GLM
        mapped_words = []
        words = trans.split(self.space)
        i = 0
        while True:
            if words[i] in self.map_dict.keys():
                word_fixed = self.map_dict[words[i]]
                mapped_words.extend(word_fixed.split(' '))
                # print('fixed: %s => %s' % (w, word_fixed))
                i += 1
            elif ' '.join(words[i:i + 2]) in self.map_dict.keys():
                word_fixed = self.map_dict[' '.join(words[i:i + 2])]
                mapped_words.extend(word_fixed.split(' '))
                # print('fixed: %s => %s' % (w, word_fixed))
                i += 2
            else:
                mapped_words.append(words[i])
                i += 1

            if i == len(words):
                break

        trans = self.space.join(mapped_words)
        return trans
