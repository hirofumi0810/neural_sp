#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions for phone mapping."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Map2phone39(object):
    """Map from 61 or 48 phones to 39 phones.
    Args:
        label_type (string): phone48 or phone61
        map_file_path: path to the mapping file
    """

    def __init__(self, label_type, map_file_path):
        self.label_type = label_type

        # Read the mapping file
        self.map_dict = {}
        with open(map_file_path) as f:
            for line in f:
                line = line.strip().split()
                if label_type == 'phone61':
                    if line[1] != 'nan':
                        self.map_dict[line[0]] = line[2]
                    else:
                        self.map_dict[line[0]] = ''
                elif label_type == 'phone48':
                    if line[1] != 'nan':
                        self.map_dict[line[1]] = line[2]

    def __call__(self, phone_list):
        """
        Args:
            phone_list (list): list of phones (string)
        Returns:
            phone_list (list): list of 39 phones (string)
        """
        if self.label_type == 'phone39':
            return phone_list

        if len(phone_list) == 0:
            return phone_list

        # Map to 39 phones
        for i in range(len(phone_list)):
            phone_list[i] = self.map_dict[phone_list[i]]

        # Ignore q (only if 61 phones)
        while '' in phone_list:
            phone_list.remove('')

        return phone_list
