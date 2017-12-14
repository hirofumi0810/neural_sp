
"""Base class for loading dataset for the CTC and attention-based model.
   In this class, all data will be loaded at each step.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import basename
import math
import numpy as np

from utils.dataset.base import Base
from utils.io.inputs.frame_stacking import stack_frame
from utils.io.inputs.splicing import do_splice

# NOTE: Loading numpy is faster than loading htk


class DatasetBase(Base):

    def __init__(self, *args, **kwargs):
        super(DatasetBase, self).__init__(*args, **kwargs)

    def make_batch(self, data_indices):
        """
        Args:
            data_indices (np.ndarray):
        Returns:
            inputs: list of input data of size
                `[num_gpus, B, T_in, input_size]`
            labels: list of target labels of size
                `[num_gpus, B, T_out]`
            inputs_seq_len: list of length of inputs of size
                `[num_gpus, B]`
            labels_seq_len: list of length of target labels of size
                `[num_gpus, B]`
            input_names: list of file name of input data of size
                `[num_gpus, B]`
        """
        input_path_list = np.array(self.df['input_path'][data_indices])
        str_indices_list = np.array(self.df['transcript'][data_indices])

        if not hasattr(self, 'input_size'):
            self.input_size = self.input_channel * \
                (1 + int(self.use_delta) + int(self.use_double_delta))
            self.input_size *= self.num_stack
            self.input_size *= self.splice

        # Compute max frame num in mini-batch
        max_frame_num = max(self.df['frame_num'][data_indices])
        max_frame_num = math.ceil(max_frame_num / self.num_skip)

        # Compute max target label length in mini-batch
        max_seq_len = max(
            map(lambda x: len(x.split(' ')), str_indices_list)) + 2
        # NOTE: add <SOS> and <EOS>

        # Initialization
        inputs = np.zeros(
            (len(data_indices), max_frame_num, self.input_size),
            dtype=np.float32)
        labels = np.array(
            [[self.pad_value] * max_seq_len] * len(data_indices))
        inputs_seq_len = np.zeros((len(data_indices),), dtype=np.int32)
        labels_seq_len = np.zeros((len(data_indices),), dtype=np.int32)
        input_names = np.array(list(
            map(lambda path: basename(path).split('.')[0],
                np.array(self.df['input_path'][data_indices]))))

        # Set values of each data in mini-batch
        for i_batch in range(len(data_indices)):
            # Load input data
            if self.save_format == 'numpy':
                data_i_tmp = self.load_npy(input_path_list[i_batch])
            elif self.save_format == 'htk':
                data_i_tmp = self.load_htk(input_path_list[i_batch])
            else:
                raise TypeError

            if self.use_double_delta:
                data_i = data_i_tmp
            elif self.use_delta:
                data_i = np.concatenate(
                    (data_i, data_i_tmp[:self.input_channel * 2]), axis=0)
            else:
                data_i = data_i_tmp[:self.input_channel]

            # Frame stacking
            data_i = stack_frame(data_i, self.num_stack, self.num_skip)
            frame_num = data_i.shape[0]

            # Splicing
            data_i = do_splice(data_i, self.splice, self.num_stack)

            inputs[i_batch, : frame_num, :] = data_i
            inputs_seq_len[i_batch] = frame_num
            if self.is_test:
                labels[i_batch, 0] = self.df['transcript'][data_indices[i_batch]]
                # NOTE: transcript is not tokenized
            else:
                indices = list(map(int, str_indices_list[i_batch].split(' ')))
                label_num = len(indices)
                if self.model_type == 'attention':
                    labels[i_batch, 0] = self.sos_index
                    labels[i_batch, 1:label_num + 1] = indices
                    labels[i_batch, label_num + 1] = self.eos_index
                    labels_seq_len[i_batch] = label_num + 2
                    # NOTE: include <SOS> and <EOS>
                elif self.model_type == 'ctc':
                    labels[i_batch, 0:label_num] = indices
                    labels_seq_len[i_batch] = label_num
                else:
                    raise TypeError

        # Now we split the mini-batch data by num_gpus
        # inputs = self.split_per_device(inputs, self.num_gpus)
        # labels = self.split_per_device(labels, self.num_gpus)
        # inputs_seq_len = self.split_per_device(inputs_seq_len, self.num_gpus)
        # labels_seq_len = self.split_per_device(labels_seq_len, self.num_gpus)
        # input_names = self.split_per_device(input_names, self.num_gpus)

        return inputs, labels, inputs_seq_len, labels_seq_len, input_names
