#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for loading dataset for phoneme-to-word models.
   In this class, all data will be loaded at each step.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import basename, isfile, join
import pandas as pd
import codecs
import logging
from tqdm import tqdm
logger = logging.getLogger('training')

from src.dataset.base import Base
from src.utils.io.labels.word import Idx2word, Word2idx
from src.utils.io.labels.character import Idx2char, Char2idx
from src.utils.io.labels.phone import Idx2phone, Phone2idx
from src.utils.directory import mkdir_join


class Dataset(Base):

    def __init__(self, corpus, data_save_path, model_type,
                 data_size, data_type, label_type_in, label_type, label_type_sub,
                 batch_size, max_epoch=None,
                 max_frame_num=2000, min_frame_num=40,
                 shuffle=False, sort_utt=False, reverse=False,
                 sort_stop_epoch=None, tool='htk',
                 num_enque=None, dynamic_batching=False, vocab=False,
                 use_ctc=False, subsampling_factor=1,
                 use_ctc_sub=False, subsampling_factor_sub=1):
        """A class for loading dataset.
        Args:
            corpus (string): the name of corpus
            data_save_path (string): path to saved data
            model_type (string):
            data_size (string):
            data_type (string):
            label_type_in (string):
            label_type (string):
            label_type_sub (string):
            batch_size (int): the size of mini-batch
            max_epoch (int): the max epoch. None means infinite loop.
            max_frame_num (int): Exclude utteraces longer than this value
            min_frame_num (int): Exclude utteraces shorter than this value
            shuffle (bool): if True, shuffle utterances.
                This is disabled when sort_utt is True.
            sort_utt (bool): if True, sort all utterances in the ascending order
            reverse (bool): if True, sort utteraces in the descending order
            sort_stop_epoch (int): After sort_stop_epoch, training will revert
                back to a random order
            tool (string): htk or librosa or python_speech_features
            num_enque (int): the number of elements to enqueue
            dynamic_batching (bool): if True, batch size will be chainged
                dynamically in training
            vocab (bool or string):
            use_ctc (bool):
            subsampling_factor (int):
            use_ctc_sub (bool):
            subsampling_factor_sub (int):
        """
        self.corpus = corpus
        self.model_type = model_type
        self.data_type = data_type
        self.data_size = data_size
        self.label_type_in = label_type_in
        self.label_type = label_type
        self.label_type_sub = label_type_sub
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.shuffle = shuffle
        self.sort_utt = sort_utt
        self.sort_stop_epoch = sort_stop_epoch
        self.tool = tool
        self.num_enque = num_enque
        self.dynamic_batching = dynamic_batching

        # Corpus depending
        if corpus in ['csj', 'swbd', 'wsj']:
            self.is_test = True if 'eval' in data_type else False
        elif corpus in['librispeech', 'timit']:
            self.is_test = True if 'test' in data_type else False
        else:
            raise NotImplementedError

        # TODO: fix this
        if corpus == 'librispeech':
            if data_type == 'train':
                data_type += '_' + data_size

        self.vocab_file_path_in = join(
            data_save_path, 'vocab', data_size, label_type_in + '.txt')
        if vocab and data_size != '' and data_size != vocab:
            self.vocab_file_path = join(
                data_save_path, 'vocab', vocab, label_type + '.txt')
            self.vocab_file_path_sub = join(
                data_save_path, 'vocab', vocab, label_type_sub + '.txt')
            vocab_file_path_org = join(
                data_save_path, 'vocab', data_size, label_type + '.txt')
        else:
            self.vocab_file_path = join(
                data_save_path, 'vocab', data_size, label_type + '.txt')
            self.vocab_file_path_sub = join(
                data_save_path, 'vocab', data_size, label_type_sub + '.txt')

        self.idx2phone = Idx2phone(self.vocab_file_path_in)
        self.phone2idx = Phone2idx(self.vocab_file_path_in)
        assert 'phone' in label_type_in

        # main task
        if label_type == 'word':
            self.idx2word = Idx2word(self.vocab_file_path)
            self.word2idx = Word2idx(self.vocab_file_path)
        elif 'character' in label_type:
            self.idx2char = Idx2char(self.vocab_file_path)
            self.char2idx = Char2idx(self.vocab_file_path)
        else:
            raise ValueError(label_type)

        # sub task
        if 'character' in label_type_sub:
            self.idx2char = Idx2char(self.vocab_file_path_sub)
            self.char2idx = Char2idx(self.vocab_file_path_sub)
        elif 'phone' in label_type_sub:
            self.idx2phone = Idx2phone(self.vocab_file_path_sub)
            self.phone2idx = Phone2idx(self.vocab_file_path_sub)
        else:
            raise ValueError(label_type_sub)

        super(Dataset, self).__init__(vocab_file_path_in=self.vocab_file_path_in,
                                      vocab_file_path=self.vocab_file_path,
                                      vocab_file_path_sub=self.vocab_file_path_sub)

        # Load dataset file
        if vocab and data_size != '' and data_size != vocab and not self.is_test:
            dataset_path_in = mkdir_join(
                data_save_path, 'dataset', tool, data_size, data_type, label_type_in + '.csv')
            dataset_path = mkdir_join(
                data_save_path, 'dataset', tool, data_size + '_' + vocab, data_type, label_type + '.csv')
            dataset_path_sub = mkdir_join(
                data_save_path, 'dataset', tool, data_size + '_' + vocab, data_type, label_type_sub + '.csv')

            # Change token indices
            if not isfile(dataset_path):
                dataset_path_org = join(
                    data_save_path, 'dataset', tool, data_size, data_type, label_type + '.csv')
                df = pd.read_csv(dataset_path_org, encoding='utf-8')
                df = df.loc[:, ['frame_num', 'input_path', 'transcript']]

                # Change vocabulary
                org2new = {}
                str2idx_org = {}
                str2idx_new = {}
                # new vocab
                with codecs.open(self.vocab_file_path, 'r', 'utf-8') as f:
                    vocab_count = 0
                    for line in f:
                        if line.strip() != '':
                            str2idx_new[line.strip()] = vocab_count
                            vocab_count += 1
                # original vocab
                with codecs.open(vocab_file_path_org, 'r', 'utf-8') as f:
                    vocab_count = 0
                    for line in f:
                        if line.strip() != '':
                            str2idx_org[line.strip()] = vocab_count
                            vocab_count += 1
                for k, v in str2idx_org.items():
                    if k in str2idx_new.keys():
                        org2new[v] = str2idx_new[k]
                    else:
                        org2new[v] = str2idx_new['OOV']

                # Update the transcript
                for i in tqdm(df['transcript'].index):
                    df['transcript'][i] = ' '.join(
                        list(map(lambda x: str(org2new[int(x)]), df['transcript'][i].split(' '))))

                # Save as a new file
                df.to_csv(dataset_path_in, encoding='utf-8')
            else:
                df_in = pd.read_csv(dataset_path_in, encoding='utf-8')
                df_in = df_in.loc[:, ['frame_num', 'input_path', 'transcript']]
                df = pd.read_csv(dataset_path, encoding='utf-8')
                df = df.loc[:, ['frame_num', 'input_path', 'transcript']]

            if not isfile(dataset_path_sub):
                raise NotImplementedError
        else:
            dataset_path_in = join(
                data_save_path, 'dataset', tool, data_size, data_type, label_type_in + '.csv')
            dataset_path = join(
                data_save_path, 'dataset', tool, data_size, data_type, label_type + '.csv')
            dataset_path_sub = join(
                data_save_path, 'dataset', tool, data_size, data_type, label_type_sub + '.csv')
            df_in = pd.read_csv(dataset_path_in, encoding='utf-8')
            df_in = df_in.loc[:, ['frame_num', 'input_path', 'transcript']]
            df = pd.read_csv(dataset_path, encoding='utf-8')
            df = df.loc[:, ['frame_num', 'input_path', 'transcript']]
            df_sub = pd.read_csv(dataset_path_sub, encoding='utf-8')
            df_sub = df_sub.loc[:, ['frame_num', 'input_path', 'transcript']]

        # Remove inappropriate utteraces
        if not self.is_test:
            print('Original utterance num (input): %d' % len(df_in))
            print('Original utterance num (output, main): %d' % len(df))
            print('Original utterance num (output, sub): %d' % len(df_sub))
            utt_num_orig_in = len(df_in)
            utt_num_orig = len(df)
            utt_num_orig_sub = len(df_sub)

            # For Switchboard
            if corpus == 'swbd' and 'train' in data_type:
                df_in = df_in[df_in.apply(lambda x: not(len(x['transcript'].split(' '))
                                                        <= 3 and x['frame_num'] >= 1000), axis=1)]
                df = df[df.apply(lambda x: not(len(x['transcript'].split(' '))
                                               <= 24 and x['frame_num'] >= 1000), axis=1)]
                df_sub = df_sub[df_sub.apply(lambda x: not(len(x['transcript'].split(' '))
                                                           <= 24 and x['frame_num'] >= 1000), axis=1)]

            # Remove by threshold
            df_in = df_in[df_in.apply(
                lambda x: min_frame_num <= x['frame_num'] <= max_frame_num, axis=1)]
            df = df[df.apply(
                lambda x: min_frame_num <= x['frame_num'] <= max_frame_num, axis=1)]
            df_sub = df_sub[df_sub.apply(
                lambda x: min_frame_num <= x['frame_num'] <= max_frame_num, axis=1)]
            print('Removed utterance num (threshold, input): %d' %
                  (utt_num_orig_in - len(df_in)))
            print('Removed utterance num (threshold, output, main): %d' %
                  (utt_num_orig - len(df)))
            print('Removed utterance num (threshold, output, sub): %d' %
                  (utt_num_orig_sub - len(df_sub)))

            # Remove for CTC loss calculatioon
            if use_ctc and subsampling_factor > 1:
                pass
            if use_ctc_sub and subsampling_factor_sub > 1:
                pass

            # Make up the number
            if not (len(df_in) == len(df) == len(df)):
                diff = df_in.index.difference(df.index)
                df_in = df_in.drop(diff)
                diff = df.index.difference(df_in.index)
                df = df.drop(diff)
                assert len(df_in) == len(df)

                diff = df_in.index.difference(df_sub.index)
                df_in = df_in.drop(diff)
                diff = df_sub.index.difference(df_in.index)
                df_sub = df_sub.drop(diff)
                assert len(df_in) == len(df_sub)

                diff = df.index.difference(df_sub.index)
                df = df.drop(diff)
                diff = df_sub.index.difference(df.index)
                df_sub = df_sub.drop(diff)
                assert len(df) == len(df_sub)

        # Sort paths to input & label
        if sort_utt:
            df_in = df_in.sort_values(by='frame_num', ascending=not reverse)
            df = df.sort_values(by='frame_num', ascending=not reverse)
            df_sub = df_sub.sort_values(by='frame_num', ascending=not reverse)
        else:
            df_in = df_in.sort_values(by='input_path', ascending=True)
            df = df.sort_values(by='input_path', ascending=True)
            df_sub = df_sub.sort_values(by='input_path', ascending=True)

        assert len(df_in) == len(df) == len(df_sub)

        self.df_in = df_in
        self.df = df
        self.df_sub = df_sub
        self.rest = set(list(df_in.index))

    def select_batch_size(self, batch_size, min_frame_num_batch):
        return batch_size

    def make_batch(self, data_indices):
        """Create mini-batch per step.
        Args:
            data_indices (np.ndarray):
        Returns:
            batch (dict):
                xs (list): feature labels of size `[B, L_in]`
                ys (list): target labels in the main task of size `[B, L]`
                ys_sub (list): target labels in the sub task of size `[B, L_sub]`
                input_names (list): file names of input data of size `[B]`
        """
        if self.is_test:
            xs = [self.df_in['transcript'][data_indices[b]]
                  for b in range(len(data_indices))]
            ys = [self.df['transcript'][data_indices[b]]
                  for b in range(len(data_indices))]
            ys_sub = [self.df_sub['transcript'][data_indices[b]]
                      for b in range(len(data_indices))]
            # NOTE: transcript is not tokenized
        else:
            xs = [list(map(int, self.df_in['transcript'][i].split(' ')))
                  for i in data_indices]
            ys = [list(map(int, self.df['transcript'][i].split(' ')))
                  for i in data_indices]
            ys_sub = [list(map(int, self.df_sub['transcript'][i].split(' ')))
                      for i in data_indices]

        # TODO: fix later
        try:
            input_names = list(
                map(lambda path: basename(path).split('.')[0],
                    self.df['input_path'][data_indices]))
        except:
            input_names = self.df.index.values.tolist()

        return {'xs': xs, 'ys': ys, 'ys_sub': ys_sub, 'input_names': input_names}
