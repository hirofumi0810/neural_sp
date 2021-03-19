# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Custom Dataset."""

import kaldiio
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset

from neural_sp.datasets.token_converter.character import (
    Char2idx,
    Idx2char,
)
from neural_sp.datasets.token_converter.phone import (
    Idx2phone,
    Phone2idx,
)
from neural_sp.datasets.token_converter.word import (
    Idx2word,
    Word2idx,
)
from neural_sp.datasets.token_converter.wordpiece import (
    Idx2wp,
    Wp2idx,
)
from neural_sp.datasets.alignment import (
    load_ctc_alignment,
    WordAlignmentConverter,
)
from neural_sp.datasets.utils import count_vocab_size


class CustomDataset(Dataset):

    def __init__(self, corpus, tsv_path, tsv_path_sub1, tsv_path_sub2,
                 dict_path, dict_path_sub1, dict_path_sub2, nlsyms,
                 unit, unit_sub1, unit_sub2,
                 wp_model, wp_model_sub1, wp_model_sub2,
                 min_n_frames, max_n_frames,
                 subsample_factor, subsample_factor_sub1, subsample_factor_sub2,
                 ctc, ctc_sub1, ctc_sub2,
                 sort_by, short2long, is_test,
                 discourse_aware=False, simulate_longform=False, first_n_utterances=-1,
                 word_alignment_dir=None, ctc_alignment_dir=None):
        """Custom Dataset class.

        Args:
            corpus (str): name of corpus
            tsv_path (str): path to the dataset tsv file
            dict_path (str): path to the dictionary
            nlsyms (str): path to the non-linguistic symbols file
            unit (str): word/wp/char/phone/word_char
            wp_model (): path to the word-piece model for sentencepiece
            min_n_frames (int): exclude utterances shorter than this value
            max_n_frames (int): exclude utterances longer than this value
            subsample_factor (int):
            ctc (bool):
            sort_by (str): sort all utterances in the ascending order
                input: sort by input length
                output: sort by output length
                shuffle: shuffle all utterances
            short2long (bool): sort utterances in the descending order
            is_test (bool):
            discourse_aware (bool): sort in the discourse order
            simulate_longform (bool): simulate long-form uttterance
            first_n_utterances (int): evaluate the first N utterances
            word_alignment_dir (str): path to word alignment directory
            ctc_alignment_dir (str): path to CTC alignment directory

        """
        super(Dataset, self).__init__()

        np.random.seed(1)

        # meta deta accessed by dataloader
        self._corpus = corpus
        self._set = os.path.basename(tsv_path).split('.')[0]
        self._vocab = count_vocab_size(dict_path)
        self._unit = unit
        self._unit_sub1 = unit_sub1
        self._unit_sub2 = unit_sub2

        self.is_test = is_test
        self.sort_by = sort_by
        # if shuffle_bucket:
        #     assert sort_by in ['input', 'output']
        if discourse_aware:
            assert not is_test
        if simulate_longform:
            assert is_test
        self.simulate_longform = simulate_longform

        self.subsample_factor = subsample_factor
        self.word_alignment_dir = word_alignment_dir
        self.ctc_alignment_dir = ctc_alignment_dir

        self._idx2token = []
        self._token2idx = []

        # Set index converter
        if unit in ['word', 'word_char']:
            self._idx2token += [Idx2word(dict_path)]
            self._token2idx += [Word2idx(dict_path, word_char_mix=(unit == 'word_char'))]
        elif unit == 'wp':
            self._idx2token += [Idx2wp(dict_path, wp_model)]
            self._token2idx += [Wp2idx(dict_path, wp_model)]
        elif unit in ['char']:
            self._idx2token += [Idx2char(dict_path)]
            self._token2idx += [Char2idx(dict_path, nlsyms=nlsyms)]
        elif 'phone' in unit:
            self._idx2token += [Idx2phone(dict_path)]
            self._token2idx += [Phone2idx(dict_path)]
        else:
            raise ValueError(unit)

        for i in range(1, 3):
            dict_path_sub = locals()['dict_path_sub' + str(i)]
            wp_model_sub = locals()['wp_model_sub' + str(i)]
            unit_sub = locals()['unit_sub' + str(i)]
            if dict_path_sub:
                setattr(self, '_vocab_sub' + str(i), count_vocab_size(dict_path_sub))

                # Set index converter
                if unit_sub:
                    if unit_sub == 'wp':
                        self._idx2token += [Idx2wp(dict_path_sub, wp_model_sub)]
                        self._token2idx += [Wp2idx(dict_path_sub, wp_model_sub)]
                    elif unit_sub == 'char':
                        self._idx2token += [Idx2char(dict_path_sub)]
                        self._token2idx += [Char2idx(dict_path_sub, nlsyms=nlsyms)]
                    elif 'phone' in unit_sub:
                        self._idx2token += [Idx2phone(dict_path_sub)]
                        self._token2idx += [Phone2idx(dict_path_sub)]
                    else:
                        raise ValueError(unit_sub)
            else:
                setattr(self, '_vocab_sub' + str(i), -1)

        # Load dataset tsv file
        df = pd.read_csv(tsv_path, encoding='utf-8', delimiter='\t')
        df = df.loc[:, ['utt_id', 'speaker', 'feat_path',
                        'xlen', 'xdim', 'text', 'token_id', 'ylen', 'ydim']]
        for i in range(1, 3):
            if locals()['tsv_path_sub' + str(i)]:
                df_sub = pd.read_csv(locals()['tsv_path_sub' + str(i)], encoding='utf-8', delimiter='\t')
                df_sub = df_sub.loc[:, ['utt_id', 'speaker', 'feat_path',
                                        'xlen', 'xdim', 'text', 'token_id', 'ylen', 'ydim']]
                setattr(self, 'df_sub' + str(i), df_sub)
            else:
                setattr(self, 'df_sub' + str(i), None)

        self._input_dim = kaldiio.load_mat(df['feat_path'][0]).shape[-1]

        # Remove inappropriate utterances
        print(f"Original utterance num: {len(df)}")
        n_utts = len(df)
        if is_test or discourse_aware:
            df = df[df.apply(lambda x: x['ylen'] > 0, axis=1)]
            print(f"Removed {n_utts - len(df)} empty utterances")
            if first_n_utterances > 0:
                n_utts = len(df)
                df = df[df.apply(lambda x: x['ylen'] > 0, axis=1)]
                df = df.truncate(before=0, after=first_n_utterances - 1)
                print(f"Select first {len(df)} utterances")
        else:
            df = df[df.apply(lambda x: min_n_frames <= x[
                'xlen'] <= max_n_frames, axis=1)]
            df = df[df.apply(lambda x: x['ylen'] > 0, axis=1)]
            print(f"Removed {n_utts - len(df)} utterances (threshold)")

            if ctc and subsample_factor > 1:
                n_utts = len(df)
                df = df[df.apply(lambda x: x['ylen'] <= (x['xlen'] // subsample_factor), axis=1)]
                print(f"Removed {n_utts - len(df)} utterances (for CTC)")

            for i in range(1, 3):
                df_sub = getattr(self, 'df_sub' + str(i))
                ctc_sub = locals()['ctc_sub' + str(i)]
                subsample_factor_sub = locals()['subsample_factor_sub' + str(i)]
                if df_sub is not None:
                    if ctc_sub and subsample_factor_sub > 1:
                        df_sub = df_sub[df_sub.apply(
                            lambda x: x['ylen'] <= (x['xlen'] // subsample_factor_sub), axis=1)]

                    if len(df) != len(df_sub):
                        n_utts = len(df)
                        df = df.drop(df.index.difference(df_sub.index))
                        print(f"Removed {n_utts - len(df)} utterances (for CTC, sub{i})")
                        for j in range(1, i + 1):
                            setattr(self, 'df_sub' + str(j),
                                    getattr(self, 'df_sub' + str(j)).drop(getattr(self, 'df_sub' + str(j)).index.difference(df.index)))

        if corpus == 'swbd':
            # 1. serialize
            # df['session'] = df['speaker'].apply(lambda x: str(x).split('-')[0])
            # 2. not serialize
            df['session'] = df['speaker'].apply(lambda x: str(x))
        else:
            df['session'] = df['speaker'].apply(lambda x: str(x))

        # Sort tsv records
        if discourse_aware:
            # Sort by onset (start time)
            df = df.assign(prev_utt='')
            df = df.assign(line_no=list(range(len(df))))
            if corpus == 'swbd':
                df['onset'] = df['utt_id'].apply(lambda x: int(x.split('_')[-1].split('-')[0]))
            elif corpus == 'csj':
                df['onset'] = df['utt_id'].apply(lambda x: int(x.split('_')[1]))
            elif corpus == 'tedlium2':
                df['onset'] = df['utt_id'].apply(lambda x: int(x.split('-')[-2]))
            else:
                raise NotImplementedError(corpus)
            df = df.sort_values(by=['session', 'onset'], ascending=True)

            # Extract previous utterances
            groups = df.groupby('session').groups
            df['prev_utt'] = df.apply(
                lambda x: [df.loc[i, 'line_no']
                           for i in groups[x['session']] if df.loc[i, 'onset'] < x['onset']], axis=1)
            df['n_prev_utt'] = df.apply(lambda x: len(x['prev_utt']), axis=1)
            df['n_utt_in_session'] = df.apply(
                lambda x: len([i for i in groups[x['session']]]), axis=1)
            df = df.sort_values(by=['n_utt_in_session'], ascending=short2long)

            # NOTE: this is used only when LM is trained with serialize: true
            # if is_test and corpus == 'swbd':
            #     # Sort by onset
            #     df['onset'] = df['utt_id'].apply(lambda x: int(x.split('_')[-1].split('-')[0]))
            #     df = df.sort_values(by=['session', 'onset'], ascending=True)

        elif not is_test:
            if sort_by == 'input':
                df = df.sort_values(by=['xlen'], ascending=short2long)
            elif sort_by == 'output':
                df = df.sort_values(by=['ylen'], ascending=short2long)
            elif sort_by == 'shuffle':
                df = df.reindex(np.random.permutation(self.df.index))

        # Fit word alignment to vocabulary
        if word_alignment_dir is not None:
            alignment2boundary = WordAlignmentConverter(dict_path, wp_model)
            n_utts = len(df)
            df['trigger_points'] = df.apply(lambda x: alignment2boundary(
                word_alignment_dir, x['speaker'], x['utt_id'], x['text']), axis=1)
            # remove utterances which do not have the alignment
            df = df[df.apply(lambda x: x['trigger_points'] is not None, axis=1)]
            print(f"Removed {n_utts - len(df)} utterances (for word alignment)")
        elif ctc_alignment_dir is not None:
            n_utts = len(df)
            df['trigger_points'] = df.apply(lambda x: load_ctc_alignment(
                ctc_alignment_dir, x['speaker'], x['utt_id']), axis=1)
            # remove utterances which do not have the alignment
            df = df[df.apply(lambda x: x['trigger_points'] is not None, axis=1)]
            print(f"Removed {n_utts - len(df)} utterances (for CTC alignment)")

        # Re-indexing
        if discourse_aware:
            self.df = df
            for i in range(1, 3):
                if getattr(self, 'df_sub' + str(i)) is not None:
                    setattr(self, 'df_sub' + str(i),
                            getattr(self, 'df_sub' + str(i)).reindex(df.index))
        else:
            self.df = df.reset_index()
            for i in range(1, 3):
                if getattr(self, 'df_sub' + str(i)) is not None:
                    setattr(self, 'df_sub' + str(i),
                            getattr(self, 'df_sub' + str(i)).reindex(df.index).reset_index())

    def __len__(self):
        return len(self.df)

    @property
    def n_frames(self):
        return self.df['xlen'].sum()

    def __getitem__(self, i):
        """Create mini-batch per step.

        Args:
            indices (int): indices of dataframe in the current mini-batch
        Returns:
            mini_batch_dict (dict):
                xs (List): input data of size `[T, input_dim]`
                xlens (List): lengths of xs
                ys (List): reference labels in the main task of size `[L]`
                ys_sub1 (List): reference labels in the 1st auxiliary task of size `[L_sub1]`
                ys_sub2 (List): reference labels in the 2nd auxiliary task of size `[L_sub2]`
                utt_ids (List): name of each utterance
                speakers (List): name of each speaker
                sessions (List): name of each session

        """
        # inputs
        feat_path = self.df['feat_path'][i]
        xs = kaldiio.load_mat(feat_path)
        xlen = self.df['xlen'][i]

        # external alignment
        trigger_points = None
        if self.word_alignment_dir is not None:
            trigger_points = np.zeros(self.df['ylen'][i] + 1, dtype=np.int32)
            p = self.df['trigger_points'][i]
            trigger_points[:len(p)] = p - 1  # 0-indexed
            # speacial treatment for the last token
            trigger_points[len(p) - 1] = min(trigger_points[len(p) - 1], xlen - 1)
            # NOTE: <eos> is not treated here
            assert trigger_points.max() <= xlen - 1, (p, xlen)
            trigger_points //= self.subsample_factor
        elif self.ctc_alignment_dir is not None:
            trigger_points = np.zeros((1, self.df['ylen'][i] + 1), dtype=np.int32)
            p = self.df['trigger_points'][i]  # including <eos>
            trigger_points[:len(p)] = p  # already 0-indexed

        # main outputs
        text = self.df['text'][i]
        if self.is_test:
            ys = self._token2idx[0](text)
        else:
            ys = list(map(int, str(self.df['token_id'][i]).split()))

        # sub1 outputs
        ys_sub1 = []
        if self.df_sub1 is not None:
            ys_sub1 = list(map(int, str(self.df_sub1['token_id'][i]).split()))
        elif self._vocab_sub1 > 0 and not self.is_test:
            ys_sub1 = self._token2idx[1](text)

        # sub2 outputs
        ys_sub2 = []
        if self.df_sub2 is not None:
            ys_sub2 = list(map(int, str(self.df_sub2['token_id'][i]).split()))
        elif self._vocab_sub2 > 0 and not self.is_test:
            ys_sub2 = self._token2idx[2](text)

        mini_batch_dict = {
            'xs': xs,
            'xlens': xlen,
            'ys': ys,
            'ys_sub1': ys_sub1,
            'ys_sub2': ys_sub2,
            'utt_ids': self.df['utt_id'][i],
            'speakers': self.df['speaker'][i],
            'sessions': self.df['session'][i],
            'text': text,
            'feat_path': feat_path,  # for plot
            'trigger_points': trigger_points,
            'longform': self.simulate_longform,
        }

        return mini_batch_dict
