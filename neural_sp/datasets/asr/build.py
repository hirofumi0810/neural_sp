# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for loading dataset for ASR.
   In this class, all data will be loaded at each step.
   You can use the multi-GPU version.
"""

import numpy as np

from neural_sp.datasets.asr.sampler import CustomBatchSampler
from neural_sp.datasets.asr.dataloader import CustomDataLoader
from neural_sp.datasets.asr.dataset import CustomDataset


def build_dataloader(args, tsv_path, batch_size, is_test=False,
                     sort_by='utt_id', short2long=False, sort_stop_epoch=1e10,
                     tsv_path_sub1=False, tsv_path_sub2=False,
                     num_workers=0, pin_memory=False,
                     first_n_utterances=-1, word_alignment_dir=None, ctc_alignment_dir=None,
                     longform_max_n_frames=0):

    dataset = CustomDataset(corpus=args.corpus,
                            tsv_path=tsv_path,
                            tsv_path_sub1=tsv_path_sub1,
                            tsv_path_sub2=tsv_path_sub2,
                            dict_path=args.dict,
                            dict_path_sub1=args.dict_sub1,
                            dict_path_sub2=args.dict_sub2,
                            nlsyms=args.nlsyms,
                            unit=args.unit,
                            unit_sub1=args.unit_sub1,
                            unit_sub2=args.unit_sub2,
                            wp_model=args.wp_model,
                            wp_model_sub1=args.wp_model_sub1,
                            wp_model_sub2=args.wp_model_sub2,
                            min_n_frames=args.min_n_frames,
                            max_n_frames=args.max_n_frames,
                            subsample_factor=args.subsample_factor,
                            subsample_factor_sub1=args.subsample_factor_sub1,
                            subsample_factor_sub2=args.subsample_factor_sub2,
                            ctc=args.ctc_weight > 0,
                            ctc_sub1=args.ctc_weight_sub1 > 0,
                            ctc_sub2=args.ctc_weight_sub2 > 0,
                            sort_by=sort_by,
                            short2long=short2long,
                            is_test=is_test,
                            first_n_utterances=first_n_utterances,
                            simulate_longform=longform_max_n_frames > 0,
                            word_alignment_dir=word_alignment_dir,
                            ctc_alignment_dir=ctc_alignment_dir)

    batch_sampler = CustomBatchSampler(df=dataset.df,  # filtered
                                       batch_size=batch_size,
                                       dynamic_batching=args.dynamic_batching,
                                       shuffle_bucket=args.shuffle_bucket and not is_test,
                                       sort_stop_epoch=args.sort_stop_epoch,
                                       discourse_aware=args.discourse_aware,
                                       longform_max_n_frames=longform_max_n_frames)

    dataloader = CustomDataLoader(dataset=dataset,
                                  batch_sampler=batch_sampler,
                                  collate_fn=custom_collate_fn,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    return dataloader


def custom_collate_fn(data):
    """Custom collate_fn.

    Args:
        data (List[dict]):
    Returns:
        data (List[dict]):

    """
    tmp = {k: [data_i[k] for data_i in data] for k in data[0].keys()}

    # reduce to a single utterance by concatenation
    if tmp['longform'][0]:
        tmp['xs'] = [np.concatenate(tmp['xs'], axis=0)]
        tmp['xlens'] = [sum(tmp['xlens'])]
        ys_cat = []
        for y in tmp['ys']:
            ys_cat += y
        tmp['ys'] = [ys_cat]
        tmp['text'] = [' '.join(tmp['text'])]

    # triggered points
    if tmp['trigger_points'][0] is None:
        tmp['trigger_points'] = None
    else:
        bs = len(tmp['ys'])
        ymax = max([len(tmp['ys'][i]) for i in range(bs)])
        trigger_points = np.zeros((bs, ymax + 1), dtype=np.int32)
        for b in range(bs):
            trigger_points[b, :len(tmp['trigger_points'][b])] = tmp['trigger_points'][b]

    return tmp
