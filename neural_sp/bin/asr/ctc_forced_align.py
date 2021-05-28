#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conduct forced alignment with pre-trained CTC model."""

import codecs
import logging
import os
import shutil
import sys
from tqdm import tqdm

from neural_sp.bin.args_asr import parse_args_eval
from neural_sp.bin.eval_utils import average_checkpoints
from neural_sp.bin.train_utils import set_logger
from neural_sp.datasets.asr.build import build_dataloader
from neural_sp.models.seq2seq.speech2text import Speech2Text
from neural_sp.utils import mkdir_join

logger = logging.getLogger(__name__)


def main():

    # Load configuration
    args, dir_name = parse_args_eval(sys.argv[1:])

    # Setting for logging
    if os.path.isfile(os.path.join(args.recog_dir, 'align.log')):
        os.remove(os.path.join(args.recog_dir, 'align.log'))
    set_logger(os.path.join(args.recog_dir, 'align.log'), stdout=args.recog_stdout)

    # Load ASR model
    model = Speech2Text(args, dir_name)
    average_checkpoints(model, args.recog_model[0], n_average=args.recog_n_average)

    if not args.recog_unit:
        args.recog_unit = args.unit

    logger.info('recog unit: %s' % args.recog_unit)
    logger.info('batch size: %d' % args.recog_batch_size)

    # GPU setting
    if args.recog_n_gpus >= 1:
        model.cudnn_setting(deterministic=True, benchmark=False)
        model.cuda()

    for s in args.recog_sets:
        # Align all utterances
        args.min_n_frames = 0
        args.max_n_frames = 1e5

        # Load dataloader
        dataloader = build_dataloader(args=args,
                                      tsv_path=s,
                                      batch_size=args.recog_batch_size)

        save_path = mkdir_join(args.recog_dir, 'ctc_forced_alignments')

        # Clean directory
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)

        pbar = tqdm(total=len(dataloader))
        for batch in dataloader:
            trigger_points = model.ctc_forced_align(batch['xs'], batch['ys'])  # `[B, L]`

            for b in range(len(batch['xs'])):
                save_path_spk = mkdir_join(save_path, batch['speakers'][b])
                save_path_utt = mkdir_join(save_path_spk, batch['utt_ids'][b] + '.txt')

                tokens = dataloader.idx2token[0](batch['ys'][b], return_list=True)
                with codecs.open(save_path_utt, 'w', encoding="utf-8") as f:
                    for i_tok, tok in enumerate(tokens):
                        f.write('%s %d\n' % (tok, trigger_points[b, i_tok]))
                    f.write('%s %d\n' % ('<eos>', trigger_points[b, len(tokens)]))

            pbar.update(len(batch['xs']))

        pbar.close()


if __name__ == '__main__':
    main()
