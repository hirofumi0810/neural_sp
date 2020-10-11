#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conduct forced alignment with the pre-trained CTC model."""

import codecs
import logging
import os
import shutil
import sys
from tqdm import tqdm

from neural_sp.bin.args_asr import parse_args_eval
from neural_sp.bin.eval_utils import average_checkpoints
from neural_sp.bin.train_utils import (
    load_checkpoint,
    set_logger
)
from neural_sp.datasets.asr import build_dataloader
from neural_sp.models.seq2seq.speech2text import Speech2Text
from neural_sp.utils import mkdir_join

logger = logging.getLogger(__name__)


def main():

    # Load configuration
    args, recog_params, dir_name = parse_args_eval(sys.argv[1:])

    # Setting for logging
    if os.path.isfile(os.path.join(args.recog_dir, 'align.log')):
        os.remove(os.path.join(args.recog_dir, 'align.log'))
    set_logger(os.path.join(args.recog_dir, 'align.log'), stdout=args.recog_stdout)

    for i, s in enumerate(args.recog_sets):
        # Load dataloader
        dataloader = build_dataloader(args=args,
                                      tsv_path=s,
                                      batch_size=1,
                                      is_test=True)

        if i == 0:
            # Load the ASR model
            model = Speech2Text(args, dir_name)
            epoch = int(args.recog_model[0].split('-')[-1])
            if args.recog_n_average > 1:
                # Model averaging for Transformer
                model = average_checkpoints(model, args.recog_model[0],
                                            n_average=args.recog_n_average)
            else:
                load_checkpoint(args.recog_model[0], model)

            if not args.recog_unit:
                args.recog_unit = args.unit

            logger.info('recog unit: %s' % args.recog_unit)
            logger.info('epoch: %d' % epoch)
            logger.info('batch size: %d' % args.recog_batch_size)

            # GPU setting
            if args.recog_n_gpus >= 1:
                model.cudnn_setting(deterministic=True, benchmark=False)
                model.cuda()

        save_path = mkdir_join(args.recog_dir, 'ctc_forced_alignments')

        # Clean directory
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)

        pbar = tqdm(total=len(dataloader))
        while True:
            batch, is_new_epoch = dataloader.next(recog_params['recog_batch_size'])
            trigger_points = model.ctc_forced_align(batch['xs'], batch['ys'])  # `[B, L]`

            for b in range(len(batch['xs'])):
                save_path_spk = mkdir_join(save_path, batch['speakers'][b])
                save_path_utt = mkdir_join(save_path_spk, batch['utt_ids'][b] + '.txt')

                tokens = dataloader.idx2token[0](batch['ys'][b], return_list=True)
                with codecs.open(save_path_utt, 'w', encoding="utf-8") as f:
                    for i, tok in enumerate(tokens):
                        f.write('%s %d\n' % (tok, trigger_points[b, i]))
                # TODO: consider down sampling

            pbar.update(len(batch['xs']))

            if is_new_epoch:
                break

        pbar.close()


if __name__ == '__main__':
    main()
