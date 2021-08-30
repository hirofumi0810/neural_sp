# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate model by accuracy."""

import logging
from tqdm import tqdm


logger = logging.getLogger(__name__)


def eval_accuracy(models, dataloader, batch_size=1, progressbar=False):
    """Evaluate a Seq2seq by teacher-forcing accuracy.

    Args:
        models (List): models to evaluate
        dataloader (torch.utils.data.DataLoader): evaluation dataloader
        batch_size (int): batch size
        progressbar (bool): if True, visualize progressbar
    Returns:
        accuracy (float): Average accuracy

    """
    total_acc = 0
    n_tokens = 0

    # Reset data counter
    dataloader.reset()

    if progressbar:
        pbar = tqdm(total=len(dataloader))

    for batch in dataloader:
        _, observation = models[0](batch, task='all', is_eval=True)
        n_tokens_b = sum([len(y) for y in batch['ys']])
        _acc = observation.get('acc.att', observation.get('acc.att-sub1', 0))
        total_acc += _acc * n_tokens_b
        n_tokens += n_tokens_b

        if progressbar:
            pbar.update(len(batch['ys']))

    if progressbar:
        pbar.close()

    # Reset data counters
    dataloader.reset(is_new_epoch=True)

    accuracy = total_acc / n_tokens

    logger.debug('Accuracy (%s): %.2f %%' % (dataloader.set, accuracy))

    return accuracy
