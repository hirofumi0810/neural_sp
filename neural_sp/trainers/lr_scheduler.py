# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Learning rate scheduler."""

from glob import glob
import logging
import os
import torch

from neural_sp.trainers.optimizer import set_optimizer

logger = logging.getLogger(__name__)


class LRScheduler(object):
    """Learning rate scheduler (wrapper for optimizer).

    Args:
        optimizer (torch.optim): optimizer
        base_lr (float): maximum of learning rate
        decay_type (str): always/metric
            always) decay per epoch regardless of validation metric
            metric) decay if validation metric is not improved
        decay_start_epoch (int): the epoch to start decay
        decay_rate (float): the rate to decay the current learning rate
        decay_patient_n_epochs (int): decay learning rate if results have not been
            improved for 'decay_patient_n_epochs'
        early_stop_patient_n_epochs (int): number of epochs to tolerate stopping the training
            when validation performance is not improved
        lower_better (bool):
            True) The lower, the better.
            False) The higher, the better.
        warmup_start_lr (float): initial learning rate for warmup
        warmup_n_steps (int): steps for learning rate warmup
        peak_lr (float): peak learning rate. This can be enabled only for Noam optimizer.
        model_size (int): d_model for Transformer like learning rate scheduling
        factor (float): factor of learning rate for Transformer
        noam (bool): learning rate scheduling for Transformer
        save_checkpoints_topk (int): save top-k checkpoints

    """

    def __init__(self, optimizer, base_lr,
                 decay_type, decay_start_epoch, decay_rate,
                 decay_patient_n_epochs=0, early_stop_patient_n_epochs=-1, lower_better=True,
                 warmup_start_lr=0, warmup_n_steps=0, peak_lr=1e6,
                 model_size=0, factor=1, noam=False, save_checkpoints_topk=1):

        self.optimizer = optimizer
        self.noam = noam

        self._step = 0
        self._epoch = 0

        # for warmup
        if noam:
            self.decay_type = 'warmup'
            assert warmup_n_steps > 0
            self.base_lr = factor * (model_size ** -0.5)
        else:
            self.base_lr = base_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_n_steps = warmup_n_steps
        self.lr = self.base_lr
        self.peak_lr = peak_lr

        # for decay
        self.lower_better = lower_better
        self.decay_type = decay_type
        self.decay_start_epoch = decay_start_epoch
        self.decay_rate = decay_rate
        self.decay_patient_n_epochs = decay_patient_n_epochs
        self.not_improved_n_epochs = 0
        self.early_stop_patient_n_epochs = early_stop_patient_n_epochs

        # for performance monitoring
        self._is_topk = False
        self.topk = save_checkpoints_topk
        assert save_checkpoints_topk >= 1
        self.topk_list = []

    @property
    def n_steps(self):
        return self._step

    @property
    def n_epochs(self):
        return self._epoch

    @property
    def is_topk(self):
        return self._is_topk

    @property
    def is_early_stop(self):
        return self.not_improved_n_epochs >= self.early_stop_patient_n_epochs

    def step(self, skip_optimizer=False):
        self._step += 1
        if not skip_optimizer:
            self.optimizer.step()
        if self.noam:
            self._noam_lr()
        else:
            self._warmup_lr()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _noam_lr(self):
        """Warm up and decay learning rate per step based on Transformer."""
        self.lr = self.base_lr * min(self._step ** (-0.5),
                                     self._step * (self.warmup_n_steps ** (-1.5)))
        self.lr = min(self.lr, self.peak_lr)
        self._update_lr()

    def _warmup_lr(self):
        """Warm up learning rate per step by increasing linearly."""
        if self.warmup_n_steps > 0 and self._step <= self.warmup_n_steps:
            self.lr = (self.base_lr - self.warmup_start_lr) / \
                self.warmup_n_steps * self._step + self.warmup_start_lr
            self._update_lr()

    def epoch(self, metric=None):
        """Decay learning rate per epoch.

        Args:
            metric: (float): A metric to evaluate

        """
        self._epoch += 1
        self._is_topk = False
        is_best = False

        if metric is not None and not self.lower_better:
            metric *= -1

        if metric is not None:
            if len(self.topk_list) < self.topk or metric < self.topk_list[-1][1]:
                topk = sum([v < metric for (ep, v) in self.topk_list]) + 1
                logger.info('||||| Top-%d Score |||||' % topk)
                self.topk_list.append((self.n_epochs, metric))
                self.topk_list = sorted(self.topk_list, key=lambda x: x[1])[:self.topk]
                self._is_topk = True
                is_best = (topk == 1)
                for k, (ep, v) in enumerate(self.topk_list):
                    logger.info('----- Top-%d: epoch%d (%.3f)' % (k + 1, ep, v))

        if not self.noam and self._epoch >= self.decay_start_epoch:
            if self.decay_type == 'metric':
                if is_best:
                    # Improved
                    self.not_improved_n_epochs = 0
                elif self.not_improved_n_epochs < self.decay_patient_n_epochs:
                    # Not improved, but learning rate is not decayed
                    self.not_improved_n_epochs += 1
                else:
                    # Not improved, and learning rate is decayed
                    self.not_improved_n_epochs = 0
                    self.lr *= self.decay_rate
                    self._update_lr()
                    logger.info('Epoch %d: reducing learning rate to %.7f'
                                % (self._epoch, self.lr))
            elif self.decay_type == 'always':
                # if is_best:
                #     self.not_improved_n_epochs = 0
                # else:
                #     self.not_improved_n_epochs += 1
                self.lr *= self.decay_rate
                self._update_lr()
                logger.info('Epoch %d: reducing learning rate to %.7f'
                            % (self._epoch, self.lr))

    def _update_lr(self):
        """Reduce learning rate."""
        for param_group in self.optimizer.param_groups:
            if isinstance(self.optimizer, torch.optim.Adadelta):
                param_group['eps'] = self.lr
            else:
                param_group['lr'] = self.lr

    def save_checkpoint(self, model, save_path, remove_old=True, amp=None,
                        epoch_detail=None):
        """Save checkpoint.

        Args:
            model (torch.nn.Module):
            save_path (str): path to the directory to save a model
            optimizer (LRScheduler): optimizer wrapped by LRScheduler class
            remove_old (bool): if True, all checkpoints
                worse than the top-k ones are deleted
            amp ():
            epoch_detail (float): fine-grained epoch (used for MBR training)

        """
        if epoch_detail is None:
            epoch_detail = self.n_epochs
        model_path = os.path.join(save_path, 'model.epoch-' + str(epoch_detail))

        # Remove old checkpoints
        if remove_old:
            for path in glob(os.path.join(save_path, 'model.epoch-*')):
                if 'model.epoch-avg' in path:
                    continue
                epoch = int(path.split('-')[-1])
                if epoch not in [ep for (ep, v) in self.topk_list]:
                    os.remove(path)

        # Save parameters, optimizer, step index etc.
        checkpoint = {
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": self.get_state_dict(),  # LRScheduler class
        }
        if amp is not None:
            checkpoint['amp_state_dict'] = amp.state_dict()
        torch.save(checkpoint, model_path)

        logger.info("=> Saved checkpoint (epoch:%s): %s" % (str(epoch_detail), model_path))

    def get_state_dict(self):
        """Return state of scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.

        """
        dict = {k: v for k, v in self.__dict__.items() if k not in ['optimizer']}
        dict['optimizer_state_dict'] = self.optimizer.state_dict()
        return dict

    def load_state_dict(self, state_dict, use_cuda):
        """Load scheduler's state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
            use_cuda (bool): GPU mode

        """
        self.__dict__.update({k: v for k, v in state_dict.items() if k != 'optimizer_state_dict'})
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        # https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/4
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if use_cuda and isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    def convert_to_sgd(self, model, lr, weight_decay, decay_type, decay_rate):
        self.lr = lr
        self.decay_type = decay_type
        self.decay_rate = decay_rate
        self.noam = False

        # weight_decay = self.optimizer.defaults['weight_decay']
        self.optimizer = set_optimizer(model, 'sgd', lr, weight_decay)
        logger.info('========== Convert to SGD ==========')
