# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Reporter during training."""

import csv
import glob
import os
import numpy as np
from matplotlib import pyplot as plt
import logging
import matplotlib
from tensorboardX import SummaryWriter
import wandb

matplotlib.use('Agg')

plt.style.use('ggplot')
grey = '#878f99'
blue = '#4682B4'
orange = '#D2691E'
green = '#82b74b'

logger = logging.getLogger(__name__)


class Reporter:
    """Report loss, accuracy etc. during training."""

    def __init__(self, args, model, use_tensorboard=True):
        self.save_path = args.save_path

        # tensorboard
        if use_tensorboard:
            self.tf_writer = SummaryWriter(args.save_path)
        else:
            self.tf_writer = None

        # wandb
        self.use_wandb = args.use_wandb
        if self.use_wandb:
            if args.resume and getattr(args, 'wandb_id', None) is not None:
                id = args.wandb_id
            else:
                id = wandb.util.generate_id()
                args.wandb_id = id
            self._wanbd_id = id
            wandb.init(project=args.corpus, name=os.path.basename(args.save_path),
                       id=id, allow_val_change=True)
            for k, v in args.items():
                if 'recog' in k:
                    continue
                setattr(wandb.config, k, v)
            wandb.watch(model)
        else:
            self._wanbd_id = None

        self.obsv_train = {'loss': {}, 'acc': {}, 'ppl': {}}
        self.obsv_train_local = {'loss': {}, 'acc': {}, 'ppl': {}}
        self.obsv_dev = {'loss': {}, 'acc': {}, 'ppl': {}}
        self.obsv_eval = []

        self._step = 0
        self._epoch = 0
        self.steps = []
        self.epochs = []

    @property
    def wandb_id(self):
        return self._wanbd_id

    @property
    def n_steps(self):
        return self._step

    @property
    def n_epochs(self):
        return self._epoch

    def add_observation(self, observation, is_eval=False):
        """Restore observation per step.

        Args:
            observation (dict):
            is_eval (bool):

        """
        for k, v in observation.items():
            if v is None:
                continue
            metric, name = k.split('.')  # e.g., loss-ctc, acc-att
            # NOTE: metric: loss, acc, ppl

            if v == float("inf") or v == -float("inf"):
                logger.warning("WARNING: received an inf %s for %s." % (metric, k))

            if is_eval:
                # average for training
                if name not in self.obsv_train[metric].keys():
                    self.obsv_train[metric][name] = []
                train_local_avg = np.mean(self.obsv_train_local[metric][name])
                self.obsv_train[metric][name].append(train_local_avg)
                logger.info('%s (train): %.3f' % (k, train_local_avg))

                if name not in self.obsv_dev[metric].keys():
                    self.obsv_dev[metric][name] = []
                self.obsv_dev[metric][name].append(v)
                logger.info('%s (dev): %.3f' % (k, v))
                self.add_scalar('dev' + '/' + metric + '/' + name, v, is_eval=True)
            else:
                if name not in self.obsv_train_local[metric].keys():
                    self.obsv_train_local[metric][name] = []
                self.obsv_train_local[metric][name].append(v)
                self.add_scalar('train' + '/' + metric + '/' + name, v)

    def _log_wandb(self):
        """Add scalar values to wandb."""
        if self.use_wandb:
            wandb.log({'epoch': self._epoch}, step=self._step, commit=True)

    def add_scalar(self, key, value, is_eval=False):
        """Add scalar value to tensorboard and wandb."""
        if self.tf_writer is not None and value is not None:
            self.tf_writer.add_scalar(key, value, self._step)
        if self.use_wandb and value is not None:
            wandb.log({key: value}, step=self._step, commit=False)

    def add_tensorboard_histogram(self, key, value):
        """Add histogram value to tensorboard."""
        if self.tf_writer is not None:
            self.tf_writer.add_histogram(key, value, self._step)

    def resume(self, n_steps, n_epochs):
        self._step = n_steps
        self._epoch = n_epochs

        # Load CSV files
        for path in glob.glob(os.path.join(self.save_path, '*.csv')):
            if os.path.isfile(path):
                metric, name = os.path.basename(path).split('.')[0].split('-')

                with open(path, "r") as f:
                    reader = csv.DictReader(f, delimiter=",",
                                            fieldnames=['step', 'train', 'dev'])
                    lines = [row for row in reader if int(float(row['step'])) <= n_steps]
                    # [('step', val), ('train', val), ('dev', val)]

                    self.steps = [int(float(line['step'])) for line in lines]
                    self.obsv_train[metric][name] = [float(line['train']) for line in lines]
                    self.obsv_dev[metric][name] = [float(line['dev']) for line in lines]

    def step(self, is_eval=False):
        if is_eval:
            self.steps.append(self._step)
            self.obsv_train_local = {'loss': {}, 'acc': {}, 'ppl': {}}  # reset
            # NOTE: don't reset in add() because of multiple tasks
        else:
            self._log_wandb()
            self._step += 1
            # NOTE: different from the step counter in Noam Optimizer

    def epoch(self, metric=None, name='edit_distance'):
        self._epoch += 1
        if metric is None:
            return
        self.epochs.append(self._epoch)

        # register
        self.obsv_eval.append(metric)

        plt.clf()
        upper = 0.1
        plt.plot(self.epochs, self.obsv_eval, orange,
                 label='dev', linestyle='-')
        plt.xlabel('epoch', fontsize=12)
        plt.ylabel(name, fontsize=12)
        if max(self.obsv_eval) > 1:
            upper = min(100, max(self.obsv_eval) + 1)
        else:
            upper = min(upper, max(self.obsv_eval))
        plt.ylim([0, upper])
        plt.legend(loc="upper right", fontsize=12)
        if os.path.isfile(os.path.join(self.save_path, name + ".png")):
            os.remove(os.path.join(self.save_path, name + ".png"))
        plt.savefig(os.path.join(self.save_path, name + ".png"))

    def snapshot(self):
        # linestyles = ['solid', 'dashed', 'dotted', 'dashdotdotted']
        linestyles = ['-', '--', '-.', ':', ':', ':', ':', ':', ':', ':', ':', ':']
        for metric in self.obsv_train.keys():
            plt.clf()
            upper = 0.1
            for i, (k, v) in enumerate(sorted(self.obsv_train[metric].items())):
                # skip non-observed values
                if np.mean(self.obsv_train[metric][k]) == 0:
                    continue

                plt.plot(self.steps, self.obsv_train[metric][k], blue,
                         label=k + " (train)", linestyle=linestyles[i])
                plt.plot(self.steps, self.obsv_dev[metric][k], orange,
                         label=k + " (dev)", linestyle=linestyles[i])
                upper = max(upper, max(self.obsv_train[metric][k]))
                upper = max(upper, max(self.obsv_dev[metric][k]))

                # Save as csv file
                csv_path = os.path.join(self.save_path, metric + '-' + k + ".csv")
                if os.path.isfile(csv_path):
                    os.remove(csv_path)
                loss_graph = np.column_stack(
                    (self.steps, self.obsv_train[metric][k], self.obsv_dev[metric][k]))
                np.savetxt(csv_path, loss_graph, delimiter=",")  # no header

            if upper > 1:
                upper = min(upper + 10, 300)  # for CE, CTC loss

            plt.xlabel('step', fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.ylim([0, upper])
            plt.legend(loc="upper right", fontsize=12)
            png_path = os.path.join(self.save_path, metric + ".png")
            if os.path.isfile(png_path):
                os.remove(png_path)
            plt.savefig(png_path)

    def close(self):
        self.tf_writer.close()
