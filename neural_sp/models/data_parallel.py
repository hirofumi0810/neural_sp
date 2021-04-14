# Copyright 2018 Kyoto University (Hirofumi Inylensguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Custom class for data parallel training."""

import torch.nn as nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel.scatter_gather import gather


class CustomDataParallel(DataParallel):

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(CustomDataParallel, self).__init__(module, device_ids, output_device, dim)

    def scatter(self, inputs, target_gpus, device_ids):
        if len(self.device_ids) <= 1:
            return [inputs], [target_gpus]

        def scatter_map(obj, i):
            if isinstance(obj, list) and len(obj) > 0:
                return [a[i] for a in zip(*[iter(obj)] * len(self.device_ids))]

        # assert len(inputs) == 1  # (batch,)
        inputs = inputs[0]

        # After scatter_map is called, a scatter_map cell will exist. This cell
        # has a reference to the actual function scatter_map, which has references
        # to a closure that has a reference to the scatter_map cell (because the
        # fn is recursive). To avoid this reference cycle, we set the function to
        # None, clearing the cell
        try:
            res = [{k: scatter_map(v, i) for k, v in inputs.items()} for i in range(len(self.device_ids))]
        finally:
            scatter_map = None

        target_gpus = [target_gpus] * len(self.device_ids)

        return res, target_gpus

    def gather(self, outputs, output_device):
        n_returns = len(outputs[0])
        assert n_returns == 2
        n_gpus = len(outputs)

        losses = [output[0] for output in outputs]
        observation_avg = {k: sum([output[1][k] for output in outputs]) / n_gpus
                           for k, v in outputs[0][1].items() if v is not None}

        return gather(losses, output_device, dim=self.dim).mean(), observation_avg


class CPUWrapperASR(nn.Module):
    def __init__(self, model):
        super(CPUWrapperASR, self).__init__()
        self.module = model

    def forward(self, batch, task, is_eval=False, teacher=None, teacher_lm=None):
        return self.module(batch, task, is_eval, teacher, teacher_lm)


class CPUWrapperLM(nn.Module):
    def __init__(self, model):
        super(CPUWrapperLM, self).__init__()
        self.module = model

    def forward(self, ys, state=None, is_eval=False, n_caches=0,
                ylens=[], predict_last=False):
        return self.module(ys, state, is_eval, n_caches, ylens, predict_last)
