# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Base class for all models."""

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import vector_to_parameters, parameters_to_vector

np.random.seed(1)

logger = logging.getLogger(__name__)


class ModelBase(nn.Module):
    """A base class for all models. All models have to inherit this class."""

    def __init__(self, *args, **kwargs):

        super().__init__()
        logger.info('Overriding ModelBase class.')

    @property
    def torch_version(self):
        return float('.'.join(torch.__version__.split('.')[:2]))

    @property
    def num_params_dict(self):
        if not hasattr(self, '_nparams_dict'):
            self._nparams_dict = {}
            for n, p in self.named_parameters():
                self._nparams_dict[n] = p.view(-1).size(0)
        return self._nparams_dict

    @property
    def total_parameters(self):
        if not hasattr(self, '_nparams'):
            self._nparams = 0
            for n, p in self.named_parameters():
                self._nparams += p.view(-1).size(0)
        return self._nparams

    @property
    def use_cuda(self):
        return torch.cuda.is_available()

    @property
    def device_id(self):
        return torch.cuda.device_of(next(self.parameters())).idx

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def define_name(dir_name, args):
        raise NotImplementedError

    def reset_parameters(self, param_init):
        raise NotImplementedError

    def init_forget_gate_bias_with_one(self):
        """Initialize bias in forget gate with 1. See detail in

            https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745

        """
        for n, p in self.named_parameters():
            if p.dim() == 1 and 'bias_ih' in n:
                dim = p.size(0)
                start, end = dim // 4, dim // 2
                p.data[start:end].fill_(1.)
                logger.info('Initialize %s with 1 (bias in forget gate)' % (n))

    def add_weight_noise(self, std):
        """Add variational Gaussian noise to model parameters.

        Args:
            std (float): standard deviation

        """
        with torch.no_grad():
            param_vector = parameters_to_vector(self.parameters())
            normal_dist = torch.distributions.Normal(loc=torch.tensor([0.]), scale=torch.tensor([std]))
            noise = normal_dist.sample(param_vector.size())
            if self.device_id >= 0:
                noise = noise.cuda(self.device_id)
            param_vector.add_(noise[0])
        vector_to_parameters(param_vector, self.parameters())

    def cudnn_setting(self, deterministic=False, benchmark=True):
        """CuDNN setting.

        Args:
            deterministic (bool):
            benchmark (bool):

        """
        assert self.use_cuda
        if benchmark:
            torch.backends.cudnn.benchmark = True
        elif deterministic:
            torch.backends.cudnn.enabled = False
            # NOTE: this is slower than GPU mode.
        logger.info("torch.backends.cudnn.benchmark: %s" % torch.backends.cudnn.benchmark)
        logger.info("torch.backends.cudnn.enabled: %s" % torch.backends.cudnn.enabled)
