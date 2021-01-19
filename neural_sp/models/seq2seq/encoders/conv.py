# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""CNN encoder."""

from distutils.util import strtobool
import logging
import math
import numpy as np
import torch
import torch.nn as nn

from neural_sp.models.modules.initialization import init_with_lecun_normal
from neural_sp.models.seq2seq.encoders.encoder_base import EncoderBase

logger = logging.getLogger(__name__)


class ConvEncoder(EncoderBase):
    """CNN encoder.

    Args:
        input_dim (int): dimension of input features (freq * channel)
        in_channel (int): number of channels of input features
        channels (list): number of channles in CNN blocks
        kernel_sizes (list): size of kernels in CNN blocks
        strides (list): strides in CNN blocks
        poolings (list): size of poolings in CNN blocks
        dropout (float): probability to drop nodes in hidden-hidden connection
        batch_norm (bool): apply batch normalization
        layer_norm (bool): apply layer normalization
        residual (bool): apply residual connections
        bottleneck_dim (int): dimension of the bridge layer after the last layer
        param_init (float): mean of uniform distribution for parameter initialization
        layer_norm_eps (float): epsilon value for layer normalization

    """

    def __init__(self, input_dim, in_channel, channels,
                 kernel_sizes, strides, poolings,
                 dropout, batch_norm, layer_norm, residual,
                 bottleneck_dim, param_init, layer_norm_eps=1e-12):

        super(ConvEncoder, self).__init__()

        (channels, kernel_sizes, strides, poolings), is_1dconv = parse_cnn_config(
            channels, kernel_sizes, strides, poolings)

        self.is_1dconv = is_1dconv
        self.in_channel = in_channel
        assert input_dim % in_channel == 0
        self.input_freq = input_dim // in_channel
        self.residual = residual

        assert len(channels) > 0
        assert len(channels) == len(kernel_sizes) == len(strides) == len(poolings)

        self.layers = nn.ModuleList()
        C_i = input_dim if is_1dconv else in_channel
        in_freq = self.input_freq
        for lth in range(len(channels)):
            if is_1dconv:
                block = Conv1dBlock(in_channel=C_i,
                                    out_channel=channels[lth],
                                    kernel_size=kernel_sizes[lth],  # T
                                    stride=strides[lth],  # T
                                    pooling=poolings[lth],  # T
                                    dropout=dropout,
                                    batch_norm=batch_norm,
                                    layer_norm=layer_norm,
                                    layer_norm_eps=layer_norm_eps,
                                    residual=residual)
            else:
                block = Conv2dBlock(input_dim=in_freq,
                                    in_channel=C_i,
                                    out_channel=channels[lth],
                                    kernel_size=kernel_sizes[lth],
                                    stride=strides[lth],
                                    pooling=poolings[lth],
                                    dropout=dropout,
                                    batch_norm=batch_norm,
                                    layer_norm=layer_norm,
                                    layer_norm_eps=layer_norm_eps,
                                    residual=residual)
            self.layers += [block]
            in_freq = block.output_dim
            C_i = channels[lth]

        self._odim = C_i if is_1dconv else int(C_i * in_freq)

        self.bridge = None
        if bottleneck_dim > 0 and bottleneck_dim != self._odim:
            self.bridge = nn.Linear(self._odim, bottleneck_dim)
            self._odim = bottleneck_dim

        # calculate subsampling factor
        self._factor = 1
        if poolings:
            for p in poolings:
                self._factor *= p if is_1dconv else p[0]

        self.calculate_context_size(kernel_sizes, strides, poolings)

        self.reset_parameters(param_init)

    @staticmethod
    def add_args(parser, args):
        """Add arguments."""
        group = parser.add_argument_group("CNN encoder")
        group.add_argument('--conv_in_channel', type=int, default=1,
                           help='input dimension of the first CNN block')
        group.add_argument('--conv_channels', type=str, default="",
                           help='delimited list of channles in each CNN block')
        group.add_argument('--conv_kernel_sizes', type=str, default="",
                           help='delimited list of kernel sizes in each CNN block')
        group.add_argument('--conv_strides', type=str, default="",
                           help='delimited list of strides in each CNN block')
        group.add_argument('--conv_poolings', type=str, default="",
                           help='delimited list of poolings in each CNN block')
        group.add_argument('--conv_batch_norm', type=strtobool, default=False,
                           help='apply batch normalization in each CNN block')
        group.add_argument('--conv_layer_norm', type=strtobool, default=False,
                           help='apply layer normalization in each CNN block')
        group.add_argument('--conv_bottleneck_dim', type=int, default=0,
                           help='dimension of the bottleneck layer between CNN and the subsequent RNN/Transformer layers')
        return parser

    @staticmethod
    def define_name(dir_name, args):
        assert 'conv' in args.enc_type
        dir_name = args.enc_type.replace('conv_', '')
        if args.conv_channels and len(args.conv_channels.split('_')) > 0:
            tmp = dir_name
            dir_name = 'conv' + str(len(args.conv_channels.split('_'))) + 'L'
            if args.conv_batch_norm:
                dir_name += 'bn'
            if args.conv_layer_norm:
                dir_name += 'ln'
            dir_name += tmp
        return dir_name

    def calculate_context_size(self, kernel_sizes, strides, poolings):
        self._context_size = 0
        context_size_bottom = 0
        factor = 1
        for lth in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[lth] if self.is_1dconv else kernel_sizes[lth][0]
            pooling = poolings[lth] if self.is_1dconv else poolings[lth][0]

            lookahead = (kernel_size - 1) // 2
            lookahead *= 2
            # NOTE: each CNN block has 2 CNN layers

            if factor == 1:
                self._context_size += lookahead
                context_size_bottom = self._context_size
            else:
                self._context_size += context_size_bottom * lookahead
                context_size_bottom *= pooling
            factor *= pooling

    @property
    def context_size(self):
        return self._context_size

    def reset_parameters(self, param_init):
        """Initialize parameters with lecun style."""
        logger.info('===== Initialize %s with lecun style =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            init_with_lecun_normal(n, p, param_init)

    def forward(self, xs, xlens, lookback=False, lookahead=False):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, F]`
            xlens (IntTenfor): `[B]` (on CPU)
        Returns:
            xs (FloatTensor): `[B, T', F']`
            xlens (IntTenfor): `[B]` (on CPU)

        """
        B, T, F = xs.size()
        C_i = self.in_channel
        if not self.is_1dconv:
            xs = xs.view(B, T, C_i, F // C_i).contiguous().transpose(2, 1)  # `[B, C_i, T, F // C_i]`

        for block in self.layers:
            xs, xlens = block(xs, xlens, lookback=lookback, lookahead=lookahead)
        if not self.is_1dconv:
            B, C_o, T, F = xs.size()
            xs = xs.transpose(2, 1).contiguous().view(B, T, -1)  # `[B, T', C_o * F']`

        # Bridge layer
        if self.bridge is not None:
            xs = self.bridge(xs)

        return xs, xlens


class Conv1dBlock(EncoderBase):
    """1d-CNN block."""

    def __init__(self, in_channel, out_channel,
                 kernel_size, stride, pooling,
                 dropout, batch_norm, layer_norm, layer_norm_eps, residual):

        super(Conv1dBlock, self).__init__()

        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.residual = residual
        self.dropout = nn.Dropout(p=dropout)

        # 1st layer
        self.conv1 = nn.Conv1d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=1)
        self._odim = out_channel
        self.batch_norm1 = nn.BatchNorm1d(out_channel) if batch_norm else lambda x: x
        self.layer_norm1 = nn.LayerNorm(out_channel,
                                        eps=layer_norm_eps) if layer_norm else lambda x: x

        # 2nd layer
        self.conv2 = nn.Conv1d(in_channels=out_channel,
                               out_channels=out_channel,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=1)
        self._odim = out_channel
        self.batch_norm2 = nn.BatchNorm1d(out_channel) if batch_norm else lambda x: x
        self.layer_norm2 = nn.LayerNorm(out_channel,
                                        eps=layer_norm_eps) if layer_norm else lambda x: x

        # Max Pooling
        self.pool = None
        if pooling > 1:
            self.pool = nn.MaxPool1d(kernel_size=pooling,
                                     stride=pooling,
                                     padding=0,
                                     ceil_mode=True)
            # NOTE: If ceil_mode is False, remove last feature when the dimension of features are odd.
            self._odim = self._odim
            if self._odim % 2 != 0:
                self._odim = (self._odim // 2) * 2
                # TODO(hirofumi0810): more efficient way?

    def forward(self, xs, xlens, lookback=False, lookahead=False):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, F]`
            xlens (IntTensor): `[B]` (on CPU)
            lookback (bool): truncate the leftmost frames
                because of lookback frames for context
            lookahead (bool): truncate the rightmost frames
                because of lookahead frames for context
        Returns:
            xs (FloatTensor): `[B, T', F']`
            xlens (IntTensor): `[B]` (on CPU)

        """
        residual = xs

        xs = self.conv1(xs.transpose(2, 1)).transpose(2, 1)
        xs = self.batch_norm1(xs)
        xs = self.layer_norm1(xs)
        xs = torch.relu(xs)
        xs = self.dropout(xs)
        xlens = update_lens_1d(xlens, self.conv1)

        xs = self.conv2(xs.transpose(2, 1)).transpose(2, 1)
        xs = self.batch_norm2(xs)
        xs = self.layer_norm2(xs)
        if self.residual and xs.size() == residual.size():
            xs += residual  # NOTE: this is the same place as in ResNet
        xs = torch.relu(xs)
        xs = self.dropout(xs)
        xlens = update_lens_1d(xlens, self.conv2)

        if self.pool is not None:
            xs = self.pool(xs.transpose(2, 1)).transpose(2, 1)
            xlens = update_lens_1d(xlens, self.pool)

        return xs, xlens


class Conv2dBlock(EncoderBase):
    """2d-CNN block."""

    def __init__(self, input_dim, in_channel, out_channel,
                 kernel_size, stride, pooling,
                 dropout, batch_norm, layer_norm, layer_norm_eps, residual):

        super(Conv2dBlock, self).__init__()

        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.residual = residual
        self.dropout = nn.Dropout(p=dropout)
        self.time_axis = 0

        # 1st layer
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=tuple(kernel_size),
                               stride=tuple(stride),
                               padding=(1, 1))
        self._odim = update_lens_2d(torch.IntTensor([input_dim]), self.conv1, dim=1)[0].item()
        self.batch_norm1 = nn.BatchNorm2d(out_channel) if batch_norm else lambda x: x
        self.layer_norm1 = LayerNorm2D(out_channel, self._odim,
                                       eps=layer_norm_eps) if layer_norm else lambda x: x

        # 2nd layer
        self.conv2 = nn.Conv2d(in_channels=out_channel,
                               out_channels=out_channel,
                               kernel_size=tuple(kernel_size),
                               stride=tuple(stride),
                               padding=(1, 1))
        self._odim = update_lens_2d(torch.IntTensor([self._odim]), self.conv2, dim=1)[0].item()
        self.batch_norm2 = nn.BatchNorm2d(out_channel) if batch_norm else lambda x: x
        self.layer_norm2 = LayerNorm2D(out_channel, self._odim,
                                       eps=layer_norm_eps) if layer_norm else lambda x: x

        # Max Pooling
        self.pool = None
        self._factor = 1
        if len(pooling) > 0 and np.prod(pooling) > 1:
            self.pool = nn.MaxPool2d(kernel_size=tuple(pooling),
                                     stride=tuple(pooling),
                                     padding=(0, 0),
                                     ceil_mode=True)
            # NOTE: If ceil_mode is False, remove last feature when the dimension of features are odd.
            self._odim = update_lens_2d(torch.IntTensor([self._odim]), self.pool, dim=1)[0].item()
            if self._odim % 2 != 0:
                self._odim = (self._odim // 2) * 2
                # TODO(hirofumi0810): more efficient way?

            # calculate subsampling factor
            self._factor *= pooling[0]

    def forward(self, xs, xlens, lookback=False, lookahead=False):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, C_i, T, F]`
            xlens (IntTensor): `[B]` (on CPU)
            lookback (bool): truncate leftmost frames for lookback in CNN context
            lookahead (bool): truncate rightmost frames for lookahead in CNN context
        Returns:
            xs (FloatTensor): `[B, C_o, T', F']`
            xlens (IntTensor): `[B]` (on CPU)

        """
        residual = xs

        xs = self.conv1(xs)
        xs = self.batch_norm1(xs)
        xs = self.layer_norm1(xs)
        xs = torch.relu(xs)
        xs = self.dropout(xs)
        xlens = update_lens_2d(xlens, self.conv1, dim=0)
        stride = self.conv1.stride[self.time_axis]
        if lookback and xs.size(2) > stride:
            xmax = xs.size(2)
            xs = xs[:, :, stride:]
            xlens = xlens - (xmax - xs.size(2))
        if lookahead and xs.size(2) > stride:
            xmax = xs.size(2)
            xs = xs[:, :, :xs.size(2) - stride]
            xlens = xlens - (xmax - xs.size(2))

        xs = self.conv2(xs)
        xs = self.batch_norm2(xs)
        xs = self.layer_norm2(xs)
        if self.residual and xs.size() == residual.size():
            xs += residual  # NOTE: this is the same place as in ResNet
        xs = torch.relu(xs)
        xs = self.dropout(xs)
        xlens = update_lens_2d(xlens, self.conv2, dim=0)
        stride = self.conv2.stride[self.time_axis]
        if lookback and xs.size(2) > stride:
            xmax = xs.size(2)
            xs = xs[:, :, stride:]
            xlens = xlens - (xmax - xs.size(2))
        if lookahead and xs.size(2) > stride:
            xmax = xs.size(2)
            xs = xs[:, :, :xs.size(2) - stride]
            xlens = xlens - (xmax - xs.size(2))

        if self.pool is not None:
            xs = self.pool(xs)
            xlens = update_lens_2d(xlens, self.pool, dim=0)

        return xs, xlens


class LayerNorm2D(nn.Module):
    """Layer normalization for CNN outputs."""

    def __init__(self, channel, idim, eps=1e-12):

        super(LayerNorm2D, self).__init__()
        self.norm = nn.LayerNorm([channel, idim], eps=eps)

    def forward(self, xs):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, C, T, F]`
        Returns:
            xs (FloatTensor): `[B, C, T, F]`

        """
        B, C, T, F = xs.size()
        xs = xs.transpose(2, 1).contiguous()
        xs = self.norm(xs)
        xs = xs.transpose(2, 1)
        return xs


def update_lens_1d(seq_lens, layer):
    """Update lengths (frequency or time).

    Args:
        seq_lens (IntTensor): `[B]`
        layer (nn.Conv1d or nn.MaxPool1d):
    Returns:
        seq_lens (IntTensor): `[B]`

    """
    if seq_lens is None:
        return seq_lens
    assert isinstance(seq_lens, torch.IntTensor)
    assert type(layer) in [nn.Conv1d, nn.MaxPool1d]
    # seq_lens = [_update_1d(seq_len.item(), layer) for seq_len in seq_lens]
    seq_lens = [_update_1d(seq_len, layer) for seq_len in seq_lens]
    seq_lens = torch.IntTensor(seq_lens)
    return seq_lens


def _update_1d(seq_len, layer):
    if type(layer) == nn.MaxPool1d and layer.ceil_mode:
        return math.ceil(
            (seq_len + 1 + 2 * layer.padding - (layer.kernel_size - 1) - 1) // layer.stride + 1)
    else:
        return math.floor(
            (seq_len + 2 * layer.padding[0] - (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1)


def update_lens_2d(seq_lens, layer, dim=0):
    """Update lengths (frequency or time).

    Args:
        seq_lens (IntTensor): `[B]`
        layer (nn.Conv2d or nn.MaxPool2d):
        dim (int):
    Returns:
        seq_lens (IntTensor): `[B]`

    """
    if seq_lens is None:
        return seq_lens
    assert isinstance(seq_lens, torch.IntTensor)
    assert type(layer) in [nn.Conv2d, nn.MaxPool2d]
    # seq_lens = [_update_2d(seq_len.item(), layer, dim) for seq_len in seq_lens]
    seq_lens = [_update_2d(seq_len, layer, dim) for seq_len in seq_lens]
    seq_lens = torch.IntTensor(seq_lens)
    return seq_lens


def _update_2d(seq_len, layer, dim):
    if type(layer) == nn.MaxPool2d and layer.ceil_mode:
        return math.ceil(
            (seq_len + 1 + 2 * layer.padding[dim] - (layer.kernel_size[dim] - 1) - 1) // layer.stride[dim] + 1)
    else:
        return math.floor(
            (seq_len + 2 * layer.padding[dim] - (layer.kernel_size[dim] - 1) - 1) // layer.stride[dim] + 1)


def parse_cnn_config(channels, kernel_sizes, strides, poolings):
    _channels, _kernel_sizes, _strides, _poolings = [], [], [], []
    is_1dconv = '(' not in kernel_sizes
    if len(channels) > 0:
        _channels = [int(c) for c in channels.split('_')]
    if len(kernel_sizes) > 0:
        if is_1dconv:
            _kernel_sizes = [int(c) for c in kernel_sizes.split('_')]
        else:
            _kernel_sizes = [[int(c.split(',')[0].replace('(', '')),
                              int(c.split(',')[1].replace(')', ''))] for c in kernel_sizes.split('_')]
    if len(strides) > 0:
        if is_1dconv:
            assert '(' not in _strides and ')' not in _strides
            _strides = [int(s) for s in strides.split('_')]
        else:
            _strides = [[int(s.split(',')[0].replace('(', '')),
                         int(s.split(',')[1].replace(')', ''))] for s in strides.split('_')]
    if len(poolings) > 0:
        if is_1dconv:
            assert '(' not in poolings and ')' not in poolings
            _poolings = [int(p) for p in poolings.split('_')]
        else:
            _poolings = [[int(p.split(',')[0].replace('(', '')),
                          int(p.split(',')[1].replace(')', ''))] for p in poolings.split('_')]
    return (_channels, _kernel_sizes, _strides, _poolings), is_1dconv
