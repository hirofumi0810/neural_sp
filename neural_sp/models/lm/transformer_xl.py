# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""TransformerXL language model."""

import copy
from distutils.util import strtobool
import logging
import math
import os
import random
import shutil
import torch
import torch.nn as nn

from neural_sp.models.lm.lm_base import LMBase
from neural_sp.models.modules.initialization import init_like_transformer_xl
from neural_sp.models.modules.positional_embedding import XLPositionalEmbedding
from neural_sp.models.modules.transformer import TransformerDecoderBlock
from neural_sp.models.torch_utils import tensor2np
from neural_sp.utils import mkdir_join

import matplotlib
matplotlib.use('Agg')

random.seed(1)

logger = logging.getLogger(__name__)


class TransformerXL(LMBase):
    """TransformerXL language model."""

    def __init__(self, args, save_path=None):

        super(LMBase, self).__init__()
        logger.info(self.__class__.__name__)

        self.lm_type = args.lm_type
        self.save_path = save_path

        self.d_model = args.transformer_d_model
        self.n_layers = args.n_layers
        self.n_heads = args.transformer_n_heads
        self.lsm_prob = args.lsm_prob

        if args.mem_len > 0:
            self.mem_len = args.mem_len
        else:
            self.mem_len = args.bptt
        if args.recog_mem_len > 0:
            self.mem_len = args.recog_mem_len

        self.vocab = args.vocab
        self.eos = 2
        self.pad = 3
        # NOTE: reserved in advance

        # for cache
        self.cache_theta = 0.2  # smoothing parameter
        self.cache_lambda = 0.2  # cache weight
        self.cache_ids = []
        self.cache_keys = []
        self.cache_attn = []
        self.embed_cache = None

        # positional embedding
        self.pos_emb = XLPositionalEmbedding(self.d_model, args.dropout_in)
        self.u_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_model // self.n_heads))
        self.v_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_model // self.n_heads))
        # NOTE: u_bias and v_bias are global parameters

        self.embed = nn.Embedding(self.vocab, self.d_model, padding_idx=self.pad)
        self.scale = math.sqrt(self.d_model)  # for token embedding
        self.dropout_emb = nn.Dropout(p=args.dropout_in)  # for token embedding
        self.layers = nn.ModuleList([copy.deepcopy(TransformerDecoderBlock(
            self.d_model, args.transformer_d_ff, 'scaled_dot',
            self.n_heads, args.dropout_hidden, args.dropout_att, args.dropout_layer,
            args.transformer_layer_norm_eps, args.transformer_ffn_activation, args.transformer_param_init,
            src_tgt_attention=False, memory_transformer=True)) for lth in range(self.n_layers)])
        self.norm_out = nn.LayerNorm(self.d_model, eps=args.transformer_layer_norm_eps)

        self.adaptive_softmax = None
        self.output = None
        if args.adaptive_softmax:
            self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
                self.d_model, self.vocab,
                cutoffs=[round(self.vocab / 15), 3 * round(self.vocab / 15)],
                # cutoffs=[self.vocab // 25, 3 * self.vocab // 5],
                div_value=4.0)
        else:
            self.output = nn.Linear(self.d_model, self.vocab)
            if args.tie_embedding:
                self.output.weight = self.embed.weight

        self.reset_parameters()

    @property
    def output_dim(self):
        return self.d_model

    @staticmethod
    def add_args(parser, args):
        """Add arguments."""
        group = parser.add_argument_group("Transformer-XL LM")
        group.add_argument('--transformer_d_model', type=int, default=256,
                           help='number of units in the MHA layer')
        group.add_argument('--transformer_d_ff', type=int, default=2048,
                           help='number of units in the FFN layer')
        # group.add_argument('--transformer_ffn_bottleneck_dim', type=int, default=0,
        #                    help='bottleneck dimension in the FFN layer')
        group.add_argument('--transformer_n_heads', type=int, default=4,
                           help='number of heads in the MHA layer')
        group.add_argument('--transformer_layer_norm_eps', type=float, default=1e-12,
                           help='epsilon value for layer normalization')
        group.add_argument('--transformer_ffn_activation', type=str, default='relu',
                           choices=['relu', 'gelu', 'gelu_accurate', 'glu', 'swish'],
                           help='nonlinear activation for the FFN layer')
        group.add_argument('--transformer_param_init', type=str, default='xavier_uniform',
                           choices=['xavier_uniform', 'pytorch'],
                           help='parameter initialization')
        group.add_argument('--dropout_att', type=float, default=0.1,
                           help='dropout probability for the attention weights')
        group.add_argument('--dropout_layer', type=float, default=0.0,
                           help='LayerDrop probability for Transformer layers')
        # XL specific
        group.add_argument('--mem_len', type=int, default=0,
                           help='number of tokens for memory in TransformerXL during training')
        return parser

    @staticmethod
    def define_name(dir_name, args):
        dir_name = args.lm_type
        dir_name += str(args.transformer_d_model) + 'dmodel'
        dir_name += str(args.transformer_d_ff) + 'dff'
        dir_name += str(args.n_layers) + 'L'
        dir_name += str(args.transformer_n_heads) + 'H'
        if args.tie_embedding:
            dir_name += '_tie'
        if args.adaptive_softmax:
            dir_name += '_adaptiveSM'
        if args.mem_len > 0:
            dir_name += '_mem' + str(args.mem_len)
        return dir_name

    def reset_parameters(self):
        """Initialize parameters with normal distribution."""
        logger.info('===== Initialize %s with normal distribution =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            init_like_transformer_xl(n, p, std=0.02)

    def init_memory(self):
        """Initialize memory."""
        return [torch.empty(0, dtype=torch.float).to(self.device)
                for _ in range(self.n_layers)]

    def update_memory(self, memory_prev, hidden_states):
        """Update memory.

        Args:
            memory_prev (list): length `n_layers`, each of which contains `[B, mlen, d_model]`
            hidden_states (list): length `n_layers`, each of which contains `[B, L, d_model]`
        Returns:
            new_mems (list): length `n_layers`, each of which contains `[B, mlen, d_model]`

        """
        if memory_prev is None:
            memory_prev = self.init_memory()  # 0-th to L-1-th layer
        assert len(hidden_states) == len(memory_prev)
        mlen = memory_prev[0].size(1) if memory_prev[0].dim() > 1 else 0
        qlen = hidden_states[0].size(1)

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + qlen
            start_idx = max(0, end_idx - self.mem_len)
            for m, h in zip(memory_prev, hidden_states):
                cat = torch.cat([m, h], dim=1)  # `[B, mlen + qlen, d_model]`
                new_mems.append(cat[:, start_idx:end_idx].detach())  # `[B, self.mem_len, d_model]`

        return new_mems

    def decode(self, ys, state=None, mems=None, cache=None, incremental=False,
               emb_cache=False):
        """Decode function.

        Args:
            ys (LongTensor): `[B, L]`
            state (list): dummy interfance for RNNLM
            mems (list): length `n_layers`, each of which contains a FloatTensor `[B, mlen, d_model]`
            cache (list): length `L`, each of which contains a FloatTensor `[B, L-1, d_model]`
            incremental (bool): ASR decoding mode
            emb_cache (bool): precompute token embeddings for fast infernece
        Returns:
            logits (FloatTensor): `[B, L, vocab]`
            out (FloatTensor): `[B, L, d_model]`
            new_cache (list): length `n_layers`, each of which contains a FloatTensor `[B, L, d_model]`

        """
        # for ASR decoding
        if cache is None:
            cache = [None] * self.n_layers  # 1-th to L-th layer

        if mems is None:
            mems = self.init_memory()
            mlen = 0
        else:
            mlen = mems[0].size(1)

        bs, ylen = ys.size()[:2]
        if incremental and cache[0] is not None:
            ylen = cache[0].size(1) + 1

        # Create the self-attention mask
        causal_mask = ys.new_ones(ylen, ylen + mlen).byte()
        causal_mask = torch.tril(causal_mask, diagonal=0 + mlen, out=causal_mask).unsqueeze(0)
        causal_mask = causal_mask.repeat([bs, 1, 1])  # `[B, L, L+mlen]`

        # Pre-compute embedding
        if emb_cache and self.embed_cache is None:
            indices = torch.arange(0, self.vocab, 1, dtype=torch.int64).to(ys.device)
            self.embed_cache = self.dropout_emb(self.embed(indices) * self.scale)  # `[1, vocab, emb_dim]`

        if self.embed_cache is not None:
            out = self.embed_cache[ys]
        else:
            out = self.dropout_emb(self.embed(ys.long()) * self.scale)

        pos_embs = self.pos_emb(ys, mlen=mlen)

        new_mems = [None] * self.n_layers
        new_cache = [None] * self.n_layers
        hidden_states = [out]
        for lth, (mem, layer) in enumerate(zip(mems, self.layers)):
            if incremental and mlen > 0 and mem.size(0) != bs:
                mem = mem.repeat([bs, 1, 1])
            out = layer(out, causal_mask, cache=cache[lth],
                        pos_embs=pos_embs, memory=mem, u_bias=self.u_bias, v_bias=self.v_bias)
            if incremental:
                new_cache[lth] = out
            elif lth < self.n_layers - 1:
                hidden_states.append(out)
                # NOTE: outputs from the last layer is not used for memory
            if not self.training and layer.yy_aws is not None:
                setattr(self, 'yy_aws_layer%d' % lth, tensor2np(layer.yy_aws))
        out = self.norm_out(out)
        if self.adaptive_softmax is None:
            logits = self.output(out)
        else:
            logits = out

        if incremental:
            # NOTE: do not update memory here during ASR decoding
            return logits, out, new_cache
        else:
            # Update memory
            new_mems = self.update_memory(mems, hidden_states)
            return logits, out, new_mems

    def plot_attention(self, n_cols=4):
        """Plot attention for each head in all layers."""
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator

        save_path = mkdir_join(self.save_path, 'att_weights')

        # Clean directory
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)

        for lth in range(self.n_layers):
            if not hasattr(self, 'yy_aws_layer%d' % lth):
                continue

            yy_aws = getattr(self, 'yy_aws_layer%d' % lth)

            plt.clf()
            fig, axes = plt.subplots(self.n_heads // n_cols, n_cols, figsize=(20, 8))
            for h in range(self.n_heads):
                if self.n_heads > n_cols:
                    ax = axes[h // n_cols, h % n_cols]
                else:
                    ax = axes[h]
                ax.imshow(yy_aws[-1, h, :, :], aspect="auto")
                ax.grid(False)
                ax.set_xlabel("Input (head%d)" % h)
                ax.set_ylabel("Output (head%d)" % h)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            fig.tight_layout()
            fig.savefig(os.path.join(save_path, 'layer%d.png' % (lth)))
            plt.close()
