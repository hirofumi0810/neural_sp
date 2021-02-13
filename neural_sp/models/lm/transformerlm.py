# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer language model."""

import copy
import logging
import os
import random
import shutil
import torch
import torch.nn as nn

from neural_sp.models.lm.lm_base import LMBase
from neural_sp.models.modules.positional_embedding import PositionalEncoding
from neural_sp.models.modules.transformer import TransformerDecoderBlock
from neural_sp.models.torch_utils import tensor2np
from neural_sp.utils import mkdir_join

import matplotlib
matplotlib.use('Agg')

random.seed(1)

logger = logging.getLogger(__name__)


class TransformerLM(LMBase):
    """Transformer language model."""

    def __init__(self, args, save_path=None):

        super(LMBase, self).__init__()
        logger.info(self.__class__.__name__)

        self.lm_type = args.lm_type
        self.save_path = save_path

        self.d_model = args.transformer_d_model
        self.n_layers = args.n_layers
        self.n_heads = args.transformer_n_heads
        self.lsm_prob = args.lsm_prob
        self.tie_embedding = args.tie_embedding

        self.mem_len = args.mem_len
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

        self.embed = nn.Embedding(self.vocab, self.d_model, padding_idx=self.pad)
        self.pos_enc = PositionalEncoding(self.d_model, args.dropout_in, args.transformer_pe_type,
                                          args.transformer_param_init)
        self.layers = nn.ModuleList([copy.deepcopy(TransformerDecoderBlock(
            self.d_model, args.transformer_d_ff, 'scaled_dot',
            self.n_heads, args.dropout_hidden, args.dropout_att, args.dropout_layer,
            args.transformer_layer_norm_eps, args.transformer_ffn_activation, args.transformer_param_init,
            src_tgt_attention=False)) for lth in range(self.n_layers)])
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
        group = parser.add_argument_group("Transformer LM")
        group.add_argument('--transformer_d_model', type=int, default=256,
                           help='number of units in the MHA layer')
        group.add_argument('--transformer_d_ff', type=int, default=2048,
                           help='number of units in the FFN layer')
        # group.add_argument('--transformer_ffn_bottleneck_dim', type=int, default=0,
        #                    help='bottleneck dimension in the FFN layer')
        group.add_argument('--transformer_n_heads', type=int, default=4,
                           help='number of heads in the MHA layer')
        group.add_argument('--transformer_pe_type', type=str, default='add',
                           choices=['add', 'concat', 'none', '1dconv3L'],
                           help='type of positional encoding')
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
        # memory
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
        dir_name += 'pe' + str(args.transformer_pe_type)
        if args.tie_embedding:
            dir_name += '_tie'
        if args.adaptive_softmax:
            dir_name += '_adaptiveSM'
        return dir_name

    def reset_parameters(self):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s =====' % self.__class__.__name__)
        # see https://github.com/pytorch/fairseq/blob/master/fairseq/models/transformer.py
        # embedding
        nn.init.normal_(self.embed.weight, mean=0., std=self.d_model**-0.5)
        nn.init.constant_(self.embed.weight[self.pad], 0)
        # output layer
        if self.output is not None and not self.tie_embedding:
            nn.init.xavier_uniform_(self.output.weight)
            nn.init.constant_(self.output.bias, 0.)
            # nn.init.normal_(self.embed.weight, mean=0., std=self.d_model**-0.5)

    def embed_token_id(self, indices):
        """Embed token IDs.

        Args:
            indices (LongTensor): `[B]`
        Returns:
            ys_emb (FloatTensor): `[B, vocab, emb_dim]`

        """
        if self.embed_cache is None or self.training:
            ys_emb = self.embed(indices)
        else:
            ys_emb = self.embed_cache[indices]
        return ys_emb

    def decode(self, ys, state=None, mems=None, cache=None, incremental=False):
        """Decode function.

        Args:
            ys (LongTensor): `[B, L]`
            state (List): dummy interfance for RNNLM
            mems (List): length `n_layers` (inter-utterance),
                each of which contains a FloatTensor of size `[B, mlen, d_model]`
            cache (List): length `n_layers` (intra-utterance),
                each of which contains a FloatTensor of size `[B, L-1, d_model]`
            incremental (bool): ASR decoding mode
        Returns:
            logits (FloatTensor): `[B, L, vocab]`
            out (FloatTensor): `[B, L, d_model]`
            new_cache (List): length `n_layers`,
                each of which contains a FloatTensor of size `[B, L, d_model]`

        """
        # for ASR decoding
        if cache is None:
            cache = [None] * self.n_layers  # 1-th to L-th layer

        bs, ylen = ys.size()[:2]
        n_hist = 0
        if incremental and cache[0] is not None:
            n_hist = cache[0].size(1)
            ylen += n_hist

        # Create the self-attention mask
        causal_mask = ys.new_ones(ylen, ylen).byte()
        causal_mask = torch.tril(causal_mask).unsqueeze(0)
        causal_mask = causal_mask.repeat([bs, 1, 1])  # `[B, L, L]`

        out = self.pos_enc(self.embed_token_id(ys), scale=True, offset=max(0, n_hist))  # scaled + dropout

        new_cache = [None] * self.n_layers
        hidden_states = [out]
        for lth, layer in enumerate(self.layers):
            out = layer(out, causal_mask, cache=cache[lth])
            if incremental:
                new_cache[lth] = out
            elif lth < self.n_layers - 1:
                hidden_states.append(out)
                # NOTE: outputs from the last layer is not used for cache
            if not self.training and layer.yy_aws is not None:
                setattr(self, 'yy_aws_layer%d' % lth, tensor2np(layer.yy_aws))
        out = self.norm_out(out)
        if self.adaptive_softmax is None:
            logits = self.output(out)
        else:
            logits = out

        return logits, out, new_cache

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
