# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer decoder (including CTC loss calculation)."""

import copy
from distutils.util import strtobool
from distutils.version import LooseVersion
import logging
import math
import numpy as np
import random
import torch
import torch.nn as nn

from neural_sp.models.criterion import cross_entropy_lsm
from neural_sp.models.lm.rnnlm import RNNLM
from neural_sp.models.modules.positional_embedding import PositionalEncoding
from neural_sp.models.modules.transformer import TransformerDecoderBlock
from neural_sp.models.seq2seq.decoders.beam_search import BeamSearch
from neural_sp.models.seq2seq.decoders.ctc import (
    CTC,
    CTCPrefixScore
)
from neural_sp.models.seq2seq.decoders.decoder_base import DecoderBase
from neural_sp.models.torch_utils import (
    append_sos_eos,
    compute_accuracy,
    make_pad_mask,
    tensor2np,
    tensor2scalar
)

random.seed(1)

logger = logging.getLogger(__name__)

torch_12_plus = LooseVersion("1.3") > LooseVersion(torch.__version__) >= LooseVersion("1.2")


class TransformerDecoder(DecoderBase):
    """Transformer decoder.

    Args:
        special_symbols (dict):
            eos (int): index for <eos> (shared with <sos>)
            unk (int): index for <unk>
            pad (int): index for <pad>
            blank (int): index for <blank>
        enc_n_units (int): number of units of encoder outputs
        attn_type (str): type of attention mechanism
        n_heads (int): number of attention heads
        n_layers (int): number of self-attention layers
        d_model (int): dimension of MultiheadAttentionMechanism
        d_ff (int): dimension of PositionwiseFeedForward
        ffn_bottleneck_dim (int): bottleneck dimension for light-weight FFN layer
        pe_type (str): type of positional encoding
        layer_norm_eps (float): epsilon value for layer normalization
        ffn_activation (str): nonolinear function for PositionwiseFeedForward
        vocab (int): number of nodes in softmax layer
        tie_embedding (bool): tie parameters of embedding and output layers
        dropout (float): dropout probability for linear layers
        dropout_emb (float): dropout probability for embedding layer
        dropout_att (float): dropout probability for attention distributions
        dropout_layer (float): LayerDrop probability for layers
        dropout_head (float): HeadDrop probability for attention heads
        lsm_prob (float): label smoothing probability
        ctc_weight (float): CTC loss weight
        ctc_lsm_prob (float): label smoothing probability for CTC
        ctc_fc_list (List): fully-connected layer configuration before the CTC softmax
        backward (bool): decode in the backward order
        global_weight (float): global loss weight for multi-task learning
        mtl_per_batch (bool): change mini-batch per task for multi-task training
        param_init (str): parameter initialization method
        mma_chunk_size (int): chunk size for chunkwise attention. -1 means infinite lookback.
        mma_n_heads_mono (int): number of MMA head
        mma_n_heads_chunk (int): number of hard chunkwise attention head
        mma_init_r (int): initial bias value for MMA
        mma_eps (float): epsilon value for MMA
        mma_std (float): standard deviation of Gaussian noise for MMA
        mma_no_denominator (bool): remove demominator in MMA
        mma_1dconv (bool): 1dconv for MMA
        mma_quantity_loss_weight (float): quantity loss weight for MMA
        mma_headdiv_loss_weight (float): head divergence loss for MMA
        latency_metric (str): latency metric
        latency_loss_weight (float): latency loss weight for MMA
        mma_first_layer (int): first layer to enable source-target attention (start from idx:1)
        share_chunkwise_attention (bool): share chunkwise attention in the same layer of MMA
        external_lm (RNNLM): external RNNLM for LM fusion
        lm_fusion (str): type of LM fusion

    """

    def __init__(self, special_symbols,
                 enc_n_units, attn_type, n_heads, n_layers,
                 d_model, d_ff, ffn_bottleneck_dim,
                 pe_type, layer_norm_eps, ffn_activation,
                 vocab, tie_embedding,
                 dropout, dropout_emb, dropout_att, dropout_layer, dropout_head,
                 lsm_prob, ctc_weight, ctc_lsm_prob, ctc_fc_list, backward,
                 global_weight, mtl_per_batch, param_init,
                 mma_chunk_size, mma_n_heads_mono, mma_n_heads_chunk,
                 mma_init_r, mma_eps, mma_std,
                 mma_no_denominator, mma_1dconv,
                 mma_quantity_loss_weight, mma_headdiv_loss_weight,
                 latency_metric, latency_loss_weight,
                 mma_first_layer, share_chunkwise_attention,
                 external_lm, lm_fusion):

        super(TransformerDecoder, self).__init__()

        self.eos = special_symbols['eos']
        self.unk = special_symbols['unk']
        self.pad = special_symbols['pad']
        self.blank = special_symbols['blank']
        self.vocab = vocab
        self.enc_n_units = enc_n_units
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pe_type = pe_type
        self.lsm_prob = lsm_prob
        self.att_weight = global_weight - ctc_weight
        self.ctc_weight = ctc_weight
        self.bwd = backward
        self.mtl_per_batch = mtl_per_batch

        self.prev_spk = ''
        self.lmstate_final = None

        # for attention plot
        self.aws_dict = {}
        self.data_dict = {}

        # for MMA
        self.attn_type = attn_type
        self.quantity_loss_weight = mma_quantity_loss_weight
        self._quantity_loss_weight = mma_quantity_loss_weight  # for curriculum
        self.mma_first_layer = max(1, mma_first_layer)
        self.headdiv_loss_weight = mma_headdiv_loss_weight

        self.latency_metric = latency_metric
        self.latency_loss_weight = latency_loss_weight
        self.ctc_trigger = (self.latency_metric in ['ctc_sync'])
        if self.ctc_trigger:
            assert 0 < self.ctc_weight < 1

        if ctc_weight > 0:
            self.ctc = CTC(eos=self.eos,
                           blank=self.blank,
                           enc_n_units=enc_n_units,
                           vocab=vocab,
                           dropout=dropout,
                           lsm_prob=ctc_lsm_prob,
                           fc_list=ctc_fc_list,
                           param_init=0.1,
                           backward=backward)

        if self.att_weight > 0:
            # token embedding
            self.embed = nn.Embedding(self.vocab, d_model, padding_idx=self.pad)
            self.pos_enc = PositionalEncoding(d_model, dropout_emb, pe_type, param_init)
            # decoder
            self.layers = nn.ModuleList([copy.deepcopy(TransformerDecoderBlock(
                d_model, d_ff, attn_type, n_heads, dropout, dropout_att, dropout_layer,
                layer_norm_eps, ffn_activation, param_init,
                src_tgt_attention=False if lth < mma_first_layer - 1 else True,
                mma_chunk_size=mma_chunk_size,
                mma_n_heads_mono=mma_n_heads_mono,
                mma_n_heads_chunk=mma_n_heads_chunk,
                mma_init_r=mma_init_r,
                mma_eps=mma_eps,
                mma_std=mma_std,
                mma_no_denominator=mma_no_denominator,
                mma_1dconv=mma_1dconv,
                dropout_head=dropout_head,
                lm_fusion=lm_fusion,
                ffn_bottleneck_dim=ffn_bottleneck_dim,
                share_chunkwise_attention=share_chunkwise_attention)) for lth in range(n_layers)])
            self.norm_out = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.output = nn.Linear(d_model, self.vocab)
            if tie_embedding:
                self.output.weight = self.embed.weight

            self.lm = external_lm
            if external_lm is not None:
                self.lm_output_proj = nn.Linear(external_lm.output_dim, d_model)

            self.reset_parameters(param_init)

    @staticmethod
    def add_args(parser, args):
        """Add arguments."""
        group = parser.add_argument_group("Transformer decoder")
        # Transformer common
        if not hasattr(args, 'transformer_layer_norm_eps'):
            group.add_argument('--transformer_ffn_bottleneck_dim', type=int, default=0,
                               help='bottleneck dimension in the FFN layer')
            group.add_argument('--transformer_layer_norm_eps', type=float, default=1e-12,
                               help='epsilon value for layer normalization')
            group.add_argument('--transformer_ffn_activation', type=str, default='relu',
                               choices=['relu', 'gelu', 'gelu_accurate', 'glu', 'swish'],
                               help='nonlinear activation for the FFN layer')
            group.add_argument('--transformer_param_init', type=str, default='xavier_uniform',
                               choices=['xavier_uniform', 'pytorch'],
                               help='parameter initialization')
        # Transformer decoder specific
        group.add_argument('--transformer_dec_d_model', type=int, default=256,
                           help='number of units in the MHA layer for Transformer decoder')
        group.add_argument('--transformer_dec_d_ff', type=int, default=2048,
                           help='number of units in the FFN layer for Transformer decoder')
        group.add_argument('--transformer_dec_n_heads', type=int, default=4,
                           help='number of heads in the MHA layer for Transformer decoder')
        group.add_argument('--transformer_dec_attn_type', type=str, default='scaled_dot',
                           choices=['scaled_dot', 'mocha'],
                           help='type of attention mechasnism for Transformer decoder')
        group.add_argument('--transformer_dec_pe_type', type=str, default='add',
                           choices=['add', 'none', '1dconv3L'],
                           help='type of positional encoding for the Transformer decoder')
        group.add_argument('--dropout_dec_layer', type=float, default=0.0,
                           help='LayerDrop probability for Transformer decoder layers')
        group.add_argument('--dropout_head', type=float, default=0.0,
                           help='HeadDrop probability for masking out a head in the Transformer decoder')
        # MMA specific
        parser.add_argument('--mocha_n_heads_mono', type=int, default=1,
                            help='number of heads for monotonic attention')
        parser.add_argument('--mocha_n_heads_chunk', type=int, default=1,
                            help='number of heads for chunkwise attention')
        parser.add_argument('--mocha_chunk_size', type=int, default=1,
                            help='chunk size for MMA. -1 means infinite lookback.')
        parser.add_argument('--mocha_init_r', type=float, default=-4,
                            help='initialization of bias parameter for monotonic attention')
        parser.add_argument('--mocha_eps', type=float, default=1e-6,
                            help='epsilon value to avoid numerical instability for MMA')
        parser.add_argument('--mocha_std', type=float, default=1.0,
                            help='standard deviation of Gaussian noise for MMA during training')
        parser.add_argument('--mocha_no_denominator', type=strtobool, default=False,
                            help='remove denominator (set to 1) in the alpha recurrence in MMA')
        parser.add_argument('--mocha_1dconv', type=strtobool, default=False,
                            help='1dconv for MMA')
        parser.add_argument('--mocha_quantity_loss_weight', type=float, default=0.0,
                            help='quantity loss weight for MMA')
        parser.add_argument('--mocha_latency_metric', type=str, default='',
                            choices=['', 'ctc_sync'],
                            help='differentiable latency metric for MMA')
        parser.add_argument('--mocha_latency_loss_weight', type=float, default=0.0,
                            help='latency loss weight for MMA')
        group.add_argument('--mocha_first_layer', type=int, default=1,
                           help='the initial layer to have a MMA function')
        group.add_argument('--mocha_head_divergence_loss_weight', type=float, default=0.0,
                           help='head divergence loss weight for MMA')
        group.add_argument('--share_chunkwise_attention', type=strtobool, default=False,
                           help='share chunkwise attention heads among monotonic attention heads in the same layer')

        return parser

    @staticmethod
    def define_name(dir_name, args):
        dir_name += '_' + args.dec_type

        dir_name += str(args.transformer_dec_d_model) + 'dmodel'
        dir_name += str(args.transformer_dec_d_ff) + 'dff'
        if args.transformer_ffn_bottleneck_dim > 0:
            dir_name += str(args.transformer_ffn_bottleneck_dim) + 'bn'
        dir_name += str(args.dec_n_layers) + 'L'
        dir_name += str(args.transformer_dec_n_heads) + 'H'
        dir_name += 'pe' + str(args.transformer_dec_pe_type)
        dir_name += args.transformer_dec_attn_type

        # streaming
        if args.transformer_dec_attn_type == 'mocha':
            dir_name += '_ma' + str(args.mocha_n_heads_mono) + 'H'
            dir_name += '_ca' + str(args.mocha_n_heads_chunk) + 'H'
            dir_name += '_w' + str(args.mocha_chunk_size)
            dir_name += '_bias' + str(args.mocha_init_r)
            if args.mocha_no_denominator:
                dir_name += '_denom1'
            if args.mocha_1dconv:
                dir_name += '_1dconv'
            if args.mocha_quantity_loss_weight > 0:
                dir_name += '_qua' + str(args.mocha_quantity_loss_weight)
            if args.mocha_head_divergence_loss_weight != 0:
                dir_name += '_headdiv' + str(args.mocha_head_divergence_loss_weight)
            if args.mocha_latency_metric:
                dir_name += '_' + args.mocha_latency_metric
                dir_name += str(args.mocha_latency_loss_weight)
            if args.share_chunkwise_attention:
                dir_name += '_share'
            if args.mocha_first_layer > 1:
                dir_name += '_from' + str(args.mocha_first_layer) + 'L'

        if args.dropout_dec_layer > 0:
            dir_name += '_LD' + str(args.dropout_dec_layer)
        if args.dropout_head > 0:
            dir_name += '_HD' + str(args.dropout_head)
        if args.tie_embedding:
            dir_name += '_tie'

        return dir_name

    def reset_parameters(self, param_init):
        """Initialize parameters."""
        if param_init == 'xavier_uniform':
            logger.info('===== Initialize %s with Xavier uniform distribution =====' % self.__class__.__name__)
            # see https://github.com/pytorch/fairseq/blob/master/fairseq/models/transformer.py
            # embedding
            nn.init.normal_(self.embed.weight, mean=0., std=self.d_model**-0.5)
            nn.init.constant_(self.embed.weight[self.pad], 0.)
            # output layer
            nn.init.xavier_uniform_(self.output.weight)
            # nn.init.normal_(self.output.weight, mean=0., std=self.d_model**-0.5)
            nn.init.constant_(self.output.bias, 0.)

    def forward(self, eouts, elens, ys, task='all',
                teacher_logits=None, recog_params={}, idx2token=None, trigger_points=None):
        """Forward pass.

        Args:
            eouts (FloatTensor): `[B, T, d_model]`
            elens (IntTensor): `[B]`
            ys (List): length `[B]`, each of which contains a list of size `[L]`
            task (str): all/ys*/ys_sub*
            teacher_logits (FloatTensor): `[B, L, vocab]`
            recog_params (dict): parameters for MBR training
            idx2token ():
            trigger_points (np.ndarray): `[B, L]`
        Returns:
            loss (FloatTensor): `[1]`
            observation (dict):

        """
        observation = {'loss': None, 'loss_att': None, 'loss_ctc': None, 'loss_mbr': None,
                       'acc_att': None, 'ppl_att': None}
        loss = eouts.new_zeros(1)

        # CTC loss
        trigger_points = None
        if self.ctc_weight > 0 and (task == 'all' or 'ctc' in task):
            ctc_forced_align = (self.ctc_trigger and self.training) or self.attn_type == 'triggered_attention'
            loss_ctc, trigger_points = self.ctc(eouts, elens, ys, forced_align=ctc_forced_align)
            observation['loss_ctc'] = tensor2scalar(loss_ctc)
            if self.mtl_per_batch:
                loss += loss_ctc
            else:
                loss += loss_ctc * self.ctc_weight

        # XE loss
        if self.att_weight > 0 and (task == 'all' or 'ctc' not in task):
            loss_att, acc_att, ppl_att, losses_auxiliary = self.forward_att(
                eouts, elens, ys, trigger_points=trigger_points)
            observation['loss_att'] = tensor2scalar(loss_att)
            observation['acc_att'] = acc_att
            observation['ppl_att'] = ppl_att
            if self.attn_type == 'mocha':
                if self._quantity_loss_weight > 0:
                    loss_att += losses_auxiliary['loss_quantity'] * self._quantity_loss_weight
                observation['loss_quantity'] = tensor2scalar(losses_auxiliary['loss_quantity'])
            if self.headdiv_loss_weight > 0:
                loss_att += losses_auxiliary['loss_headdiv'] * self.headdiv_loss_weight
                observation['loss_headdiv'] = tensor2scalar(losses_auxiliary['loss_headdiv'])
            if self.latency_metric:
                observation['loss_latency'] = tensor2scalar(losses_auxiliary['loss_latency']) if self.training else 0
                if self.latency_metric != 'decot' and self.latency_loss_weight > 0:
                    loss_att += losses_auxiliary['loss_latency'] * self.latency_loss_weight
            if self.mtl_per_batch:
                loss += loss_att
            else:
                loss += loss_att * self.att_weight

        observation['loss'] = tensor2scalar(loss)
        return loss, observation

    def forward_att(self, eouts, elens, ys, trigger_points=None):
        """Compute XE loss for the Transformer decoder.

        Args:
            eouts (FloatTensor): `[B, T, d_model]`
            elens (IntTensor): `[B]`
            ys (List): length `[B]`, each of which contains a list of size `[L]`
            trigger_points (IntTensor): `[B, L]`
        Returns:
            loss (FloatTensor): `[1]`
            acc (float): accuracy for token prediction
            ppl (float): perplexity
            losses_auxiliary (dict):

        """
        losses_auxiliary = {}

        # Append <sos> and <eos>
        ys_in, ys_out, ylens = append_sos_eos(ys, self.eos, self.eos, self.pad, self.device, self.bwd)
        if not self.training:
            self.data_dict['elens'] = tensor2np(elens)
            self.data_dict['ylens'] = tensor2np(ylens)
            self.data_dict['ys'] = tensor2np(ys_out)

        # Create target self-attention mask
        bs, ymax = ys_in.size()[:2]
        tgt_mask = (ys_out != self.pad).unsqueeze(1).repeat([1, ymax, 1])
        causal_mask = tgt_mask.new_ones(ymax, ymax, dtype=tgt_mask.dtype)
        if torch_12_plus:
            causal_mask = causal_mask.byte()
        causal_mask = torch.tril(causal_mask, out=causal_mask).unsqueeze(0)
        tgt_mask = tgt_mask & causal_mask  # `[B, L (query), L (key)]`

        # Create source-target mask
        src_mask = make_pad_mask(elens.to(self.device)).unsqueeze(1).repeat([1, ymax, 1])  # `[B, L, T]`

        # Create attention padding mask for quantity loss
        if self.attn_type == 'mocha':
            attn_mask = (ys_out != self.pad).unsqueeze(1).unsqueeze(3)  # `[B, 1, L, 1]`
        else:
            attn_mask = None

        # external LM integration
        lmout = None
        if self.lm is not None:
            self.lm.eval()
            with torch.no_grad():
                lmout, lmstate, _ = self.lm.predict(ys_in, None)
            lmout = self.lm_output_proj(lmout)

        out = self.pos_enc(self.embed(ys_in))  # scaled + dropout

        xy_aws_layers = []
        xy_aws = None
        for lth, layer in enumerate(self.layers):
            out = layer(out, tgt_mask, eouts, src_mask, mode='parallel', lmout=lmout)
            # Attention padding
            xy_aws = layer.xy_aws
            if xy_aws is not None and self.attn_type == 'mocha':
                xy_aws_masked = xy_aws.masked_fill_(attn_mask.expand_as(xy_aws) == 0, 0)
                # NOTE: attention padding is quite effective for quantity loss
                xy_aws_layers.append(xy_aws_masked.clone())
            if not self.training:
                self.aws_dict['yy_aws_layer%d' % lth] = tensor2np(layer.yy_aws)
                self.aws_dict['xy_aws_layer%d' % lth] = tensor2np(layer.xy_aws)
                self.aws_dict['xy_aws_beta_layer%d' % lth] = tensor2np(layer.xy_aws_beta)
                self.aws_dict['xy_aws_p_choose%d' % lth] = tensor2np(layer.xy_aws_p_choose)
                self.aws_dict['yy_aws_lm_layer%d' % lth] = tensor2np(layer.yy_aws_lm)
        logits = self.output(self.norm_out(out))

        # Compute XE loss (+ label smoothing)
        loss, ppl = cross_entropy_lsm(logits, ys_out, self.lsm_prob, self.pad, self.training)

        # Quantity loss
        losses_auxiliary['loss_quantity'] = 0.
        if self.attn_type == 'mocha':
            # Average over all heads across all layers
            n_tokens_ref = tgt_mask[:, -1, :].sum(1).float()  # `[B]`
            # NOTE: count <eos> tokens
            n_tokens_pred = sum([torch.abs(aws.sum(3).sum(2).sum(1) / aws.size(1))
                                 for aws in xy_aws_layers])  # `[B]`
            n_tokens_pred /= len(xy_aws_layers)
            losses_auxiliary['loss_quantity'] = torch.mean(torch.abs(n_tokens_pred - n_tokens_ref))

        # Compute token-level accuracy in teacher-forcing
        acc = compute_accuracy(logits, ys_out, self.pad)

        return loss, acc, ppl, losses_auxiliary

    def greedy(self, eouts, elens, max_len_ratio, idx2token,
               exclude_eos=False, refs_id=None, utt_ids=None, speakers=None,
               cache_states=True):
        """Greedy decoding.

        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            elens (IntTensor): `[B]`
            max_len_ratio (int): maximum sequence length of tokens
            idx2token (): converter from index to token
            exclude_eos (bool): exclude <eos> from hypothesis
            refs_id (List): reference list
            utt_ids (List): utterance id list
            speakers (List): speaker list
            cache_states (bool): cache decoder states for fast decoding
        Returns:
            hyps (List): length `[B]`, each of which contains arrays of size `[L]`
            aws (List): length `[B]`, each of which contains arrays of size `[H * n_layers, L, T]`

        """
        bs, xmax = eouts.size()[:2]
        ys = eouts.new_zeros((bs, 1), dtype=torch.int64).fill_(self.eos)

        cache = [None] * self.n_layers

        hyps_batch = []
        ylens = torch.zeros(bs).int()
        eos_flags = [False] * bs
        xy_aws_layers_steps = []
        ymax = math.ceil(xmax * max_len_ratio)
        for i in range(ymax):
            causal_mask = eouts.new_ones(i + 1, i + 1, dtype=torch.uint8)
            if torch_12_plus:
                causal_mask = causal_mask.byte()
            causal_mask = torch.tril(causal_mask, out=causal_mask).unsqueeze(0).repeat([bs, 1, 1])

            new_cache = [None] * self.n_layers
            xy_aws_layers = []
            out = self.pos_enc(self.embed(ys))  # scaled + dropout
            for lth, layer in enumerate(self.layers):
                out = layer(out, causal_mask, eouts, None, cache=cache[lth])
                new_cache[lth] = out
                if layer.xy_aws is not None:
                    xy_aws_layers.append(layer.xy_aws[:, :, -1:])

            if cache_states:
                cache = new_cache[:]

            # Pick up 1-best
            y = self.output(self.norm_out(out))[:, -1:].argmax(-1)
            hyps_batch += [y]
            xy_aws_layers = torch.stack(xy_aws_layers, dim=2)  # `[B, H, n_layers, 1, T]`
            xy_aws_layers_steps.append(xy_aws_layers)

            # Count lengths of hypotheses
            for b in range(bs):
                if not eos_flags[b]:
                    if y[b].item() == self.eos:
                        eos_flags[b] = True
                    ylens[b] += 1  # include <eos>

            # Break if <eos> is outputed in all mini-batch
            if sum(eos_flags) == bs:
                break
            if i == ymax - 1:
                break

            ys = torch.cat([ys, y], dim=-1)

        # Concatenate in L dimension
        hyps_batch = tensor2np(torch.cat(hyps_batch, dim=1))
        xy_aws_layers_steps = torch.cat(xy_aws_layers_steps, dim=-2)  # `[B, H, n_layers, L, T]`
        xy_aws_layers_steps = xy_aws_layers_steps.reshape(bs, self.n_heads * self.n_layers, ys.size(1), xmax)
        xy_aws = tensor2np(xy_aws_layers_steps)

        # Truncate by the first <eos> (<sos> in case of the backward decoder)
        if self.bwd:
            # Reverse the order
            hyps = [hyps_batch[b, :ylens[b]][::-1] for b in range(bs)]
            aws = [xy_aws[b, :, :ylens[b], :][:, ::-1] for b in range(bs)]
        else:
            hyps = [hyps_batch[b, :ylens[b]] for b in range(bs)]
            aws = [xy_aws[b, :, :ylens[b], :] for b in range(bs)]

        # Exclude <eos> (<sos> in case of the backward decoder)
        if exclude_eos:
            if self.bwd:
                hyps = [hyps[b][1:] if eos_flags[b] else hyps[b] for b in range(bs)]
                aws = [aws[b][:, 1:] if eos_flags[b] else aws[b] for b in range(bs)]
            else:
                hyps = [hyps[b][:-1] if eos_flags[b] else hyps[b] for b in range(bs)]
                aws = [aws[b][:, :-1] if eos_flags[b] else aws[b] for b in range(bs)]

        if idx2token is not None:
            for b in range(bs):
                if utt_ids is not None:
                    logger.debug('Utt-id: %s' % utt_ids[b])
                if refs_id is not None and self.vocab == idx2token.vocab:
                    logger.debug('Ref: %s' % idx2token(refs_id[b]))
                if self.bwd:
                    logger.debug('Hyp: %s' % idx2token(hyps[b][::-1]))
                else:
                    logger.debug('Hyp: %s' % idx2token(hyps[b]))
                logger.info('=' * 200)
                # NOTE: do not show with logger.info here

        return hyps, aws

    def beam_search(self, eouts, elens, params, idx2token=None,
                    lm=None, lm_second=None, lm_second_bwd=None, ctc_log_probs=None,
                    nbest=1, exclude_eos=False,
                    refs_id=None, utt_ids=None, speakers=None,
                    ensmbl_eouts=[], ensmbl_elens=[], ensmbl_decs=[], cache_states=True):
        """Beam search decoding.

        Args:
            eouts (FloatTensor): `[B, T, d_model]`
            elens (IntTensor): `[B]`
            params (dict): hyperparameters for decoding
            idx2token (): converter from index to token
            lm (torch.nn.module): firsh path LM
            lm_second (torch.nn.module): second path LM
            lm_second_bwd (torch.nn.module): secoding path backward LM
            ctc_log_probs (FloatTensor):
            nbest (int):
            exclude_eos (bool): exclude <eos> from hypothesis
            refs_id (List): reference list
            utt_ids (List): utterance id list
            speakers (List): speaker list
            ensmbl_eouts (List[FloatTensor]): encoder outputs for ensemble models
            ensmbl_elens (List[IntTensor]) encoder outputs for ensemble models
            ensmbl_decs (List[torch.nn.Module): decoders for ensemble models
            cache_states (bool): cache decoder states for fast decoding
        Returns:
            nbest_hyps_idx (List): length `[B]`, each of which contains list of N hypotheses
            aws (List): length `[B]`, each of which contains arrays of size `[H, L, T]`
            scores (List):

        """
        bs, xmax, _ = eouts.size()
        n_models = len(ensmbl_decs) + 1

        beam_width = params['recog_beam_width']
        assert 1 <= nbest <= beam_width
        ctc_weight = params['recog_ctc_weight']
        max_len_ratio = params['recog_max_len_ratio']
        min_len_ratio = params['recog_min_len_ratio']
        lp_weight = params['recog_length_penalty']
        length_norm = params['recog_length_norm']
        lm_weight = params['recog_lm_weight']
        lm_weight_second = params['recog_lm_second_weight']
        lm_weight_second_bwd = params['recog_lm_bwd_weight']
        eos_threshold = params['recog_eos_threshold']
        lm_state_carry_over = params['recog_lm_state_carry_over']
        softmax_smoothing = params['recog_softmax_smoothing']
        eps_wait = params['recog_mma_delay_threshold']

        helper = BeamSearch(beam_width, self.eos, ctc_weight, self.device)
        lm = helper.verify_lm_eval_mode(lm, lm_weight)
        lm_second = helper.verify_lm_eval_mode(lm_second, lm_weight_second)
        lm_second_bwd = helper.verify_lm_eval_mode(lm_second_bwd, lm_weight_second_bwd)

        if ctc_log_probs is not None:
            assert ctc_weight > 0
            ctc_log_probs = tensor2np(ctc_log_probs)

        nbest_hyps_idx, aws, scores = [], [], []
        eos_flags = []
        for b in range(bs):
            # Initialization per utterance
            lmstate = None
            ys = eouts.new_zeros((1, 1), dtype=torch.int64).fill_(self.eos)

            # For joint CTC-Attention decoding
            ctc_prefix_scorer = None
            if ctc_log_probs is not None:
                if self.bwd:
                    ctc_prefix_scorer = CTCPrefixScore(ctc_log_probs[b][::-1], self.blank, self.eos)
                else:
                    ctc_prefix_scorer = CTCPrefixScore(ctc_log_probs[b], self.blank, self.eos)

            if speakers is not None:
                if speakers[b] == self.prev_spk:
                    if lm_state_carry_over and isinstance(lm, RNNLM):
                        lmstate = self.lmstate_final
                self.prev_spk = speakers[b]

            end_hyps = []
            hyps = [{'hyp': [self.eos],
                     'ys': ys,
                     'cache': None,
                     'score': 0.,
                     'score_att': 0.,
                     'score_ctc': 0.,
                     'score_lm': 0.,
                     'aws': [None],
                     'lmstate': lmstate,
                     'ensmbl_cache': [[None] * dec.n_layers for dec in ensmbl_decs] if n_models > 1 else None,
                     'ctc_state': ctc_prefix_scorer.initial_state() if ctc_prefix_scorer is not None else None,
                     'quantity_rate': 1.,
                     'streamable': True,
                     'streaming_failed_point': 1000}]
            streamable_global = True
            ymax = math.ceil(elens[b] * max_len_ratio)
            for i in range(ymax):
                # batchfy all hypotheses for batch decoding
                cache = [None] * self.n_layers
                if cache_states and i > 0:
                    for lth in range(self.n_layers):
                        cache[lth] = torch.cat([beam['cache'][lth] for beam in hyps], dim=0)
                ys = eouts.new_zeros((len(hyps), i + 1), dtype=torch.int64)
                for j, beam in enumerate(hyps):
                    ys[j, :] = beam['ys']
                if i > 0:
                    xy_aws_prev = torch.cat([beam['aws'][-1] for beam in hyps], dim=0)  # `[B, n_layers, H_ma, 1, klen]`
                else:
                    xy_aws_prev = None

                # Update LM states for shallow fusion
                y_lm = ys[:, -1:].clone()  # NOTE: this is important
                _, lmstate, scores_lm = helper.update_rnnlm_state_batch(lm, hyps, y_lm)

                # for the main model
                causal_mask = eouts.new_ones(i + 1, i + 1, dtype=torch.uint8)
                if torch_12_plus:
                    causal_mask = causal_mask.byte()
                causal_mask = torch.tril(causal_mask, out=causal_mask).unsqueeze(0).repeat([ys.size(0), 1, 1])

                out = self.pos_enc(self.embed(ys))  # scaled + dropout

                n_heads_total = 0
                eouts_b = eouts[b:b + 1, :elens[b]].repeat([ys.size(0), 1, 1])
                new_cache = [None] * self.n_layers
                xy_aws_layers = []
                xy_aws = None
                lth_s = self.mma_first_layer - 1
                for lth, layer in enumerate(self.layers):
                    out = layer(
                        out, causal_mask, eouts_b, None,
                        cache=cache[lth],
                        xy_aws_prev=xy_aws_prev[:, lth - lth_s] if lth >= lth_s and i > 0 else None,
                        eps_wait=eps_wait)
                    xy_aws = layer.xy_aws

                    new_cache[lth] = out
                    if xy_aws is not None:
                        xy_aws_layers.append(xy_aws)
                logits = self.output(self.norm_out(out[:, -1]))
                probs = torch.softmax(logits * softmax_smoothing, dim=1)
                xy_aws_layers = torch.stack(xy_aws_layers, dim=1)  # `[B, H, n_layers, L, T]`

                # Ensemble initialization
                ensmbl_cache = [[None] * dec.n_layers for dec in ensmbl_decs]
                if n_models > 1 and cache_states and i > 0:
                    for i_e, dec in enumerate(ensmbl_decs):
                        for lth in range(dec.n_layers):
                            ensmbl_cache[i_e][lth] = torch.cat([beam['ensmbl_cache'][i_e][lth] for beam in hyps], dim=0)

                # for the ensemble
                ensmbl_new_cache = [[None] * dec.n_layers for dec in ensmbl_decs]
                for i_e, dec in enumerate(ensmbl_decs):
                    out_e = dec.pos_enc(dec.embed(ys))  # scaled + dropout
                    eouts_e = ensmbl_eouts[i_e][b:b + 1, :elens[b]].repeat([ys.size(0), 1, 1])
                    for lth in range(dec.n_layers):
                        out_e = dec.layers[lth](out_e, causal_mask, eouts_e, None,
                                                cache=ensmbl_cache[i_e][lth])
                        ensmbl_new_cache[i_e][lth] = out_e
                    logits_e = dec.output(dec.norm_out(out_e[:, -1]))
                    probs += torch.softmax(logits_e * softmax_smoothing, dim=1)
                    # NOTE: sum in the probability scale (not log-scale)

                # Ensemble
                scores_att = torch.log(probs / n_models)

                new_hyps = []
                for j, beam in enumerate(hyps):
                    # Attention scores
                    total_scores_att = beam['score_att'] + scores_att[j:j + 1]
                    total_scores = total_scores_att * (1 - ctc_weight)

                    # Add LM score <before> top-K selection
                    if lm is not None:
                        total_scores_lm = beam['score_lm'] + scores_lm[j:j + 1, -1]
                        total_scores += total_scores_lm * lm_weight
                    else:
                        total_scores_lm = eouts.new_zeros(1, self.vocab)

                    total_scores_topk, topk_ids = torch.topk(
                        total_scores, k=beam_width, dim=1, largest=True, sorted=True)

                    # Add length penalty
                    if lp_weight > 0:
                        total_scores_topk += (len(beam['hyp'][1:]) + 1) * lp_weight

                    # Add CTC score
                    new_ctc_states, total_scores_ctc, total_scores_topk = helper.add_ctc_score(
                        beam['hyp'], topk_ids, beam['ctc_state'],
                        total_scores_topk, ctc_prefix_scorer)

                    new_aws = beam['aws'] + [xy_aws_layers[j:j + 1, :, :, -1:]]
                    aws_j = torch.cat(new_aws[1:], dim=3)  # `[1, H, n_layers, L, T]`
                    streaming_failed_point = beam['streaming_failed_point']

                    # forward direction
                    for k in range(beam_width):
                        idx = topk_ids[0, k].item()
                        length_norm_factor = len(beam['hyp'][1:]) + 1 if length_norm else 1
                        total_score = total_scores_topk[0, k].item() / length_norm_factor

                        if idx == self.eos:
                            # Exclude short hypotheses
                            if len(beam['hyp'][1:]) < elens[b] * min_len_ratio:
                                continue
                            # EOS threshold
                            max_score_no_eos = scores_att[j, :idx].max(0)[0].item()
                            max_score_no_eos = max(max_score_no_eos, scores_att[j, idx + 1:].max(0)[0].item())
                            if scores_att[j, idx].item() <= eos_threshold * max_score_no_eos:
                                continue

                        quantity_rate = 1.
                        if self.attn_type == 'mocha':
                            n_tokens_hyp_k = i + 1
                            n_quantity_k = aws_j[:, :, :, :n_tokens_hyp_k].int().sum().item()
                            quantity_diff = n_tokens_hyp_k * n_heads_total - n_quantity_k

                            if quantity_diff != 0:
                                if idx == self.eos:
                                    n_tokens_hyp_k -= 1  # NOTE: do not count <eos> for streamability
                                    n_quantity_k = aws_j[:, :, :, :n_tokens_hyp_k].int().sum().item()
                                else:
                                    streamable_global = False
                                if n_tokens_hyp_k * n_heads_total == 0:
                                    quantity_rate = 0
                                else:
                                    quantity_rate = n_quantity_k / (n_tokens_hyp_k * n_heads_total)

                            if beam['streamable'] and not streamable_global:
                                streaming_failed_point = i

                        new_hyps.append(
                            {'hyp': beam['hyp'] + [idx],
                             'ys': torch.cat([beam['ys'], eouts.new_zeros((1, 1), dtype=torch.int64).fill_(idx)], dim=-1),
                             'cache': [new_cache_l[j:j + 1] for new_cache_l in new_cache] if cache_states else cache,
                             'score': total_score,
                             'score_att': total_scores_att[0, idx].item(),
                             'score_ctc': total_scores_ctc[k].item(),
                             'score_lm': total_scores_lm[0, idx].item(),
                             'aws': new_aws,
                             'lmstate': {'hxs': lmstate['hxs'][:, j:j + 1],
                                         'cxs': lmstate['cxs'][:, j:j + 1]} if lmstate is not None else None,
                             'ctc_state': new_ctc_states[k] if ctc_prefix_scorer is not None else None,
                             'ensmbl_cache': [[new_cache_e_l[j:j + 1] for new_cache_e_l in new_cache_e] for new_cache_e in ensmbl_new_cache] if cache_states else None,
                             'streamable': streamable_global,
                             'streaming_failed_point': streaming_failed_point,
                             'quantity_rate': quantity_rate})

                # Local pruning
                new_hyps_sorted = sorted(new_hyps, key=lambda x: x['score'], reverse=True)[:beam_width]

                # Remove complete hypotheses
                new_hyps, end_hyps, is_finish = helper.remove_complete_hyp(
                    new_hyps_sorted, end_hyps, prune=True)
                hyps = new_hyps[:]
                if is_finish:
                    break

            # Global pruning
            if len(end_hyps) == 0:
                end_hyps = hyps[:]
            elif len(end_hyps) < nbest and nbest > 1:
                end_hyps.extend(hyps[:nbest - len(end_hyps)])

            # forward second path LM rescoring
            helper.lm_rescoring(end_hyps, lm_second, lm_weight_second,
                                normalize=length_norm, tag='second')

            # backward secodn path LM rescoring
            helper.lm_rescoring(end_hyps, lm_second_bwd, lm_weight_second_bwd,
                                normalize=length_norm, tag='second_bwd')

            # Sort by score
            end_hyps = sorted(end_hyps, key=lambda x: x['score'], reverse=True)

            for j in range(len(end_hyps[0]['aws'][1:])):
                tmp = end_hyps[0]['aws'][j + 1]
                end_hyps[0]['aws'][j + 1] = tmp.view(1, -1, tmp.size(-2), tmp.size(-1))

            # metrics for streaming infernece
            self.streamable = end_hyps[0]['streamable']
            self.quantity_rate = end_hyps[0]['quantity_rate']
            self.last_success_frame_ratio = None

            if idx2token is not None:
                if utt_ids is not None:
                    logger.info('Utt-id: %s' % utt_ids[b])
                assert self.vocab == idx2token.vocab
                logger.info('=' * 200)
                for k in range(len(end_hyps)):
                    if refs_id is not None:
                        logger.info('Ref: %s' % idx2token(refs_id[b]))
                    logger.info('Hyp: %s' % idx2token(
                        end_hyps[k]['hyp'][1:][::-1] if self.bwd else end_hyps[k]['hyp'][1:]))
                    logger.info('num tokens (hyp): %d' % len(end_hyps[k]['hyp'][1:]))
                    logger.info('log prob (hyp): %.7f' % end_hyps[k]['score'])
                    logger.info('log prob (hyp, att): %.7f' % (end_hyps[k]['score_att'] * (1 - ctc_weight)))
                    if ctc_prefix_scorer is not None:
                        logger.info('log prob (hyp, ctc): %.7f' % (end_hyps[k]['score_ctc'] * ctc_weight))
                    if lm is not None:
                        logger.info('log prob (hyp, first-path lm): %.7f' %
                                    (end_hyps[k]['score_lm'] * lm_weight))
                    if lm_second is not None:
                        logger.info('log prob (hyp, second-path lm): %.7f' %
                                    (end_hyps[k]['score_lm_second'] * lm_weight_second))
                    if lm_second_bwd is not None:
                        logger.info('log prob (hyp, second-path lm, reverse): %.7f' %
                                    (end_hyps[k]['score_lm_second_bwd'] * lm_weight_second_bwd))
                    if self.attn_type == 'mocha':
                        logger.info('streamable: %s' % end_hyps[k]['streamable'])
                        logger.info('streaming failed point: %d' % (end_hyps[k]['streaming_failed_point'] + 1))
                        logger.info('quantity rate [%%]: %.2f' % (end_hyps[k]['quantity_rate'] * 100))
                    logger.info('-' * 50)

                if self.attn_type == 'mocha' and end_hyps[0]['streaming_failed_point'] < 1000:
                    assert not self.streamable
                    aws_last_success = end_hyps[0]['aws'][1:][end_hyps[0]['streaming_failed_point'] - 1]
                    rightmost_frame = max(0, aws_last_success[0, :, 0].nonzero()[:, -1].max().item()) + 1
                    frame_ratio = rightmost_frame * 100 / xmax
                    self.last_success_frame_ratio = frame_ratio
                    logger.info('streaming last success frame ratio: %.2f' % frame_ratio)

            # N-best list
            if self.bwd:
                # Reverse the order
                nbest_hyps_idx += [[np.array(end_hyps[n]['hyp'][1:][::-1]) for n in range(nbest)]]
                aws += [[tensor2np(torch.cat(end_hyps[n]['aws'][1:][::-1], dim=2).squeeze(0)) for n in range(nbest)]]
            else:
                nbest_hyps_idx += [[np.array(end_hyps[n]['hyp'][1:]) for n in range(nbest)]]
                aws += [[tensor2np(torch.cat(end_hyps[n]['aws'][1:], dim=2).squeeze(0)) for n in range(nbest)]]
            scores += [[end_hyps[n]['score_att'] for n in range(nbest)]]

            # Check <eos>
            eos_flags.append([(end_hyps[n]['hyp'][-1] == self.eos) for n in range(nbest)])

        # Exclude <eos> (<sos> in case of the backward decoder)
        if exclude_eos:
            if self.bwd:
                nbest_hyps_idx = [[nbest_hyps_idx[b][n][1:] if eos_flags[b][n]
                                   else nbest_hyps_idx[b][n] for n in range(nbest)] for b in range(bs)]
                aws = [[aws[b][n][:, 1:] if eos_flags[b][n] else aws[b][n] for n in range(nbest)] for b in range(bs)]
            else:
                nbest_hyps_idx = [[nbest_hyps_idx[b][n][:-1] if eos_flags[b][n]
                                   else nbest_hyps_idx[b][n] for n in range(nbest)] for b in range(bs)]
                aws = [[aws[b][n][:, :-1] if eos_flags[b][n] else aws[b][n] for n in range(nbest)] for b in range(bs)]

        # Store ASR/LM state
        if isinstance(lm, RNNLM):
            self.lmstate_final = end_hyps[0]['lmstate']

        return nbest_hyps_idx, aws, scores
