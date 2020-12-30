# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""RNN decoder for Listen Attend and Spell (LAS) model (including CTC loss calculation)."""

from distutils.util import strtobool
import logging
import math
import numpy as np
import random
import torch
import torch.nn as nn

from neural_sp.evaluators.edit_distance import compute_wer
from neural_sp.models.criterion import (
    cross_entropy_lsm,
    distillation,
    MBR,
)
# from neural_sp.models.criterion import minimum_bayes_risk
from neural_sp.models.lm.rnnlm import RNNLM
from neural_sp.models.lm.transformerlm import TransformerLM
from neural_sp.models.lm.transformer_xl import TransformerXL
from neural_sp.models.modules.attention import AttentionMechanism
from neural_sp.models.modules.gmm_attention import GMMAttention
from neural_sp.models.modules.initialization import init_with_uniform
from neural_sp.models.modules.mocha import MoChA
from neural_sp.models.modules.multihead_attention import MultiheadAttentionMechanism
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
    repeat,
    pad_list,
    np2tensor,
    tensor2np,
    tensor2scalar,
)


random.seed(1)

logger = logging.getLogger(__name__)


class RNNDecoder(DecoderBase):
    """RNN decoder.

    Args:
        special_symbols (dict):
            eos (int): index for <eos> (shared with <sos>)
            unk (int): index for <unk>
            pad (int): index for <pad>
            blank (int): index for <blank>
        enc_n_units (int): number of units of encoder outputs
        attn_type (str): type of attention mechanism
        rnn_type (str): lstm/gru
        n_units (int): number of units in each RNN layer
        n_projs (int): number of units in each projection layer
        n_layers (int): number of RNN layers
        bottleneck_dim (int): dimension of bottleneck layer before the softmax layer for label generation
        emb_dim (int): dimension of embedding in target spaces.
        vocab (int): number of nodes in softmax layer
        tie_embedding (bool): tie parameters of embedding and output layers
        attn_dim (int): dimension of attention space
        attn_sharpening_factor (float): sharpening factor in softmax for attention
        attn_sigmoid_smoothing (bool): replace softmax with sigmoid for attention calculation
        attn_conv_out_channels (int): channel size of convolution in location-aware attention
        attn_conv_kernel_size (int): kernel size of convolution in location-aware attention
        attn_n_heads (int): number of attention heads
        dropout (float): dropout probability for RNN layer
        dropout_emb (float): dropout probability for embedding layer
        dropout_att (float): dropout probability for attention distributions
        lsm_prob (float): label smoothing probability
        ss_prob (float): scheduled sampling probability
        ctc_weight (float): CTC loss weight
        ctc_lsm_prob (float): label smoothing probability for CTC
        ctc_fc_list (List): fully-connected layer configuration before the CTC softmax
        mbr_training (bool): MBR training
        mbr_ce_weight (float): CE weight for regularization during MBR training
        external_lm (RNNLM): external RNNLM for LM fusion/initialization
        lm_fusion (str): type of LM fusion
        lm_init (bool): initialize decoder with pre-trained LM
        backward (bool): decode in the backward order
        global_weight (float): global loss weight for multi-task learning
        mtl_per_batch (bool): change mini-batch per task for multi-task training
        param_init (float): parameter initialization
        mocha_chunk_size (int): chunk size for MoChA
        mocha_n_heads_mono (int): number of monotonic head for MoChA
        mocha_init_r (int): initial bias value for MoChA
        mocha_eps (float): epsilon value for MoChA
        mocha_std (float): standard deviation of Gaussian noise for MoChA
        mocha_no_denominator (bool): remove denominator in MoChA
        mocha_1dconv (bool): 1dconv for MoChA
        mocha_decot_lookahead (int): lookahead frames of DeCoT for MoChA
        quantity_loss_weight (float): quantity loss weight for MoChA
        latency_metric (str): latency metric for MoChA
        latency_loss_weight (float): latency loss weight for MoChA
        gmm_attn_n_mixtures (int): number of mixtures for GMM attention
        replace_sos (bool): replace <sos> with special tokens
        distil_weight (float): soft label weight for knowledge distillation
        discourse_aware (str): state_carry_over

    """

    def __init__(self, special_symbols,
                 enc_n_units, attn_type, rnn_type, n_units, n_projs, n_layers,
                 bottleneck_dim, emb_dim, vocab, tie_embedding,
                 attn_dim, attn_sharpening_factor, attn_sigmoid_smoothing,
                 attn_conv_out_channels, attn_conv_kernel_size, attn_n_heads,
                 dropout, dropout_emb, dropout_att,
                 lsm_prob, ss_prob,
                 ctc_weight, ctc_lsm_prob, ctc_fc_list,
                 mbr_training, mbr_ce_weight,
                 external_lm, lm_fusion, lm_init,
                 backward, global_weight, mtl_per_batch, param_init,
                 mocha_chunk_size, mocha_n_heads_mono,
                 mocha_init_r, mocha_eps, mocha_std, mocha_no_denominator,
                 mocha_1dconv, mocha_decot_lookahead, quantity_loss_weight,
                 latency_metric, latency_loss_weight,
                 gmm_attn_n_mixtures, replace_sos, distillation_weight, discourse_aware):

        super(RNNDecoder, self).__init__()

        self.eos = special_symbols['eos']
        self.unk = special_symbols['unk']
        self.pad = special_symbols['pad']
        self.blank = special_symbols['blank']
        self.vocab = vocab
        self.attn_type = attn_type
        self.rnn_type = rnn_type
        assert rnn_type in ['lstm', 'gru']
        self.enc_n_units = enc_n_units
        self.dec_n_units = n_units
        self.n_projs = n_projs
        self.n_layers = n_layers
        self.lsm_prob = lsm_prob
        self.ss_prob = ss_prob
        self._ss_prob = 0  # for curriculum
        self.att_weight = global_weight - ctc_weight
        self.ctc_weight = ctc_weight
        self.lm_fusion = lm_fusion
        self.bwd = backward
        self.mtl_per_batch = mtl_per_batch
        self.replace_sos = replace_sos
        self.distil_weight = distillation_weight
        logger.info("Attention weight: %.3f" % self.att_weight)
        logger.info("CTC weight: %.3f" % self.ctc_weight)

        # for mocha and triggered attention
        self.quantity_loss_weight = quantity_loss_weight
        self._quantity_loss_weight = 0  # for curriculum
        self.latency_metric = latency_metric
        self.latency_loss_weight = latency_loss_weight
        if ('ctc_sync' in latency_metric) or attn_type == 'triggered_attention':
            assert 0 < self.ctc_weight < 1

        # for MBR training
        self.mbr_ce_weight = mbr_ce_weight
        self.mbr = MBR.apply if mbr_training else None

        # for contextualization
        self.discourse_aware = discourse_aware
        self.dstate_prev = None
        self._new_session = False

        self.prev_spk = ''
        self.dstates_final = None
        self.lmstate_final = None
        self.lmmemory = None

        # for attention plot
        self.aws_dict = {}
        self.data_dict = {}

        # for CPU decoding
        self.embed_cache = None

        if ctc_weight > 0:
            self.ctc = CTC(eos=self.eos,
                           blank=self.blank,
                           enc_n_units=enc_n_units,
                           vocab=vocab,
                           dropout=dropout,
                           lsm_prob=ctc_lsm_prob,
                           fc_list=ctc_fc_list,
                           param_init=param_init)

        if self.att_weight > 0:
            # Attention layer
            qdim = n_units if n_projs == 0 else n_projs
            if attn_type == 'mocha':
                assert attn_n_heads == 1
                self.score = MoChA(enc_n_units, qdim, attn_dim, enc_n_units,
                                   atype='add',
                                   chunk_size=mocha_chunk_size,
                                   n_heads_mono=mocha_n_heads_mono,
                                   init_r=mocha_init_r,
                                   eps=mocha_eps,
                                   noise_std=mocha_std,
                                   no_denominator=mocha_no_denominator,
                                   conv1d=mocha_1dconv,
                                   sharpening_factor=attn_sharpening_factor,
                                   decot='decot' in latency_metric,
                                   lookahead=mocha_decot_lookahead)
            elif attn_type == 'gmm':
                self.score = GMMAttention(enc_n_units, qdim, attn_dim,
                                          n_mixtures=gmm_attn_n_mixtures)
            else:
                if attn_n_heads > 1:
                    assert attn_type == 'add'
                    self.score = MultiheadAttentionMechanism(
                        enc_n_units, qdim, attn_dim, enc_n_units,
                        n_heads=attn_n_heads,
                        dropout=dropout_att,
                        atype='add')
                else:
                    self.score = AttentionMechanism(
                        enc_n_units, qdim, attn_dim, attn_type,
                        sharpening_factor=attn_sharpening_factor,
                        sigmoid_smoothing=attn_sigmoid_smoothing,
                        conv_out_channels=attn_conv_out_channels,
                        conv_kernel_size=attn_conv_kernel_size,
                        dropout=dropout_att,
                        lookahead=2)

            # Decoder
            self.rnn = nn.ModuleList()
            cell = nn.LSTMCell if rnn_type == 'lstm' else nn.GRUCell
            dec_odim = enc_n_units + emb_dim
            self.proj = repeat(nn.Linear(n_units, n_projs), n_layers) if n_projs > 0 else None
            self.dropout = nn.Dropout(p=dropout)
            for _ in range(n_layers):
                self.rnn += [cell(dec_odim, n_units)]
                dec_odim = n_projs if n_projs > 0 else n_units

            # RNNLM fusion
            if external_lm is not None and lm_fusion:
                self.linear_dec_feat = nn.Linear(dec_odim + enc_n_units, n_units)
                if lm_fusion in ['cold', 'deep']:
                    self.linear_lm_feat = nn.Linear(external_lm.output_dim, n_units)
                    self.linear_lm_gate = nn.Linear(n_units * 2, n_units)
                elif lm_fusion == 'cold_prob':
                    self.linear_lm_feat = nn.Linear(external_lm.vocab, n_units)
                    self.linear_lm_gate = nn.Linear(n_units * 2, n_units)
                else:
                    raise ValueError(lm_fusion)
                self.output_bn = nn.Linear(n_units * 2, bottleneck_dim)
            else:
                self.output_bn = nn.Linear(dec_odim + enc_n_units, bottleneck_dim)

            self.embed = nn.Embedding(vocab, emb_dim, padding_idx=self.pad)
            self.dropout_emb = nn.Dropout(p=dropout_emb)
            assert bottleneck_dim > 0, 'bottleneck_dim must be larger than zero.'
            self.output = nn.Linear(bottleneck_dim, vocab)
            if tie_embedding:
                if emb_dim != bottleneck_dim:
                    raise ValueError('When using the tied flag, n_units must be equal to emb_dim.')
                self.output.weight = self.embed.weight

        self.reset_parameters(param_init)
        # NOTE: LM registration and initialization should be performed after reset_parameters()

        # resister the external RNNLM
        self.lm = external_lm if lm_fusion else None

        # decoder initialization with pre-trained RNNLM
        if lm_init:
            assert self.att_weight > 0
            assert external_lm is not None
            assert external_lm.vocab == vocab, 'vocab'
            assert external_lm.n_units == n_units, 'n_units'
            assert external_lm.emb_dim == emb_dim, 'emb_dim'
            logger.info('===== Initialize the decoder with pre-trained RNNLM')
            assert external_lm.n_projs == 0  # TODO(hirofumi): fix later
            assert external_lm.n_units_cv == enc_n_units, 'enc_n_units'

            # RNN
            for lth in range(external_lm.n_layers):
                for n, p in external_lm.rnn[lth].named_parameters():
                    n = '_'.join(n.split('_')[:2])
                    assert getattr(self.rnn[lth], n).size() == p.size()
                    getattr(self.rnn[lth], n).data = p.data
                    logger.info('Overwrite %s' % n)
            # embedding
            assert self.embed.weight.size() == external_lm.embed.weight.size()
            self.embed.weight.data = external_lm.embed.weight.data
            logger.info('Overwrite %s' % 'embed.weight')

    @staticmethod
    def add_args(parser, args):
        """Add arguments."""
        group = parser.add_argument_group("LAS decoder")
        # common (LAS/RNN-T)
        if not hasattr(args, 'dec_n_units'):
            group.add_argument('--dec_n_units', type=int, default=512,
                               help='number of units in each decoder RNN layer')
            group.add_argument('--dec_n_projs', type=int, default=0,
                               help='number of units in the projection layer after each decoder RNN layer')
            group.add_argument('--dec_bottleneck_dim', type=int, default=1024,
                               help='number of dimensions of the bottleneck layer before the softmax layer')
            group.add_argument('--emb_dim', type=int, default=512,
                               help='number of dimensions in the embedding layer')
        # attention
        group.add_argument('--attn_type', type=str, default='location',
                           choices=['no', 'location', 'add', 'dot',
                                    'luong_dot', 'luong_general', 'luong_concat',
                                    'mocha', 'gmm', 'cif', 'triggered_attention'],
                           help='type of attention mechanism for RNN decoder')
        group.add_argument('--attn_dim', type=int, default=128,
                           help='dimension of the attention layer')
        group.add_argument('--attn_n_heads', type=int, default=1,
                           help='number of heads in the attention layer')
        group.add_argument('--attn_sharpening_factor', type=float, default=1.0,
                           help='sharpening factor')
        group.add_argument('--attn_conv_n_channels', type=int, default=10,
                           help='')
        group.add_argument('--attn_conv_width', type=int, default=201,
                           help='')
        group.add_argument('--attn_sigmoid', type=strtobool, default=False, nargs='?',
                           help='')
        group.add_argument('--gmm_attn_n_mixtures', type=int, default=5,
                           help='number of mixtures for GMM attention')
        # other
        parser.add_argument('--ss_prob', type=float, default=0.0,
                            help='probability of scheduled sampling')
        parser.add_argument('--ss_start_epoch', type=int, default=0,
                            help='epoch to turn on scheduled sampling')
        # streaming
        parser.add_argument('--mocha_n_heads_mono', type=int, default=1,
                            help='number of heads for monotonic attention')
        parser.add_argument('--mocha_n_heads_chunk', type=int, default=1,
                            help='number of heads for chunkwise attention')
        parser.add_argument('--mocha_chunk_size', type=int, default=1,
                            help='chunk size for MoChA. -1 means infinite lookback.')
        parser.add_argument('--mocha_init_r', type=float, default=-4,
                            help='initialization of bias parameter for monotonic attention')
        parser.add_argument('--mocha_eps', type=float, default=1e-6,
                            help='epsilon value to avoid numerical instability for MoChA')
        parser.add_argument('--mocha_std', type=float, default=1.0,
                            help='standard deviation of Gaussian noise for MoChA during training')
        parser.add_argument('--mocha_no_denominator', type=strtobool, default=False,
                            help='remove denominator (set to 1) in the alpha recurrence in MoChA')
        parser.add_argument('--mocha_1dconv', type=strtobool, default=False,
                            help='1dconv for MoChA')
        parser.add_argument('--mocha_quantity_loss_weight', type=float, default=0.0,
                            help='quantity loss weight for MoChA')
        parser.add_argument('--mocha_quantity_loss_start_epoch', type=int, default=0,
                            help='epoch to turn on quantity loss')
        parser.add_argument('--mocha_latency_metric', type=str, default='',
                            choices=['', 'decot', 'minlt',
                                     'ctc_sync', 'decot_ctc_sync', 'interval'],
                            help='latency metric for MoChA')
        parser.add_argument('--mocha_latency_loss_weight', type=float, default=0.0,
                            help='latency loss weight for MoChA')
        parser.add_argument('--mocha_decot_lookahead', type=int, default=0,
                            help='buffer frames in DeCoT')
        return parser

    @staticmethod
    def define_name(dir_name, args):
        dir_name += '_' + args.dec_type

        dir_name += str(args.dec_n_units) + 'H'
        if args.dec_n_projs > 0:
            dir_name += str(args.dec_n_projs) + 'P'
        dir_name += str(args.dec_n_layers) + 'L'

        dir_name += '_' + args.attn_type
        if args.attn_sigmoid:
            dir_name += '_sig'
        if 'mocha' in args.attn_type:
            dir_name += '_w' + str(args.mocha_chunk_size)
            if args.mocha_n_heads_mono > 1:
                dir_name += '_ma' + str(args.mocha_n_heads_mono) + 'H'
            if args.mocha_no_denominator:
                dir_name += '_denom1'
            if args.mocha_1dconv:
                dir_name += '_1dconv'
        elif args.attn_type in ['gmm']:
            dir_name += '_mix' + str(args.gmm_attn_n_mixtures)
        if args.attn_sharpening_factor > 1:
            dir_name += '_temp' + str(args.attn_sharpening_factor)
        if args.mocha_quantity_loss_weight > 0:
            dir_name += '_qua' + str(args.mocha_quantity_loss_weight)
        if args.mocha_latency_metric:
            dir_name += '_' + args.mocha_latency_metric
            if 'decot' in args.mocha_latency_metric:
                dir_name += str(args.mocha_decot_lookahead)
            else:
                dir_name += str(args.mocha_latency_loss_weight)
        if args.attn_n_heads > 1:
            dir_name += '_head' + str(args.attn_n_heads)
        if args.tie_embedding:
            dir_name += '_tie'

        if args.ctc_weight < 1 and args.ss_prob > 0:
            dir_name += '_ss' + str(args.ss_prob)

        return dir_name

    def reset_parameters(self, param_init):
        """Initialize parameters with uniform distribution."""
        logger.info('===== Initialize %s with uniform distribution =====' % self.__class__.__name__)
        for n, p in self.named_parameters():
            if 'score.monotonic_energy.v.weight_g' in n or 'score.monotonic_energy.r' in n:
                logger.info('Skip initialization of %s' % n)
                continue
            if 'score.chunk_energy.v.weight_g' in n or 'score.chunk_energy.r' in n:
                logger.info('Skip initialization of %s' % n)
                continue
            if 'linear_lm_gate.fc.bias' in n and p.dim() == 1:
                # Initialize bias in gating with -1 for cold fusion
                nn.init.constant_(p, -1.)  # bias
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', -1.))
                continue

            init_with_uniform(n, p, param_init)

    def forward(self, eouts, elens, ys, task='all',
                teacher_logits=None,
                recog_params={}, idx2token=None, trigger_points=None):
        """Forward pass.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
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
        if self.ctc_weight > 0 and (task == 'all' or 'ctc' in task):
            ctc_forced_align = (
                'ctc_sync' in self.latency_metric and self.training) or self.attn_type == 'triggered_attention'
            loss_ctc, ctc_trigger_points = self.ctc(eouts, elens, ys, forced_align=ctc_forced_align)
            observation['loss_ctc'] = tensor2scalar(loss_ctc)
            if self.mtl_per_batch:
                loss += loss_ctc
            else:
                loss += loss_ctc * self.ctc_weight
            if self.latency_metric in ['minlt', 'decot', 'decot_ctc_sync'] and trigger_points is not None:
                trigger_points = np2tensor(trigger_points, eouts.device)
        else:
            ctc_trigger_points = None

        # XE loss
        if self.att_weight > 0 and (task == 'all' or 'ctc' not in task) and self.mbr is None:
            loss_att, acc_att, ppl_att, loss_quantity, loss_latency = self.forward_att(
                eouts, elens, ys, teacher_logits=teacher_logits,
                ctc_trigger_points=ctc_trigger_points, forced_trigger_points=trigger_points)
            observation['loss_att'] = tensor2scalar(loss_att)
            observation['acc_att'] = acc_att
            observation['ppl_att'] = ppl_att
            if self.attn_type == 'mocha':
                if self._quantity_loss_weight > 0:
                    loss_att += loss_quantity * self._quantity_loss_weight
                observation['loss_quantity'] = tensor2scalar(loss_quantity)
            if self.latency_metric:
                observation['loss_latency'] = tensor2scalar(loss_latency) if self.training else 0
                if self.latency_loss_weight > 0:
                    loss_att += loss_latency * self.latency_loss_weight
            if self.mtl_per_batch:
                loss += loss_att
            else:
                loss += loss_att * self.att_weight

        # MBR loss
        if self.mbr is not None and (task == 'all' or 'mbr' not in task):
            nbest = recog_params['recog_beam_width']
            alpha = 1.0
            assert nbest >= 2
            loss_mbr = 0.
            loss_ce = 0.
            bs = eouts.size(0)
            for b in range(bs):
                # 1. beam search
                self.eval()
                with torch.no_grad():
                    nbest_hyps_id, _, log_scores = self.beam_search(
                        eouts[b:b + 1], elens[b:b + 1], params=recog_params,
                        nbest=nbest, exclude_eos=True)
                nbest_hyps_id_b = [np.fromiter(y, dtype=np.int64) for y in nbest_hyps_id[0]]
                log_scores_b = np2tensor(np.array(log_scores[0], dtype=np.float32), eouts.device)
                scores_b_norm = torch.softmax(alpha * log_scores_b, dim=-1)  # `[nbest]`
                # print((scores_b_norm * 100).int())

                # 2. calculate expected WER
                wers_b = np2tensor(np.array([
                    compute_wer(ref=idx2token(ys[b]).split(' '),
                                hyp=idx2token(nbest_hyps_id_b[n]).split(' '))[0] / 100
                    for n in range(nbest)], dtype=np.float32), eouts.device)
                exp_wer_b = (scores_b_norm * wers_b).sum()
                grad_b = (scores_b_norm * (wers_b - exp_wer_b)).sum()
                # print(wers_b)
                # print(exp_wer_b)
                # print(scores_b_norm * (wers_b - exp_wer_b))
                # print(grad_b)

                # 3. forward pass (teacher-forcing with hypotheses)
                self.train()
                logits_b = self.forward_mbr(eouts[b:b + 1].repeat([nbest, 1, 1]),
                                            elens[b:b + 1].repeat([nbest]),
                                            nbest_hyps_id_b)
                log_probs_b = torch.log_softmax(logits_b, dim=-1)  # `[nbest, L, vocab]`

                # 4. backward pass (attach gradient)
                _eos = eouts.new_zeros((1,), dtype=torch.int64).fill_(self.eos)
                nbest_hyps_id_b_pad = pad_list([torch.cat([np2tensor(y, eouts.device), _eos], dim=0)
                                                for y in nbest_hyps_id_b], self.pad)
                loss_mbr += self.mbr(log_probs_b, nbest_hyps_id_b_pad, exp_wer_b, grad_b)

                # 4. calculate MBR loss
                # loss_mbr_b, scores_b = minimum_bayes_risk(log_probs_b, nbest_hyps_id_b, wers_b,
                #                                           self.eos, self.pad)
                # loss_mbr += loss_mbr_b

                # 5. CE loss regularization
                loss_ce += self.forward_att(eouts[b:b + 1], elens[b:b + 1], ys[b:b + 1])[0]

                # ys_out_b = append_sos_eos([ys[b]], self.eos, self.eos, self.pad, eouts.device)[1]
                # ys_out_b = ys_out_b.repeat([nbest, 1])
                # # NOTE: truncate to match the lengths
                # ymax = min(logits_b.size(1), ys_out_b.size(1))
                # logits_b = logits_b[:, :ymax].contiguous()
                # ys_out_b = ys_out_b[:, :ymax].contiguous()
                # for k in range(nbest):
                #     loss_ce_k = cross_entropy_lsm(logits_b, ys_out_b, 0, self.pad, self.training)[0]
                #     loss_ce += loss_ce_k * scores_b_norm[k]

            # NOTE: MBR loss is accumlated over N-best and mini-batch
            loss = loss_mbr + loss_ce * self.mbr_ce_weight
            observation['loss_mbr'] = tensor2scalar(loss_mbr)
            observation['loss_att'] = tensor2scalar(loss_ce)

        observation['loss'] = tensor2scalar(loss)
        return loss, observation

    def forward_mbr(self, eouts, elens, ys_hyp):
        """Compute XE loss for the attention-based decoder.

        Args:
            eouts (FloatTensor): `[nbest, T, enc_n_units]`
            elens (IntTensor): `[nbest]`
            ys_hyp (List): length `nbest`, each of which contains a list of size `[L]`
        Returns:
            logits (FloatTensor): `[nbest, L, vocab]`

        """
        bs, xmax = eouts.size()[:2]

        # Append <sos> and <eos>
        ys_in, ys_out, ylens = append_sos_eos(ys_hyp, self.eos, self.eos, self.pad, eouts.device)

        # Initialization
        dstates = self.zero_state(bs)
        cv = eouts.new_zeros(bs, 1, self.enc_n_units)
        self.score.reset()
        aw, aws = None, []
        betas, p_chooses = [], []
        lmout, lmstate = None, None

        ys_emb = self.dropout_emb(self.embed(ys_in))
        src_mask = make_pad_mask(elens.to(eouts.device)).unsqueeze(1)  # `[B, 1, T]`
        logits = []
        for i in range(ys_in.size(1)):
            is_sample = i > 0 and self._ss_prob > 0 and random.random() < self._ss_prob

            # Update LM states for LM fusion
            if self.lm is not None:
                y_lm = self.output(logits[-1]).detach().argmax(-1) if is_sample else ys_in[:, i:i + 1]
                lmout, lmstate, _ = self.lm.predict(y_lm, lmstate)

            # Recurrency -> Score -> Generate
            y_emb = self.dropout_emb(self.embed(
                self.output(logits[-1]).detach().argmax(-1))) if is_sample else ys_emb[:, i:i + 1]
            dstates, cv, aw, attn_state, attn_v = self.decode_step(
                eouts, dstates, cv, y_emb, src_mask, aw, lmout, mode='parallel')
            aws.append(aw)  # `[B, H, 1, T]`
            logits.append(attn_v)
            if attn_state.get('beta', None) is not None:
                betas.append(attn_state['beta'])  # `[B, H, 1, T]`
            if attn_state.get('p_choose', None) is not None:
                p_chooses.append(attn_state['p_choose'])  # `[B, H, 1, T]`
            if self.attn_type in ['gmm']:
                aw = attn_state['myu']

        # for attention plot
        with torch.no_grad():
            aws = torch.cat(aws, dim=2)  # `[B, H, L, T]`
            self.data_dict['elens'] = tensor2np(elens)
            self.data_dict['ylens'] = tensor2np(ylens)
            self.data_dict['ys'] = tensor2np(ys_out)
            self.aws_dict['xy_aws'] = tensor2np(aws)
            if len(betas) > 0:
                betas = torch.cat(betas, dim=2)  # `[B, H, L, T]`
                self.aws_dict['xy_aws_beta'] = tensor2np(betas)
            if len(p_chooses) > 0:
                p_chooses = torch.cat(p_chooses, dim=2)  # `[B, H, L, T]`
                self.aws_dict['xy_aws_p_choose'] = tensor2np(p_chooses)

        logits = self.output(torch.cat(logits, dim=1))
        return logits

    def forward_att(self, eouts, elens, ys,
                    return_logits=False, teacher_logits=None,
                    ctc_trigger_points=None, forced_trigger_points=None):
        """Compute XE loss for the attention-based decoder.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
            ys (List): length `[B]`, each of which contains a list of size `[L]`
            return_logits (bool): return logits for knowledge distillation
            teacher_logits (FloatTensor): `[B, L, vocab]`
            ctc_trigger_points (IntTensor): `[B, L]`
            forced_trigger_points (IntTensor): `[B, L]`
        Returns:
            loss (FloatTensor): `[1]`
            acc (float): accuracy for token prediction
            ppl (float): perplexity
            loss_quantity (FloatTensor): `[1]`
            loss_latency (FloatTensor): `[1]`

        """
        bs, xmax = eouts.size()[:2]

        # Append <sos> and <eos>
        ys_in, ys_out, ylens = append_sos_eos(ys, self.eos, self.eos, self.pad, eouts.device, self.bwd)
        ymax = ys_in.size(1)

        if forced_trigger_points is not None:
            for b in range(bs):
                forced_trigger_points[b, ylens[b] - 1] = elens[b] - 1  # for <eos>

        # Initialization
        dstates = self.zero_state(bs)
        if self.training:
            if self.discourse_aware and not self._new_session:
                dstates = {'dstate': (self.dstate_prev['hxs'], self.dstate_prev['cxs'])}
            self.dstate_prev = {'hxs': [None] * bs, 'cxs': [None] * bs}
            self._new_session = False
        cv = eouts.new_zeros(bs, 1, self.enc_n_units)
        self.score.reset()
        aw, aws = None, []
        betas, p_chooses = [], []
        lmout, lmstate = None, None

        ys_emb = self.dropout_emb(self.embed(ys_in))
        src_mask = make_pad_mask(elens.to(eouts.device)).unsqueeze(1)  # `[B, 1, T]`
        tgt_mask = (ys_out != self.pad).unsqueeze(2)  # `[B, L, 1]`
        logits = []
        for i in range(ymax):
            is_sample = i > 0 and self._ss_prob > 0 and random.random() < self._ss_prob

            # Update LM states for LM fusion
            if self.lm is not None:
                self.lm.eval()
                with torch.no_grad():
                    y_lm = self.output(logits[-1]).detach().argmax(-1) if is_sample else ys_in[:, i:i + 1]
                    lmout, lmstate, _ = self.lm.predict(y_lm, lmstate)

            # Recurrency -> Score -> Generate
            y_emb = self.dropout_emb(self.embed(
                self.output(logits[-1]).detach().argmax(-1))) if is_sample else ys_emb[:, i:i + 1]
            dstates, cv, aw, attn_state, attn_v = self.decode_step(
                eouts, dstates, cv, y_emb, src_mask, aw, lmout, mode='parallel',
                trigger_points=forced_trigger_points[:, i:i + 1] if forced_trigger_points is not None else None)
            aws.append(aw)  # `[B, H, 1, T]`
            logits.append(attn_v)
            if attn_state.get('beta', None) is not None:
                betas.append(attn_state['beta'])  # `[B, H, 1, T]`
            if attn_state.get('p_choose', None) is not None:
                p_chooses.append(attn_state['p_choose'])  # `[B, H, 1, T]`
            if self.attn_type in ['gmm', 'sagmm']:
                aw = attn_state['myu']

            if self.training and self.discourse_aware:
                for b in [b for b, ylen in enumerate(ylens.tolist()) if i == ylen - 1]:
                    self.dstate_prev['hxs'][b] = dstates['dstate'][0][:, b:b + 1].detach()
                    if self.rnn_type == 'lstm':
                        self.dstate_prev['cxs'][b] = dstates['dstate'][1][:, b:b + 1].detach()

        if self.training and self.discourse_aware:
            if bs > 1:
                self.dstate_prev['hxs'] = torch.cat(self.dstate_prev['hxs'], dim=1)
                if self.rnn_type == 'lstm':
                    self.dstate_prev['cxs'] = torch.cat(self.dstate_prev['cxs'], dim=1)
            else:
                self.dstate_prev['hxs'] = self.dstate_prev['hxs'][0]
                if self.rnn_type == 'lstm':
                    self.dstate_prev['cxs'] = self.dstate_prev['cxs'][0]

        logits = self.output(torch.cat(logits, dim=1))

        # for knowledge distillation
        if return_logits:
            return logits

        # for attention plot
        aws = torch.cat(aws, dim=2)  # `[B, H, L, T]`
        if not self.training:
            self.data_dict['elens'] = tensor2np(elens)
            self.data_dict['ylens'] = tensor2np(ylens)
            self.data_dict['ys'] = tensor2np(ys_out)
            self.aws_dict['xy_aws'] = tensor2np(aws)
            if len(betas) > 0:
                betas = torch.cat(betas, dim=2)  # `[B, H, L, T]`
                self.aws_dict['xy_aws_beta'] = tensor2np(betas)
            if len(p_chooses) > 0:
                p_chooses = torch.cat(p_chooses, dim=2)  # `[B, H, L, T]`
                self.aws_dict['xy_p_choose'] = tensor2np(p_chooses)

        n_heads = aws.size(1)  # mono

        # Compute XE sequence loss (+ label smoothing)
        loss, ppl = cross_entropy_lsm(logits, ys_out, self.lsm_prob, self.pad, self.training)

        # Attention padding
        if self.attn_type == 'mocha' or (ctc_trigger_points is not None or forced_trigger_points is not None):
            aws = aws.masked_fill_(tgt_mask.unsqueeze(1).expand_as(aws) == 0, 0)
            # NOTE: attention padding is quite effective for quantity loss

        # Quantity loss
        loss_quantity = 0.
        if self.attn_type == 'mocha':
            # Average over all heads
            n_tokens_pred = aws.sum(3).sum(2).sum(1) / n_heads  # `[B]`
            n_tokens_ref = tgt_mask.squeeze(2).sum(1).float()  # `[B]`
            # NOTE: count <eos> tokens
            loss_quantity = torch.mean(torch.abs(n_tokens_pred - n_tokens_ref))

        # Latency loss
        loss_latency = 0.
        if self.latency_metric == 'interval':
            assert ctc_trigger_points is None
            assert aws.size(1) == 1  # TODO: extend to multi-head
            aws_prev = torch.cat([aws.new_zeros(aws.size())[:, :, -1:], aws.clone()[:, :, :-1]], dim=2)
            aws_mat = aws_prev.unsqueeze(3) * aws.unsqueeze(4)  # `[B, H, L, T, T]`
            delay_mat = aws.new_ones(xmax, xmax).float()
            delay_mat = torch.tril(delay_mat, diagonal=-1, out=delay_mat)
            delay_mat = torch.cumsum(delay_mat, dim=-2).unsqueeze(0)
            delay_mat = delay_mat.unsqueeze(1).unsqueeze(2).expand_as(aws_mat)
            loss_latency = torch.pow((aws_mat * delay_mat).sum(-1), 2).sum(-1)
            loss_latency = torch.mean(loss_latency.squeeze(1))
        elif ctc_trigger_points is not None or ('ctc_sync' not in self.latency_metric and forced_trigger_points is not None):
            if 'ctc_sync' in self.latency_metric:
                trigger_points = ctc_trigger_points
            else:
                trigger_points = forced_trigger_points

            # CTC-synchronous training/Minimum latency training/Delay constrained training
            js = torch.arange(xmax, dtype=torch.float, device=eouts.device).expand_as(aws)
            exp_trigger_points = (js * aws).sum(3)  # `[B, H, L]`
            trigger_points = trigger_points.float().unsqueeze(1)  # `[B, 1, L]`
            loss_latency = torch.abs(exp_trigger_points - trigger_points)  # `[B, H, L]`
            # NOTE: trigger_points are padded with 0
            loss_latency = loss_latency.sum() / ylens.sum()

        # Knowledge distillation
        if teacher_logits is not None:
            kl_loss = distillation(logits, teacher_logits, ylens, temperature=5.0)
            loss = loss * (1 - self.distil_weight) + kl_loss * self.distil_weight

        # Compute token-level accuracy in teacher-forcing
        acc = compute_accuracy(logits, ys_out, self.pad)

        return loss, acc, ppl, loss_quantity, loss_latency

    def decode_step(self, eouts, dstates, cv, y_emb, mask, aw, lmout,
                    mode='hard', trigger_points=None, cache=True, streaming=False):
        dstates = self.recurrency(torch.cat([y_emb, cv], dim=-1), dstates['dstate'])
        cv, aw, attn_state = self.score(eouts, eouts, dstates['dout_score'], mask, aw,
                                        cache=cache, mode=mode, trigger_points=trigger_points,
                                        streaming=streaming)
        attn_v = self.generate(cv, dstates['dout_gen'], lmout)
        return dstates, cv, aw, attn_state, attn_v

    def zero_state(self, bs):
        """Initialize decoder state.

        Args:
            bs (int): batch size
        Returns:
            dstates (dict):
                dout (FloatTensor): `[B, 1, dec_n_units]`
                dstate (tuple): A tuple of (hxs, cxs)
                    hxs (FloatTensor): `[n_layers, B, dec_n_units]`
                    cxs (FloatTensor): `[n_layers, B, dec_n_units]`

        """
        dstates = {'dstate': None}
        w = next(self.parameters())
        hxs = w.new_zeros(self.n_layers, bs, self.dec_n_units)
        cxs = w.new_zeros(self.n_layers, bs, self.dec_n_units) if self.rnn_type == 'lstm' else None
        dstates['dstate'] = (hxs, cxs)
        return dstates

    def recurrency(self, inputs, dstate):
        """Recurrency function.

        Args:
            inputs (FloatTensor): `[B, 1, emb_dim + enc_n_units]`
            dstate (tuple): A tuple of (hxs, cxs)
        Returns:
            new_dstates (dict):
                dout_score (FloatTensor): `[B, 1, dec_n_units]`
                dout_gen (FloatTensor): `[B, 1, dec_n_units]`
                dstate (tuple): A tuple of (hxs, cxs)
                    hxs (FloatTensor): `[n_layers, B, dec_n_units]`
                    cxs (FloatTensor): `[n_layers, B, dec_n_units]`

        """
        hxs, cxs = dstate
        dout = inputs.squeeze(1)

        new_dstates = {'dout_score': None,  # for attention scoring
                       'dout_gen': None,  # for token generation
                       'dstate': None}

        new_hxs, new_cxs = [], []
        for lth in range(self.n_layers):
            if self.rnn_type == 'lstm':
                h, c = self.rnn[lth](dout, (hxs[lth], cxs[lth]))
                new_cxs.append(c)
            elif self.rnn_type == 'gru':
                h = self.rnn[lth](dout, hxs[lth])
            new_hxs.append(h)
            dout = self.dropout(h)
            if self.proj is not None:
                dout = torch.relu(self.proj[lth](dout))
            # use output in the FIRST layer for attention scoring
            if lth == 0:
                new_dstates['dout_score'] = dout.unsqueeze(1)
        new_hxs = torch.stack(new_hxs, dim=0)
        if self.rnn_type == 'lstm':
            new_cxs = torch.stack(new_cxs, dim=0)

        # use output in the the LAST layer for token generation
        new_dstates['dout_gen'] = dout.unsqueeze(1)
        new_dstates['dstate'] = (new_hxs, new_cxs)
        return new_dstates

    def generate(self, cv, dout, lmout):
        """Generate function.

        Args:
            cv (FloatTensor): `[B, 1, enc_n_units]`
            dout (FloatTensor): `[B, 1, dec_n_units]`
            lmout (FloatTensor): `[B, 1, lm_n_units]`
        Returns:
            attn_v (FloatTensor): `[B, 1, vocab]`

        """
        gated_lmout = None
        if self.lm is not None:
            # LM fusion
            dec_feat = self.linear_dec_feat(torch.cat([dout, cv], dim=-1))

            if self.lm_fusion in ['cold', 'deep']:
                lmout = self.linear_lm_feat(lmout)
                gate = torch.sigmoid(self.linear_lm_gate(torch.cat([dec_feat, lmout], dim=-1)))
                gated_lmout = gate * lmout
            elif self.lm_fusion == 'cold_prob':
                lmout = self.linear_lm_feat(self.lm.output(lmout))
                gate = torch.sigmoid(self.linear_lm_gate(torch.cat([dec_feat, lmout], dim=-1)))
                gated_lmout = gate * lmout

            out = self.output_bn(torch.cat([dec_feat, gated_lmout], dim=-1))
        else:
            out = self.output_bn(torch.cat([dout, cv], dim=-1))
        attn_v = torch.tanh(out)
        return attn_v

    def greedy(self, eouts, elens, max_len_ratio, idx2token,
               exclude_eos=False, refs_id=None, utt_ids=None, speakers=None,
               trigger_points=None):
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
            trigger_points (IntTensor): `[B, T]`
        Returns:
            hyps (List): length `[B]`, each of which contains arrays of size `[L]`
            aws (List): length `[B]`, each of which contains arrays of size `[H, L, T]`

        """
        bs, xmax = eouts.size()[:2]

        # Initialization
        dstates = self.zero_state(bs)
        if self.discourse_aware and not self._new_session:
            dstates = {'dstate': (self.dstate_prev['hxs'], self.dstate_prev['cxs'])}
        self.dstate_prev = {'hxs': [None] * bs, 'cxs': [None] * bs}
        self._new_session = False
        cv = eouts.new_zeros(bs, 1, self.enc_n_units)
        self.score.reset()
        aw = None
        lmout, lmstate = None, None
        y = eouts.new_zeros((bs, 1), dtype=torch.int64).fill_(refs_id[0][0] if self.replace_sos else self.eos)

        # Create the attention mask
        src_mask = make_pad_mask(elens.to(eouts.device)).unsqueeze(1)  # `[B, 1, T]`

        if self.attn_type == 'triggered_attention':
            assert trigger_points is not None

        hyps_batch, aws_batch = [], []
        ylens = torch.zeros(bs).int()
        eos_flags = [False] * bs
        ymax = math.ceil(xmax * max_len_ratio)
        for i in range(ymax):
            # Update LM states for LM fusion
            if self.lm is not None:
                lmout, lmstate, _ = self.lm.predict(y, lmstate)

            # Recurrency -> Score -> Generate
            y_emb = self.dropout_emb(self.embed(y))
            dstates, cv, aw, attn_state, attn_v = self.decode_step(
                eouts, dstates, cv, y_emb, src_mask, aw, lmout,
                trigger_points=trigger_points[:, i:i + 1] if trigger_points is not None else None)
            aws_batch += [aw]  # `[B, H, 1, T]`
            if self.attn_type in ['gmm', 'sagmm']:
                aw = attn_state['myu']

            # Pick up 1-best
            y = self.output(attn_v).argmax(-1)
            hyps_batch += [y]

            # Count lengths of hypotheses
            for b in range(bs):
                if not eos_flags[b]:
                    if y[b].item() == self.eos:
                        eos_flags[b] = True
                        if self.discourse_aware:
                            self.dstate_prev['hxs'][b] = dstates['dstate'][0][:, b:b + 1]
                            if self.rnn_type == 'lstm':
                                self.dstate_prev['cxs'][b] = dstates['dstate'][1][:, b:b + 1]
                    ylens[b] += 1  # include <eos>

            # Break if <eos> is outputed in all mini-batch
            if sum(eos_flags) == bs:
                break
            if i == ymax - 1:
                break

        # ASR state carry over
        if self.discourse_aware:
            if bs > 1:
                self.dstate_prev['hxs'] = torch.cat(self.dstate_prev['hxs'], dim=1)
                if self.rnn_type == 'lstm':
                    self.dstate_prev['cxs'] = torch.cat(self.dstate_prev['cxs'], dim=1)
            else:
                self.dstate_prev['hxs'] = self.dstate_prev['hxs']
                if self.rnn_type == 'lstm':
                    self.dstate_prev['cxs'] = self.dstate_prev['cxs']

        # LM state carry over
        self.lmstate_final = lmstate

        # Concatenate in L dimension
        hyps_batch = tensor2np(torch.cat(hyps_batch, dim=1))
        aws_batch = tensor2np(torch.cat(aws_batch, dim=2))  # `[B, H, L, T]`

        # Truncate by the first <eos> (<sos> in case of the backward decoder)
        if self.bwd:
            # Reverse the order
            hyps = [hyps_batch[b, :ylens[b]][::-1] for b in range(bs)]
            aws = [aws_batch[b, :, :ylens[b]][::-1] for b in range(bs)]
        else:
            hyps = [hyps_batch[b, :ylens[b]] for b in range(bs)]
            aws = [aws_batch[b, :, :ylens[b]] for b in range(bs)]

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
                logger.debug('=' * 200)
                # NOTE: do not show with logger.info here

        return hyps, aws

    def initialize_beam(self, hyp, dstates, cv, lmstate, ctc_state,
                        ys=None, ensmbl_decs=[]):
        # Ensemble initialization
        ensmbl_dstate, ensmbl_cv = [], []
        for dec in ensmbl_decs:
            dec.score.reset()
            ensmbl_dstate += [dec.zero_state(1)]
            ensmbl_cv += [cv.new_zeros(1, 1, dec.enc_n_units)]

        hyps = [{'hyp': hyp,
                 'score': 0.,
                 'score_att': 0.,
                 'score_ctc': 0.,
                 'score_lm': 0.,
                 'dstates': dstates,
                 'cv': cv,
                 'aws': [None],
                 'myu': None,
                 'lmstate': lmstate,
                 'ys': ys,  # for TransformerLM
                 'ensmbl_dstate': ensmbl_dstate,
                 'ensmbl_cv': ensmbl_cv,
                 'ensmbl_aws':[[None]] * len(ensmbl_dstate),
                 'ctc_state': ctc_state,
                 'quantity_rate': 1.,
                 'streamable': True,
                 'streaming_failed_point': 1000,
                 'boundary': [],
                 'no_boundary': False}]
        return hyps

    def beam_search(self, eouts, elens, params, idx2token=None,
                    lm=None, lm_second=None, lm_second_bwd=None, ctc_log_probs=None,
                    nbest=1, exclude_eos=False,
                    refs_id=None, utt_ids=None, speakers=None,
                    ensmbl_eouts=[], ensmbl_elens=[], ensmbl_decs=[],
                    cache_states=True):
        """Beam search decoding.

        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
            params (dict): hyperparameters for decoding
            idx2token (): converter from index to token
            lm (torch.nn.module): firsh path LM
            lm_second (torch.nn.module): second path LM
            lm_second_bwd (torch.nn.module): secoding path backward LM
            ctc_log_probs (FloatTensor): `[B, T, vocab]`
            nbest (int): number of N-best list
            exclude_eos (bool): exclude <eos> from hypothesis
            refs_id (List): reference list
            utt_ids (List): utterance id list
            speakers (List): speaker list
            ensmbl_eouts (List[FloatTensor]): encoder outputs for ensemble models
            ensmbl_elens (List[IntTensor]) encoder outputs for ensemble models
            ensmbl_decs (List[torch.nn.Module): decoders for ensemble models
            cache_states (bool): cache TransformerLM/TransformerXL states for fast decoding
        Returns:
            nbest_hyps_idx (List): length `[B]`, each of which contains list of N hypotheses
            aws (List): length `[B]`, each of which contains a list of arrays of size `[H, L, T]`
                for N hypotheses
            scores (List):

        """
        bs, xmax, _ = eouts.size()

        beam_width = params['recog_beam_width']
        assert 1 <= nbest <= beam_width
        ctc_weight = params['recog_ctc_weight']
        max_len_ratio = params['recog_max_len_ratio']
        min_len_ratio = params['recog_min_len_ratio']
        lp_weight = params['recog_length_penalty']
        cp_weight = params['recog_coverage_penalty']
        cp_threshold = params['recog_coverage_threshold']
        length_norm = params['recog_length_norm']
        lm_weight = params['recog_lm_weight']
        lm_weight_second = params['recog_lm_second_weight']
        lm_weight_second_bwd = params['recog_lm_bwd_weight']
        gnmt_decoding = params['recog_gnmt_decoding']
        eos_threshold = params['recog_eos_threshold']
        asr_state_CO = params['recog_asr_state_carry_over']
        lm_state_CO = params['recog_lm_state_carry_over']
        softmax_smoothing = params['recog_softmax_smoothing']

        helper = BeamSearch(beam_width, self.eos, ctc_weight, eouts.device)
        lm = helper.verify_lm_eval_mode(lm, lm_weight)
        lm_second = helper.verify_lm_eval_mode(lm_second, lm_weight_second)
        lm_second_bwd = helper.verify_lm_eval_mode(lm_second_bwd, lm_weight_second_bwd)
        trfm_lm = isinstance(lm, TransformerLM) or isinstance(lm, TransformerXL)

        if ctc_log_probs is not None:
            assert ctc_weight > 0
            ctc_log_probs = tensor2np(ctc_log_probs)

        nbest_hyps_idx, aws, scores = [], [], []
        eos_flags = []
        for b in range(bs):
            # Initialization per utterance
            self.score.reset()
            cv = eouts.new_zeros(1, 1, self.enc_n_units)
            dstates = self.zero_state(1)
            lmstate = None
            ctc_state = None
            ys = eouts.new_zeros((1, 1), dtype=torch.int64).fill_(self.eos)  # for Transformer(XL) LM

            # For joint CTC-Attention decoding
            ctc_prefix_scorer = None
            if ctc_log_probs is not None:
                if self.bwd:
                    ctc_prefix_scorer = CTCPrefixScore(ctc_log_probs[b][::-1], self.blank, self.eos)
                else:
                    ctc_prefix_scorer = CTCPrefixScore(ctc_log_probs[b], self.blank, self.eos)
                ctc_state = ctc_prefix_scorer.initial_state()

            if speakers is not None:
                if speakers[b] == self.prev_spk:
                    if asr_state_CO:
                        dstates = self.dstates_final
                    if lm_state_CO:
                        if isinstance(lm, RNNLM):
                            lmstate = self.lmstate_final
                        elif isinstance(lm, TransformerLM):
                            ys_prev = self.lmstate_final
                            # Re-encode past tokens here
                            _, lmstate, _ = lm.predict(ys_prev)
                            ys = torch.cat([ys_prev, ys], dim=1)
                        # elif isinstance(lm, TransformerXL):
                        #     ys_prev = self.lmstate_final
                        #     # Re-encode past tokens here
                        #     _, lmstate, _ = lm.predict(ys_prev, mems=self.lmmemory)
                        #     ys = torch.cat([ys_prev, ys], dim=1)
                else:
                    self.dstates_final = None  # reset
                    self.lmstate_final = None  # reset
                    self.lmmemory = None  # reset
                self.prev_spk = speakers[b]

            end_hyps = []
            hyps = self.initialize_beam([self.eos], dstates, cv, lmstate, ctc_state,
                                        ys, ensmbl_decs)
            streamable_global = True
            ymax = math.ceil(elens[b] * max_len_ratio)
            for i in range(ymax):
                # batchfy all hypotheses for batch decoding
                y = eouts.new_zeros((len(hyps), 1), dtype=torch.int64)
                for j, beam in enumerate(hyps):
                    if self.replace_sos and i == 0:
                        prev_idx = refs_id[0][0]
                    else:
                        prev_idx = beam['hyp'][-1]
                    y[j, 0] = prev_idx
                cv = torch.cat([beam['cv'] for beam in hyps], dim=0)
                eouts_b_i = eouts[b:b + 1, :elens[b]].repeat([cv.size(0), 1, 1])
                if self.attn_type in ['gmm', 'sagmm']:
                    aw = torch.cat([beam['myu'] for beam in hyps], dim=0) if i > 0 else None
                else:
                    aw = torch.cat([beam['aws'][-1] for beam in hyps], dim=0) if i > 0 else None
                hxs = torch.cat([beam['dstates']['dstate'][0] for beam in hyps], dim=1)
                if self.rnn_type == 'lstm':
                    cxs = torch.cat([beam['dstates']['dstate'][1] for beam in hyps], dim=1)
                dstates = {'dstate': (hxs, cxs)}

                # Update LM states for LM fusion
                lmout, lmstate, scores_lm = None, None, None
                if lm is not None or self.lm is not None:
                    if trfm_lm:
                        y_lm = eouts.new_zeros((len(hyps), beam['ys'].size(1)), dtype=torch.int64)
                        for j, cand in enumerate(hyps):
                            y_lm[j, :] = cand['ys']
                    else:
                        y_lm = y

                    if i > 0 or (i == 0 and trfm_lm and lm_state_CO and self.lmstate_final is not None):
                        if isinstance(lm, RNNLM):
                            lmstate = {'hxs': torch.cat([beam['lmstate']['hxs'] for beam in hyps], dim=1),
                                       'cxs': torch.cat([beam['lmstate']['cxs'] for beam in hyps], dim=1)}
                        elif trfm_lm:
                            if isinstance(lm, TransformerLM):
                                lmstate = [torch.cat([beam['lmstate'][lth] for beam in hyps], dim=0)
                                           for lth in range(lm.n_layers)]
                            elif i > 0:
                                lmstate = [torch.cat([beam['lmstate'][lth] for beam in hyps], dim=0)
                                           for lth in range(lm.n_layers)]

                    if self.lm is not None:  # cold/deep fusion
                        lmout, lmstate, scores_lm = self.lm.predict(y_lm, lmstate)
                    elif lm is not None:  # shallow fusion
                        lmout, lmstate, scores_lm = lm.predict(y_lm, lmstate,
                                                               mems=self.lmmemory,
                                                               cache=lmstate if cache_states else None)

                # for the main model
                y_emb = self.dropout_emb(self.embed(y))
                dstates, cv, aw, attn_state, attn_v = self.decode_step(
                    eouts_b_i, dstates, cv, y_emb, None, aw, lmout)
                probs = torch.softmax(self.output(attn_v).squeeze(1) * softmax_smoothing, dim=1)

                # for the ensemble
                ensmbl_dstate, ensmbl_cv, ensmbl_aws = [], [], []
                for i_e, dec in enumerate(ensmbl_decs):
                    cv_e = torch.cat([beam['ensmbl_cv'][i_e] for beam in hyps], dim=0)
                    aw_e = torch.cat([beam['ensmbl_aws'][i_e][-1] for beam in hyps], dim=0) if i > 0 else None
                    hxs_e = torch.cat([beam['ensmbl_dstate'][i_e]['dstate'][0] for beam in hyps], dim=1)
                    if self.rnn_type == 'lstm':
                        cxs_e = torch.cat([beam['ensmbl_dstate'][i_e]['dstate'][1] for beam in hyps], dim=1)
                    dstates_e = {'dstate': (hxs_e, cxs_e)}

                    dstates_e, cv_e, aw_e, attn_state_e, attn_v_e = dec.decode_step(
                        ensmbl_eouts[i_e][b:b + 1, :ensmbl_elens[i_e][b]].repeat([cv_e.size(0), 1, 1]),
                        dstates_e, cv_e, dec.dropout_emb(dec.embed(y)), None, aw_e, lmout)

                    ensmbl_dstate += [{'dstate': (dstates_e['dstate'][0][:, j:j + 1],
                                                  dstates_e['dstate'][1][:, j:j + 1])}]
                    ensmbl_cv += [cv_e[j:j + 1]]
                    ensmbl_aws += [beam['ensmbl_aws'][i_e] + [aw_e[j:j + 1]]]
                    probs += torch.softmax(dec.output(attn_v_e).squeeze(1), dim=1)

                # Ensemble
                scores_att = torch.log(probs / (len(ensmbl_decs) + 1))

                new_hyps = []
                for j, beam in enumerate(hyps):
                    # Attention scores
                    total_scores_att = beam['score_att'] + scores_att[j:j + 1]
                    total_scores = total_scores_att * (1 - ctc_weight)
                    total_scores_topk, topk_ids = torch.topk(
                        total_scores, k=beam_width, dim=1, largest=True, sorted=True)

                    # Add LM score <after> top-K selection
                    if lm is not None:
                        total_scores_lm = beam['score_lm'] + scores_lm[j, -1, topk_ids[0]]
                        total_scores_topk += total_scores_lm * lm_weight
                    else:
                        total_scores_lm = eouts.new_zeros(beam_width)

                    # Add length penalty
                    if lp_weight > 0:
                        if gnmt_decoding:
                            lp = math.pow(6 + len(beam['hyp'][1:]), lp_weight) / math.pow(6, lp_weight)
                            total_scores_topk /= lp
                        else:
                            total_scores_topk += (len(beam['hyp'][1:]) + 1) * lp_weight

                    # Add coverage penalty
                    if cp_weight > 0:
                        aw_mat = torch.cat(beam['aws'][1:] + [aw[j:j + 1]], dim=2)  # `[B, H, L, T]`
                        aw_mat = aw_mat[:, 0, :, :]  # `[B, L, T]`
                        if gnmt_decoding:
                            aw_mat = torch.log(aw_mat.sum(-1))
                            cp = torch.where(aw_mat < 0, aw_mat, aw_mat.new_zeros(aw_mat.size())).sum()
                            # TODO(hirofumi): mask by elens[b]
                            total_scores_topk += cp * cp_weight
                        else:
                            # Recompute coverage penalty at each step
                            if cp_threshold == 0:
                                cp = aw_mat.sum() / self.score.n_heads
                            else:
                                cp = torch.where(aw_mat > cp_threshold, aw_mat,
                                                 aw_mat.new_zeros(aw_mat.size())).sum() / self.score.n_heads
                            total_scores_topk += cp * cp_weight
                    else:
                        cp = 0.

                    # Add CTC score
                    new_ctc_states, total_scores_ctc, total_scores_topk = helper.add_ctc_score(
                        beam['hyp'], topk_ids, beam['ctc_state'],
                        total_scores_topk, ctc_prefix_scorer)

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

                        streaming_failed_point = beam['streaming_failed_point']
                        quantity_rate = 1.
                        if self.attn_type == 'mocha':
                            n_heads_total = 1
                            n_quantity_k = aw[j:j + 1, :, 0].int().sum().item()
                            quantity_diff = n_heads_total - n_quantity_k

                            if quantity_diff != 0:
                                if idx == self.eos:
                                    quantity_rate = 1
                                    # NOTE: do not count <eos> for streamability
                                else:
                                    streamable_global = False
                                    quantity_rate = n_quantity_k / n_heads_total

                            if beam['streamable'] and not streamable_global:
                                streaming_failed_point = i

                        new_lmstate = None
                        if lmstate is not None:
                            if isinstance(lm, RNNLM) or isinstance(self.lm, RNNLM):
                                new_lmstate = {'hxs': lmstate['hxs'][:, j:j + 1],
                                               'cxs': lmstate['cxs'][:, j:j + 1]}
                            elif trfm_lm:
                                new_lmstate = [lmstate_l[j:j + 1] for lmstate_l in lmstate]
                            else:
                                raise ValueError

                        ys = torch.cat([beam['ys'], eouts.new_zeros((1, 1), dtype=torch.int64).fill_(idx)], dim=-1)

                        new_hyps.append(
                            {'hyp': beam['hyp'] + [idx],
                             'ys': ys,
                             'score': total_score,
                             'score_att': total_scores_att[0, idx].item(),
                             'score_cp': cp,
                             'score_ctc': total_scores_ctc[k].item(),
                             'score_lm': total_scores_lm[k].item(),
                             'dstates': {'dstate': (dstates['dstate'][0][:, j:j + 1],
                                                    dstates['dstate'][1][:, j:j + 1])},
                             'cv': cv[j:j + 1],
                             'aws': beam['aws'] + [aw[j:j + 1]],
                             'myu': attn_state['myu'][j:j + 1] if self.attn_type in ['gmm', 'sagmm'] else None,
                             'lmstate': new_lmstate,
                             'ctc_state': new_ctc_states[k] if ctc_prefix_scorer is not None else None,
                             'ensmbl_dstate': ensmbl_dstate,
                             'ensmbl_cv': ensmbl_cv,
                             'ensmbl_aws': ensmbl_aws,
                             'streamable': streamable_global,
                             'streaming_failed_point': streaming_failed_point,
                             'quantity_rate': quantity_rate})

                # Local pruning
                new_hyps_sorted = sorted(new_hyps, key=lambda x: x['score'], reverse=True)[:beam_width]

                # Remove complete hypotheses
                new_hyps, end_hyps, is_finish = helper.remove_complete_hyp(
                    new_hyps_sorted, end_hyps)
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
                    logger.info('log prob (hyp): %.7f' % end_hyps[k]['score'])
                    logger.info('log prob (hyp, att): %.7f' % (end_hyps[k]['score_att'] * (1 - ctc_weight)))
                    logger.info('log prob (hyp, cp): %.7f' % (end_hyps[k]['score_cp'] * cp_weight))
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
            if length_norm:
                scores += [[end_hyps[n]['score_att'] / len(end_hyps[n]['hyp'][1:]) for n in range(nbest)]]
            else:
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
        self.dstates_final = end_hyps[0]['dstates']
        if isinstance(lm, RNNLM):
            self.lmstate_final = end_hyps[0]['lmstate']
        elif trfm_lm:
            if isinstance(lm, TransformerXL):
                self.lmmemory = lm.update_memory(self.lmmemory, end_hyps[0]['lmstate'])
                logger.info('Memory: %d' % self.lmmemory[0].size(1))
            else:
                ys = end_hyps[0]['ys']
                # Exclude the last state corresponding to <eos>
                if ys[0, -1].item() == self.eos:
                    ys = ys[:, :-1]
                ys = ys[:, -lm.mem_len:]  # Truncate by BPTT length
            self.lmstate_final = ys

        return nbest_hyps_idx, aws, scores

    def beam_search_block_sync(self, eouts, params, idx2token, hyps,
                               lm=None, ctc_log_probs=None,
                               state_carry_over=False, emb_cache=True):
        assert eouts.size(0) == 1
        assert self.attn_type == 'mocha'

        beam_width = params['recog_beam_width']
        ctc_weight = params['recog_ctc_weight']
        max_len_ratio = params['recog_max_len_ratio']
        lp_weight = params['recog_length_penalty']
        length_norm = params['recog_length_norm']
        lm_weight = params['recog_lm_weight']
        eos_threshold = params['recog_eos_threshold']

        helper = BeamSearch(beam_width, self.eos, ctc_weight, eouts.device)
        lm = helper.verify_lm_eval_mode(lm, lm_weight)

        # pre-compute embeddings
        if emb_cache and self.embed_cache is None:
            indices = torch.arange(0, self.vocab, 1, dtype=torch.int64).to(eouts.device)
            self.embed_cache = self.dropout_emb(self.embed(indices))  # `[1, vocab, emb_dim]`

        end_hyps = []
        if hyps is None:
            # Initialization per utterance
            self.score.reset()
            cv = eouts.new_zeros(1, 1, self.enc_n_units)
            dstates = self.zero_state(1)
            lmstate = None
            ctc_state = None

            # For joint CTC-Attention decoding
            self.ctc_prefix_scorer = None
            if ctc_log_probs is not None:
                assert ctc_weight > 0
                ctc_log_probs = tensor2np(ctc_log_probs)
                if hyps is None:
                    # first block
                    self.ctc_prefix_scorer = CTCPrefixScore(ctc_log_probs[0], self.blank, self.eos)
                    ctc_state = self.ctc_prefix_scorer.initial_state()
                else:
                    self.ctc_prefix_scorer.register_new_chunk(ctc_log_probs[0])
            # TODO: add truncated version

            if state_carry_over:
                dstates = self.dstates_final
                if isinstance(lm, RNNLM):
                    lmstate = self.lmstate_final

            self.n_frames = 0
            self.chunk_size = eouts.size(1)
            hyps = self.initialize_beam([self.eos], dstates, cv, lmstate, ctc_state)
        else:
            for h in hyps:
                h['no_boundary'] = False

        ymax = math.ceil(eouts.size(1) * max_len_ratio)
        for i in range(ymax):
            # finish if no additional token boundary is found in the current block for all candidates
            if len(hyps) == 0:
                break
            if i > 0 and sum([cand['no_boundary'] for cand in hyps]) == len(hyps):
                break

            # ignore hypotheses with no boundary from batched hypotheses
            new_hyps = []
            hyps_filtered = []
            for j, beam in enumerate(hyps):
                # no token boundary found in the current block
                if beam['no_boundary']:
                    new_hyps.append(beam.copy())
                else:
                    hyps_filtered.append(beam.copy())
            if len(hyps_filtered) == 0:
                break
            hyps = hyps_filtered[:]

            # batchfy all hypotheses for batch decoding
            y = eouts.new_zeros((len(hyps), 1), dtype=torch.int64)
            for j, beam in enumerate(hyps):
                y[j, 0] = beam['hyp'][-1]
            cv = torch.cat([beam['cv'] for beam in hyps], dim=0)
            aw = torch.cat([beam['aws'][-1] for beam in hyps], dim=0) if i > 0 else None
            hxs = torch.cat([beam['dstates']['dstate'][0] for beam in hyps], dim=1)
            if self.rnn_type == 'lstm':
                cxs = torch.cat([beam['dstates']['dstate'][1] for beam in hyps], dim=1)
            dstates = {'dstate': (hxs, cxs)}

            # Update LM states for LM fusion
            lmout, lmstate, scores_lm = helper.update_rnnlm_state_batch(
                self.lm if self.lm is not None else lm, hyps, y, emb_cache=emb_cache)

            if self.embed_cache is not None:
                y_emb = self.embed_cache[y]
            else:
                y_emb = self.dropout_emb(self.embed(y))
            dstates, cv, aw, _, attn_v = self.decode_step(
                eouts[0:1], dstates, cv, y_emb, None, aw, lmout, streaming=True)
            scores_att = torch.log_softmax(self.output(attn_v).squeeze(1), dim=1)
            # NOTE: aw: `[B, H, 1, T_block]`

            for j, beam in enumerate(hyps):
                # no token boundary found in the current block for j-th utterance
                no_boundary = aw[j].sum().item() == 0
                if no_boundary:
                    beam['aws'][-1] = eouts.new_zeros(eouts.size(0), 1, 1, eouts.size(1))
                    # NOTE: the case where the first token in the current block is <eos>
                    beam['no_boundary'] = True
                    new_hyps.append(beam.copy())  # this is important to remove repeated hyps

                # Attention scores
                total_scores_att = beam['score_att'] + scores_att[j:j + 1]
                total_scores = total_scores_att * (1 - ctc_weight)

                # Add LM score <after> top-K selection
                total_scores_topk, topk_ids = torch.topk(
                    total_scores, k=beam_width, dim=1, largest=True, sorted=True)
                if lm is not None:
                    total_scores_lm = beam['score_lm'] + scores_lm[j, -1, topk_ids[0]]
                    total_scores_topk += total_scores_lm * lm_weight
                else:
                    total_scores_lm = eouts.new_zeros(beam_width)

                # Add length penalty
                total_scores_topk += (len(beam['hyp'][1:]) + 1) * lp_weight

                cp = self.n_frames
                if not no_boundary:
                    boundary_list_j = np.where(tensor2np(aw[j].sum(1).sum(0)) != 0)[0]
                    cp += int(boundary_list_j[0])
                    if len(beam['boundary']) > 0:
                        assert cp >= beam['boundary'][-1], (cp, beam['boundary'])

                # Add CTC score
                new_ctc_states, total_scores_ctc, total_scores_topk = helper.add_ctc_score(
                    beam['hyp'], topk_ids, beam['ctc_state'],
                    total_scores_topk, self.ctc_prefix_scorer, new_chunk=(i == 0))

                for k in range(beam_width):
                    idx = topk_ids[0, k].item()
                    if no_boundary and idx != self.eos:
                        continue
                    length_norm_factor = len(beam['hyp'][1:]) + 1 if length_norm else 1
                    total_score = total_scores_topk[0, k].item() / length_norm_factor

                    if idx == self.eos:
                        # EOS threshold
                        max_score_no_eos = scores_att[j, :idx].max(0)[0].item()
                        max_score_no_eos = max(max_score_no_eos, scores_att[j, idx + 1:].max(0)[0].item())
                        if scores_att[j, idx].item() <= eos_threshold * max_score_no_eos:
                            continue

                    new_hyps.append(
                        {'hyp': beam['hyp'] + [idx],
                         'score': total_score,
                         'score_att': total_scores_att[0, idx].item(),
                         'score_ctc': total_scores_ctc[k].item(),
                         'score_lm': total_scores_lm[k].item(),
                         'dstates': {'dstate': (dstates['dstate'][0][:, j:j + 1], dstates['dstate'][1][:, j:j + 1])},
                         'cv': cv[j:j + 1],
                         'aws': beam['aws'] + [aw[j:j + 1]],
                         'lmstate': {'hxs': lmstate['hxs'][:, j:j + 1],
                                     'cxs': lmstate['cxs'][:, j:j + 1]} if lmstate is not None else None,
                         'ctc_state': new_ctc_states[k] if self.ctc_prefix_scorer is not None else None,
                         'boundary': beam['boundary'] + [cp] if not no_boundary else beam['boundary'],
                         'no_boundary': no_boundary})

            # Local pruning
            new_hyps_sorted = sorted(new_hyps, key=lambda x: x['score'], reverse=True)[:beam_width]

            # Remove complete hypotheses
            new_hyps, end_hyps, is_finish = helper.remove_complete_hyp(new_hyps_sorted, end_hyps)
            hyps = new_hyps[:]
            if is_finish:
                break

        # Sort by score
        if len(end_hyps) > 0:
            end_hyps = sorted(end_hyps, key=lambda x: x['score'], reverse=True)

        merged_hyps = sorted(end_hyps + hyps, key=lambda x: x['score'], reverse=True)[:beam_width]
        if idx2token is not None:
            logger.info('=' * 200)
            for k in range(len(merged_hyps)):
                logger.info('Hyp: %s' % idx2token(merged_hyps[k]['hyp'][1:]))
                if len(merged_hyps[k]['hyp']) > 1:
                    logger.info('num tokens (hyp): %d' % len(merged_hyps[k]['hyp'][1:]))
                if len(merged_hyps[k]['boundary']) > 0:
                    logger.info('boundary: %s' % ' '.join(list(map(str, merged_hyps[k]['boundary']))))
                logger.info('no boundary: %s' % merged_hyps[k]['no_boundary'])
                logger.info('log prob (hyp): %.7f' % merged_hyps[k]['score'])
                logger.info('log prob (hyp, att): %.7f' % (merged_hyps[k]['score_att'] * (1 - ctc_weight)))
                if self.ctc_prefix_scorer is not None:
                    logger.info('log prob (hyp, ctc): %.7f' % (merged_hyps[k]['score_ctc'] * ctc_weight))
                if lm is not None:
                    logger.info('log prob (hyp, first-path lm): %.7f' % (merged_hyps[k]['score_lm'] * lm_weight))
                logger.info('-' * 50)

        aws = None

        # Store ASR/LM state
        if len(end_hyps) > 0:
            self.dstates_final = end_hyps[0]['dstates']
            self.lmstate_final = end_hyps[0]['lmstate']

        self.n_frames += eouts.size(1)
        self.score.reset_block()

        return end_hyps, hyps, aws
