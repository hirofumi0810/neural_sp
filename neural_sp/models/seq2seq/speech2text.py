# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Speech to text sequence-to-sequence model."""

import copy
import logging
import math
import numpy as np
import random
import torch
import torch.nn as nn

from neural_sp.bin.train_utils import load_checkpoint
from neural_sp.models.base import ModelBase
from neural_sp.models.lm.rnnlm import RNNLM
from neural_sp.models.seq2seq.decoders.build import build_decoder
from neural_sp.models.seq2seq.decoders.fwd_bwd_attention import fwd_bwd_attention
from neural_sp.models.seq2seq.decoders.las import RNNDecoder
from neural_sp.models.seq2seq.decoders.rnn_transducer import RNNTransducer as RNNT
from neural_sp.models.seq2seq.encoders.build import build_encoder
from neural_sp.models.seq2seq.frontends.frame_stacking import stack_frame
from neural_sp.models.seq2seq.frontends.input_noise import add_input_noise
from neural_sp.models.seq2seq.frontends.sequence_summary import SequenceSummaryNetwork
from neural_sp.models.seq2seq.frontends.spec_augment import SpecAugment
from neural_sp.models.seq2seq.frontends.splicing import splice
from neural_sp.models.seq2seq.frontends.streaming import Streaming
from neural_sp.models.torch_utils import (
    np2tensor,
    tensor2np,
    pad_list
)
from neural_sp.utils import mkdir_join

random.seed(1)

logger = logging.getLogger(__name__)


class Speech2Text(ModelBase):
    """Speech to text sequence-to-sequence model."""

    def __init__(self, args, save_path=None, idx2token=None):

        super(ModelBase, self).__init__()

        self.save_path = save_path

        # for encoder, decoder
        self.input_type = args.input_type
        self.input_dim = args.input_dim
        self.enc_type = args.enc_type
        self.dec_type = args.dec_type

        # for OOV resolution
        self.enc_n_layers = args.enc_n_layers
        self.enc_n_layers_sub1 = args.enc_n_layers_sub1
        self.subsample = [int(s) for s in args.subsample.split('_')]

        # for decoder
        self.vocab = args.vocab
        self.vocab_sub1 = args.vocab_sub1
        self.vocab_sub2 = args.vocab_sub2
        self.blank = 0
        self.unk = 1
        self.eos = 2
        self.pad = 3
        # NOTE: reserved in advance

        # for the sub tasks
        self.main_weight = 1.0 - args.sub1_weight - args.sub2_weight
        self.sub1_weight = args.sub1_weight
        self.sub2_weight = args.sub2_weight
        self.mtl_per_batch = args.mtl_per_batch
        self.task_specific_layer = args.task_specific_layer

        # for CTC
        self.ctc_weight = min(args.ctc_weight, self.main_weight)
        self.ctc_weight_sub1 = min(args.ctc_weight_sub1, self.sub1_weight)
        self.ctc_weight_sub2 = min(args.ctc_weight_sub2, self.sub2_weight)

        # for backward decoder
        self.bwd_weight = min(args.bwd_weight, self.main_weight)
        self.fwd_weight = self.main_weight - self.bwd_weight - self.ctc_weight
        self.fwd_weight_sub1 = self.sub1_weight - self.ctc_weight_sub1
        self.fwd_weight_sub2 = self.sub2_weight - self.ctc_weight_sub2

        # for MBR
        self.mbr_training = args.mbr_training
        self.recog_params = vars(args)
        self.idx2token = idx2token

        # for discourse-aware model
        self.utt_id_prev = None

        # Feature extraction
        self.input_noise_std = args.input_noise_std
        self.n_stacks = args.n_stacks
        self.n_skips = args.n_skips
        self.n_splices = args.n_splices
        self.weight_noise_std = args.weight_noise_std
        self.specaug = None
        if args.n_freq_masks > 0 or args.n_time_masks > 0:
            assert args.n_stacks == 1 and args.n_skips == 1
            assert args.n_splices == 1
            self.specaug = SpecAugment(F=args.freq_width,
                                       T=args.time_width,
                                       n_freq_masks=args.n_freq_masks,
                                       n_time_masks=args.n_time_masks,
                                       p=args.time_width_upper,
                                       adaptive_number_ratio=args.adaptive_number_ratio,
                                       adaptive_size_ratio=args.adaptive_size_ratio,
                                       max_n_time_masks=args.max_n_time_masks)

        # Frontend
        self.ssn = None
        if args.sequence_summary_network:
            assert args.input_type == 'speech'
            self.ssn = SequenceSummaryNetwork(args.input_dim,
                                              n_units=512,
                                              n_layers=3,
                                              bottleneck_dim=100,
                                              dropout=0,
                                              param_init=args.param_init)

        # Encoder
        self.enc = build_encoder(args)
        if args.freeze_encoder:
            for n, p in self.enc.named_parameters():
                p.requires_grad = False
                logger.info('freeze %s' % n)

        special_symbols = {
            'blank': self.blank,
            'unk': self.unk,
            'eos': self.eos,
            'pad': self.pad,
        }

        # main task
        external_lm = None
        directions = []
        if self.fwd_weight > 0 or (self.bwd_weight == 0 and self.ctc_weight > 0):
            directions.append('fwd')
        if self.bwd_weight > 0:
            directions.append('bwd')

        for dir in directions:
            # Load the LM for LM fusion and decoder initialization
            if args.external_lm and dir == 'fwd':
                external_lm = RNNLM(args.lm_conf)
                load_checkpoint(args.external_lm, external_lm)
                # freeze LM parameters
                for n, p in external_lm.named_parameters():
                    p.requires_grad = False

            # Decoder
            dec = build_decoder(args, special_symbols,
                                self.enc.output_dim,
                                args.vocab,
                                self.ctc_weight,
                                args.ctc_fc_list,
                                self.main_weight - self.bwd_weight if dir == 'fwd' else self.bwd_weight,
                                external_lm)
            setattr(self, 'dec_' + dir, dec)

        # sub task
        for sub in ['sub1', 'sub2']:
            if getattr(self, sub + '_weight') > 0:
                dec_sub = build_decoder(args, special_symbols,
                                        self.enc.output_dim,
                                        getattr(self, 'vocab_' + sub),
                                        getattr(self, 'ctc_weight_' + sub),
                                        getattr(args, 'ctc_fc_list_' + sub),
                                        getattr(self, sub + '_weight'),
                                        external_lm)
                setattr(self, 'dec_fwd_' + sub, dec_sub)

        if args.input_type == 'text':
            if args.vocab == args.vocab_sub1:
                # Share the embedding layer between input and output
                self.embed = dec.embed
            else:
                self.embed = nn.Embedding(args.vocab_sub1, args.emb_dim,
                                          padding_idx=self.pad)
                self.dropout_emb = nn.Dropout(p=args.dropout_emb)

        # Initialize bias in forget gate with 1
        # self.init_forget_gate_bias_with_one()

        # Fix all parameters except for the gating parts in deep fusion
        if args.lm_fusion == 'deep' and external_lm is not None:
            for n, p in self.named_parameters():
                if 'output' in n or 'output_bn' in n or 'linear' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

    def trigger_scheduled_sampling(self):
        # main task
        for dir in ['fwd', 'bwd']:
            if hasattr(self, 'dec_' + dir):
                logger.info('Activate scheduled sampling (main)')
                getattr(self, 'dec_' + dir).trigger_scheduled_sampling()

        # sub task
        for sub in ['sub1', 'sub2']:
            if hasattr(self, 'dec_fwd_' + sub):
                logger.info('Activate scheduled sampling (%s)' % sub)
                getattr(self, 'dec_fwd_' + sub).trigger_scheduled_sampling()

    def trigger_quantity_loss(self):
        # main task only now
        if hasattr(self, 'dec_fwd'):
            logger.info('Activate quantity loss')
            getattr(self, 'dec_fwd').trigger_quantity_loss()

    def reset_session(self):
        # main task
        for dir in ['fwd', 'bwd']:
            if hasattr(self, 'dec_' + dir):
                getattr(self, 'dec_' + dir).reset_session()

        # sub task
        for sub in ['sub1', 'sub2']:
            if hasattr(self, 'dec_fwd_' + sub):
                getattr(self, 'dec_fwd_' + sub).reset_session()

    def forward(self, batch, task, is_eval=False, teacher=None, teacher_lm=None):
        """Forward pass.

        Args:
            batch (dict):
                xs (List): input data of size `[T, input_dim]`
                xlens (List): lengths of each element in xs
                ys (List): reference labels in the main task of size `[L]`
                ys_sub1 (List): reference labels in the 1st auxiliary task of size `[L_sub1]`
                ys_sub2 (List): reference labels in the 2nd auxiliary task of size `[L_sub2]`
                utt_ids (List): name of utterances
                speakers (List): name of speakers
            task (str): all/ys*/ys_sub*
            is_eval (bool): evaluation mode
                This should be used in inference model for memory efficiency.
            teacher (Speech2Text): used for knowledge distillation from ASR
            teacher_lm (RNNLM): used for knowledge distillation from LM
        Returns:
            loss (FloatTensor): `[1]`
            observation (dict):

        """
        if is_eval:
            self.eval()
            with torch.no_grad():
                loss, observation = self._forward(batch, task)
        else:
            self.train()
            loss, observation = self._forward(batch, task, teacher, teacher_lm)

        return loss, observation

    def _forward(self, batch, task, teacher=None, teacher_lm=None):
        # Encode input features
        if self.input_type == 'speech':
            if self.mtl_per_batch:
                eout_dict = self.encode(batch['xs'], task)
            else:
                eout_dict = self.encode(batch['xs'], 'all')
        else:
            eout_dict = self.encode(batch['ys_sub1'])

        observation = {}
        loss = torch.zeros((1,), dtype=torch.float32, device=self.device)

        # for the forward decoder in the main task
        if (self.fwd_weight > 0 or (self.bwd_weight == 0 and self.ctc_weight > 0) or self.mbr_training) and task in ['all', 'ys', 'ys.ctc', 'ys.mbr']:
            teacher_logits = None
            if teacher is not None:
                teacher.eval()
                teacher_logits = teacher.generate_logits(batch)
                # TODO(hirofumi): label smoothing, scheduled sampling, dropout?
            elif teacher_lm is not None:
                teacher_lm.eval()
                teacher_logits = self.generate_lm_logits(batch['ys'], lm=teacher_lm)

            loss_fwd, obs_fwd = self.dec_fwd(eout_dict['ys']['xs'], eout_dict['ys']['xlens'],
                                             batch['ys'], task,
                                             teacher_logits, self.recog_params, self.idx2token,
                                             batch['trigger_points'])
            loss += loss_fwd
            if isinstance(self.dec_fwd, RNNT):
                observation['loss.transducer'] = obs_fwd['loss_transducer']
            else:
                observation['acc.att'] = obs_fwd['acc_att']
                observation['ppl.att'] = obs_fwd['ppl_att']
                observation['loss.att'] = obs_fwd['loss_att']
                observation['loss.mbr'] = obs_fwd['loss_mbr']
                if 'loss_quantity' not in obs_fwd.keys():
                    obs_fwd['loss_quantity'] = None
                observation['loss.quantity'] = obs_fwd['loss_quantity']

                if 'loss_latency' not in obs_fwd.keys():
                    obs_fwd['loss_latency'] = None
                observation['loss.latency'] = obs_fwd['loss_latency']

            observation['loss.ctc'] = obs_fwd['loss_ctc']

        # for the backward decoder in the main task
        if self.bwd_weight > 0 and task in ['all', 'ys.bwd']:
            loss_bwd, obs_bwd = self.dec_bwd(eout_dict['ys']['xs'], eout_dict['ys']['xlens'], batch['ys'], task)
            loss += loss_bwd
            observation['loss.att-bwd'] = obs_bwd['loss_att']
            observation['acc.att-bwd'] = obs_bwd['acc_att']
            observation['ppl.att-bwd'] = obs_bwd['ppl_att']
            observation['loss.ctc-bwd'] = obs_bwd['loss_ctc']

        # only fwd for sub tasks
        for sub in ['sub1', 'sub2']:
            # for the forward decoder in the sub tasks
            if (getattr(self, 'fwd_weight_' + sub) > 0 or getattr(self, 'ctc_weight_' + sub) > 0) and task in ['all', 'ys_' + sub, 'ys_' + sub + '.ctc']:
                if len(batch['ys_' + sub]) == 0:
                    continue
                # NOTE: this is for evaluation at the end of every opoch

                loss_sub, obs_fwd_sub = getattr(self, 'dec_fwd_' + sub)(
                    eout_dict['ys_' + sub]['xs'], eout_dict['ys_' + sub]['xlens'],
                    batch['ys_' + sub], task)
                loss += loss_sub
                if isinstance(getattr(self, 'dec_fwd_' + sub), RNNT):
                    observation['loss.transducer-' + sub] = obs_fwd_sub['loss_transducer']
                else:
                    observation['loss.att-' + sub] = obs_fwd_sub['loss_att']
                    observation['acc.att-' + sub] = obs_fwd_sub['acc_att']
                    observation['ppl.att-' + sub] = obs_fwd_sub['ppl_att']
                observation['loss.ctc-' + sub] = obs_fwd_sub['loss_ctc']

        return loss, observation

    def generate_logits(self, batch, temperature=1.0):
        # Encode input features
        if self.input_type == 'speech':
            eout_dict = self.encode(batch['xs'], task='ys')
        else:
            eout_dict = self.encode(batch['ys_sub1'], task='ys')

        # for the forward decoder in the main task
        logits = self.dec_fwd.forward_att(
            eout_dict['ys']['xs'], eout_dict['ys']['xlens'], batch['ys'],
            return_logits=True)
        return logits

    def generate_lm_logits(self, ys, lm, temperature=5.0):
        # Append <sos> and <eos>
        eos = next(lm.parameters()).new_zeros(1).fill_(self.eos).long()
        ys = [np2tensor(np.fromiter(y, dtype=np.int64), self.device)for y in ys]
        ys_in = pad_list([torch.cat([eos, y], dim=0) for y in ys], self.pad)
        lmout, _ = lm.decode(ys_in, None)
        logits = lm.output(lmout)
        return logits

    def encode(self, xs, task='all', streaming=False, lookback=False, lookahead=False):
        """Encode acoustic or text features.

        Args:
            xs (List): length `[B]`, which contains Tensor of size `[T, input_dim]`
            task (str): all/ys*/ys_sub1*/ys_sub2*
            streaming (bool): streaming encoding
            lookback (bool): truncate leftmost frames for lookback in CNN context
            lookahead (bool): truncate rightmost frames for lookahead in CNN context
        Returns:
            eout_dict (dict):

        """
        if self.input_type == 'speech':
            # Frame stacking
            if self.n_stacks > 1:
                xs = [stack_frame(x, self.n_stacks, self.n_skips) for x in xs]

            # Splicing
            if self.n_splices > 1:
                xs = [splice(x, self.n_splices, self.n_stacks) for x in xs]

            xlens = torch.IntTensor([len(x) for x in xs])
            xs = pad_list([np2tensor(x, self.device).float() for x in xs], 0.)

            # SpecAugment
            if self.specaug is not None and self.training:
                xs = self.specaug(xs)

            # Weight noise injection
            if self.weight_noise_std > 0 and self.training:
                self.add_weight_noise(std=self.weight_noise_std)

            # Input Gaussian noise injection
            if self.input_noise_std > 0 and self.training:
                xs = add_input_noise(xs, std=self.input_noise_std)

            # Sequence summary network
            if self.ssn is not None:
                xs = self.ssn(xs, xlens)

        elif self.input_type == 'text':
            xlens = torch.IntTensor([len(x) for x in xs])
            xs = [np2tensor(np.fromiter(x, dtype=np.int64), self.device) for x in xs]
            xs = pad_list(xs, self.pad)
            xs = self.dropout_emb(self.embed(xs))
            # TODO(hirofumi): fix for Transformer

        # encoder
        eout_dict = self.enc(xs, xlens, task.split('.')[0], streaming,
                             lookback, lookahead)

        if self.main_weight < 1 and self.enc_type in ['conv', 'tds', 'gated_conv']:
            for sub in ['sub1', 'sub2']:
                eout_dict['ys_' + sub]['xs'] = eout_dict['ys']['xs'].clone()
                eout_dict['ys_' + sub]['xlens'] = eout_dict['ys']['xlens'][:]

        return eout_dict

    def get_ctc_probs(self, xs, task='ys', temperature=1, topk=None):
        self.eval()
        with torch.no_grad():
            eout_dict = self.encode(xs, task)
            dir = 'fwd' if self.fwd_weight >= self.bwd_weight else 'bwd'
            if task == 'ys_sub1':
                dir += '_sub1'
            elif task == 'ys_sub2':
                dir += '_sub2'

            if task == 'ys':
                assert self.ctc_weight > 0
            elif task == 'ys_sub1':
                assert self.ctc_weight_sub1 > 0
            elif task == 'ys_sub2':
                assert self.ctc_weight_sub2 > 0
            ctc_probs, indices_topk = getattr(self, 'dec_' + dir).ctc_probs_topk(
                eout_dict[task]['xs'], temperature, topk)
            return tensor2np(ctc_probs), tensor2np(indices_topk), eout_dict[task]['xlens']

    def ctc_forced_align(self, xs, ys, task='ys'):
        """CTC-based forced alignment.

        Args:
            xs (FloatTensor): `[B, T, idim]`
            ys (List): length `B`, each of which contains a list of size `[L]`
        Returns:
            trigger_points (np.ndarray): `[B, L]`

        """
        self.eval()
        with torch.no_grad():
            eout_dict = self.encode(xs, 'ys')
            # NOTE: support the main task only
            trigger_points = getattr(self, 'dec_fwd').ctc_forced_align(
                eout_dict[task]['xs'], eout_dict[task]['xlens'], ys)
        return tensor2np(trigger_points)

    def plot_attention(self):
        """Plot attention weights during training."""
        # encoder
        self.enc._plot_attention(mkdir_join(self.save_path, 'enc_att_weights'))
        # decoder
        self.dec_fwd._plot_attention(mkdir_join(self.save_path, 'dec_att_weights'))
        if getattr(self, 'dec_fwd_sub1', None) is not None:
            self.dec_fwd_sub1._plot_attention(mkdir_join(self.save_path, 'dec_att_weights_sub1'))
        if getattr(self, 'dec_fwd_sub2', None) is not None:
            self.dec_fwd_sub2._plot_attention(mkdir_join(self.save_path, 'dec_att_weights_sub2'))

    def plot_ctc(self):
        """Plot CTC posteriors during training."""
        self.dec_fwd._plot_ctc(mkdir_join(self.save_path, 'ctc'))
        if getattr(self, 'dec_fwd_sub1', None) is not None:
            self.dec_fwd_sub1._plot_ctc(mkdir_join(self.save_path, 'ctc_sub1'))
        if getattr(self, 'dec_fwd_sub2', None) is not None:
            self.dec_fwd_sub2._plot_ctc(mkdir_join(self.save_path, 'ctc_sub2'))

    def decode_streaming(self, xs, params, idx2token, exclude_eos=False, task='ys'):
        assert task == 'ys'
        assert self.input_type == 'speech'
        assert self.ctc_weight > 0
        assert self.fwd_weight > 0
        assert len(xs) == 1  # batch size
        # assert params['recog_length_norm']
        global_params = copy.deepcopy(params)
        global_params['recog_max_len_ratio'] = 1.0
        block_sync = params['recog_block_sync']
        block_size = params['recog_block_sync_size']  # before subsampling

        streaming = Streaming(xs[0], params, self.enc, block_size)
        factor = self.enc.subsampling_factor
        block_size //= factor

        hyps = None
        best_hyp_id_stream = []
        is_reset = True  # for the first block

        stdout = False

        self.eval()
        with torch.no_grad():
            lm = getattr(self, 'lm_fwd', None)
            lm_second = getattr(self, 'lm_second', None)

            while True:
                # Encode input features block by block
                x_block, is_last_block, lookback, lookahead = streaming.extract_feature()
                if is_reset:
                    self.enc.reset_cache()
                eout_block_dict = self.encode([x_block], 'all', streaming=True,
                                              lookback=lookback, lookahead=lookahead)
                eout_block = eout_block_dict[task]['xs']
                is_reset = False  # detect the first boundary in the same block

                # CTC-based VAD
                if streaming.is_ctc_vad:
                    if self.ctc_weight_sub1 > 0:
                        ctc_probs_block = self.dec_fwd_sub1.ctc_probs(eout_block_dict['ys_sub1']['xs'])
                        # TODO: consider subsampling
                    else:
                        ctc_probs_block = self.dec_fwd.ctc_probs(eout_block)
                    is_reset = streaming.ctc_vad(ctc_probs_block, stdout=stdout)

                # Truncate the most right frames
                if is_reset and not is_last_block and streaming.bd_offset >= 0:
                    eout_block = eout_block[:, :streaming.bd_offset]
                streaming.cache_eout(eout_block)

                # Block-synchronous attention decoding
                if isinstance(self.dec_fwd, RNNT):
                    raise NotImplementedError
                elif isinstance(self.dec_fwd, RNNDecoder) and block_sync:
                    for i_block in range(math.ceil(eout_block.size(1) / block_size)):
                        eout_block_i = eout_block[:, i_block * block_size:(i_block + 1) * block_size]
                        end_hyps, hyps, _ = self.dec_fwd.beam_search_block_sync(
                            eout_block_i, params, idx2token, hyps, lm,
                            state_carry_over=False)
                    merged_hyps = sorted(end_hyps + hyps, key=lambda x: x['score'], reverse=True)
                    if len(merged_hyps) > 0:
                        best_hyp_id_prefix = np.array(merged_hyps[0]['hyp'][1:])
                        if len(best_hyp_id_prefix) > 0 and best_hyp_id_prefix[-1] == self.eos:
                            # reset beam if <eos> is generated from the best hypothesis
                            best_hyp_id_prefix = best_hyp_id_prefix[:-1]  # exclude <eos>
                            # Segmentation strategy 2:
                            # If <eos> is emitted from the decoder (not CTC),
                            # the current block is segmented.
                            if not is_reset:
                                streaming._bd_offset = eout_block.size(1) - 1
                                # TODO: fix later
                            is_reset = True
                        if len(best_hyp_id_prefix) > 0:
                            # print('\rStreaming (T:%d [frame], offset:%d [frame], blank:%d [frame]): %s' %
                            #       (streaming.offset + eout_block.size(1) * factor,
                            #        self.dec_fwd.n_frames * factor,
                            #        streaming.n_blanks * factor,
                            #        idx2token(best_hyp_id_prefix)))
                            print('\r%s' % (idx2token(best_hyp_id_prefix)))

                if is_reset:
                    # Global decoding over the segmented region
                    if not block_sync:
                        eout = streaming.pop_eouts()
                        elens = torch.IntTensor([eout.size(1)])
                        ctc_log_probs = None
                        if params['recog_ctc_weight'] > 0:
                            ctc_log_probs = torch.log(self.dec_fwd.ctc_probs(eout))
                        nbest_hyps_id_offline = self.dec_fwd.beam_search(
                            eout, elens, global_params, idx2token, lm, lm_second,
                            ctc_log_probs=ctc_log_probs)[0]
                        # print('Offline (T:%d [10ms]): %s' %
                        #       (streaming.offset + eout_block.size(1) * factor,
                        #        idx2token(nbest_hyps_id_offline[0][0])))

                    # pick up the best hyp from ended and active hypotheses
                    if not block_sync:
                        if len(nbest_hyps_id_offline[0][0]) > 0:
                            best_hyp_id_stream.extend(nbest_hyps_id_offline[0][0])
                    else:
                        if len(best_hyp_id_prefix) > 0:
                            best_hyp_id_stream.extend(best_hyp_id_prefix)
                        # print('Final (T:%d [10ms], offset:%d [10ms]): %s' %
                        #       (streaming.offset + eout_block.size(1) * factor,
                        #        self.dec_fwd.n_frames * factor,
                        #        idx2token(best_hyp_id_prefix)))
                        # print('-' * 50)
                        # for test
                        # eos_hyp = np.zeros(1, dtype=np.int32)
                        # eos_hyp[0] = self.eos
                        # best_hyp_id_stream.extend(eos_hyp)

                    # reset
                    streaming.reset(stdout=stdout)
                    hyps = None

                streaming.next_block()
                # next block will start from the frame next to the boundary
                if not is_last_block:
                    streaming.backoff(x_block, self.dec_fwd, stdout=stdout)
                if is_last_block:
                    break

            # Global decoding for tail blocks
            if not block_sync and streaming.n_cache_block > 0:
                eout = torch.cat(streaming.eout_blocks, dim=1)
                elens = torch.IntTensor([eout.size(1)])
                nbest_hyps_id_offline = self.dec_fwd.beam_search(
                    eout, elens, global_params, idx2token, lm, lm_second)[0]
                # print('MoChA: ' + idx2token(nbest_hyps_id_offline[0][0]))
                # print('*' * 50)
                if len(nbest_hyps_id_offline[0][0]) > 0:
                    best_hyp_id_stream.extend(nbest_hyps_id_offline[0][0])

            # pick up the best hyp
            if not is_reset and block_sync and len(best_hyp_id_prefix) > 0:
                best_hyp_id_stream.extend(best_hyp_id_prefix)

            if len(best_hyp_id_stream) > 0:
                return [[np.stack(best_hyp_id_stream, axis=0)]], [None]
            else:
                return [[[]]], [None]

    def streamable(self):
        return getattr(self.dec_fwd, 'streamable', False)

    def quantity_rate(self):
        return getattr(self.dec_fwd, 'quantity_rate', 1.0)

    def last_success_frame_ratio(self):
        return getattr(self.dec_fwd, 'last_success_frame_ratio', 0)

    def decode(self, xs, params, idx2token, exclude_eos=False,
               refs_id=None, refs=None, utt_ids=None, speakers=None,
               task='ys', ensemble_models=[], trigger_points=None, teacher_force=False):
        """Decode in the inference stage.

        Args:
            xs (List): length `[B]`, which contains arrays of size `[T, input_dim]`
            params (dict): hyper-parameters for decoding
                beam_width (int): the size of beam
                min_len_ratio (float):
                max_len_ratio (float):
                len_penalty (float): length penalty
                cov_penalty (float): coverage penalty
                cov_threshold (float): threshold for coverage penalty
                lm_weight (float): the weight of RNNLM score
                resolving_unk (bool): not used (to make compatible)
                fwd_bwd_attention (bool):
            idx2token (): converter from index to token
            exclude_eos (bool): exclude <eos> from best_hyps_id
            refs_id (List): gold token IDs to compute log likelihood
            refs (List): gold transcriptions
            utt_ids (List):
            speakers (List):
            task (str): ys* or ys_sub1* or ys_sub2*
            ensemble_models (List): Speech2Text classes
            trigger_points (np.ndarray): `[B, L]`
            teacher_force (bool): conduct teacher-forcing
        Returns:
            nbest_hyps_id (List[List[np.ndarray]]): length `[B]`, which contains a list of length `[n_best]` which contains arrays of size `[L]`
            aws (List[np.ndarray]): length `[B]`, which contains arrays of size `[L, T, n_heads]`

        """
        if task.split('.')[0] == 'ys':
            dir = 'bwd' if self.bwd_weight > 0 and params['recog_bwd_attention'] else 'fwd'
        elif task.split('.')[0] == 'ys_sub1':
            dir = 'fwd_sub1'
        elif task.split('.')[0] == 'ys_sub2':
            dir = 'fwd_sub2'
        else:
            raise ValueError(task)

        if utt_ids is not None:
            if self.utt_id_prev != utt_ids[0]:
                self.reset_session()
            self.utt_id_prev = utt_ids[0]

        self.eval()
        with torch.no_grad():
            # Encode input features
            eout_dict = self.encode(xs, task)

            # CTC
            if (self.fwd_weight == 0 and self.bwd_weight == 0) or (self.ctc_weight > 0 and params['recog_ctc_weight'] == 1):
                lm = getattr(self, 'lm_' + dir, None)
                lm_second = getattr(self, 'lm_second', None)
                lm_second_bwd = None  # TODO

                nbest_hyps_id = getattr(self, 'dec_' + dir).decode_ctc(
                    eout_dict[task]['xs'], eout_dict[task]['xlens'], params, idx2token,
                    lm, lm_second, lm_second_bwd,
                    params['recog_beam_width'], refs_id, utt_ids, speakers)
                return nbest_hyps_id, None

            # Attention/RNN-T
            elif params['recog_beam_width'] == 1 and not params['recog_fwd_bwd_attention']:
                best_hyps_id, aws = getattr(self, 'dec_' + dir).greedy(
                    eout_dict[task]['xs'], eout_dict[task]['xlens'],
                    params['recog_max_len_ratio'], idx2token,
                    exclude_eos, refs_id, utt_ids, speakers)
                nbest_hyps_id = [[hyp] for hyp in best_hyps_id]
            else:
                assert params['recog_batch_size'] == 1

                ctc_log_probs = None
                if params['recog_ctc_weight'] > 0:
                    ctc_log_probs = self.dec_fwd.ctc_log_probs(eout_dict[task]['xs'])

                # forward-backward decoding
                if params['recog_fwd_bwd_attention']:
                    lm = getattr(self, 'lm_fwd', None)
                    lm_bwd = getattr(self, 'lm_bwd', None)

                    # forward decoder
                    nbest_hyps_id_fwd, aws_fwd, scores_fwd = self.dec_fwd.beam_search(
                        eout_dict[task]['xs'], eout_dict[task]['xlens'],
                        params, idx2token, lm, None, lm_bwd, ctc_log_probs,
                        params['recog_beam_width'], False, refs_id, utt_ids, speakers)

                    # backward decoder
                    nbest_hyps_id_bwd, aws_bwd, scores_bwd, _ = self.dec_bwd.beam_search(
                        eout_dict[task]['xs'], eout_dict[task]['xlens'],
                        params, idx2token, lm_bwd, None, lm, ctc_log_probs,
                        params['recog_beam_width'], False, refs_id, utt_ids, speakers)

                    # forward-backward attention
                    best_hyps_id = fwd_bwd_attention(
                        nbest_hyps_id_fwd, aws_fwd, scores_fwd,
                        nbest_hyps_id_bwd, aws_bwd, scores_bwd,
                        self.eos, params['recog_gnmt_decoding'], params['recog_length_penalty'],
                        idx2token, refs_id)
                    nbest_hyps_id = [[hyp] for hyp in best_hyps_id]
                    aws = None
                else:
                    # ensemble
                    ensmbl_eouts, ensmbl_elens, ensmbl_decs = [], [], []
                    if len(ensemble_models) > 0:
                        for i_e, model in enumerate(ensemble_models):
                            enc_outs_e = model.encode(xs, task)
                            ensmbl_eouts += [enc_outs_e[task]['xs']]
                            ensmbl_elens += [enc_outs_e[task]['xlens']]
                            ensmbl_decs += [getattr(model, 'dec_' + dir)]
                            # NOTE: only support for the main task now

                    lm = getattr(self, 'lm_' + dir, None)
                    lm_second = getattr(self, 'lm_second', None)
                    lm_bwd = getattr(self, 'lm_bwd' if dir == 'fwd' else 'lm_bwd', None)

                    nbest_hyps_id, aws, scores = getattr(self, 'dec_' + dir).beam_search(
                        eout_dict[task]['xs'], eout_dict[task]['xlens'],
                        params, idx2token, lm, lm_second, lm_bwd, ctc_log_probs,
                        params['recog_beam_width'], exclude_eos, refs_id, utt_ids, speakers,
                        ensmbl_eouts, ensmbl_elens, ensmbl_decs)

            return nbest_hyps_id, aws
