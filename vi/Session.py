import logging
import math
import os.path
import os
import time
import sys
import random

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torchtext.data import BucketIterator, Iterator
from torchtext.vocab import FastText, GloVe
from torch.nn.utils import clip_grad_norm_

from vi_syntax.vi.modules.Decoder import Decoder
from vi_syntax.vi.modules.Encoder import Encoder
from vi_syntax.vi.modules.RNNLM import LanguageModel
from vi_syntax.vi.modules.RNNLM import Baseline_linear
from vi_syntax.vi.modules.PR import PR
from vi_syntax.vi.modules.Utils import compute_dda, compute_rule_acc, compute_dda_long_dep
from vi_syntax.vi.myio.IO import VIDataset


class Session(object):

    def __init__(self, opt):
        self.opt = opt
        self.logger = self.create_logger()
        self.initializers = {'glorot': init.xavier_normal_,
                             'constant': lambda x: init.constant(x, 0.01),
                             'uniform': lambda x: init.uniform(x, a=-0.1, b=0.1),
                             'normal': lambda x: init.normal_(x, mean=0, std=1)
                            }

        self.optimizers = {'sgd': lambda x: optim.SGD(x, lr=0.1, momentum=0.9),
                           'adam': lambda x: optim.Adam(x, lr=self.opt.lr),
                           'adadelta': lambda x: optim.Adadelta(x, lr=self.opt.lr),
                           'adagrad': lambda x: optim.Adagrad(x, lr=self.opt.lr, weight_decay=1e-4)
                          }

    def _build_wsj_dataset(self):
        """
        load wsj dataset
        """
        self.dataset = VIDataset()
        self.dataset.build_wsj_enhanced_dataset(data_path=self.opt.data_path, train_fname=self.opt.train_fname,
                                                valid_fname = self.opt.valid_fname, test_fname=self.opt.test_fname,
                                                max_length=9999, cluster_fname=self.opt.cluster_fname,
                                                min_freq=self.opt.min_freq,
                                                vectors=[GloVe(name='6B', dim='300',
                                                               cache = self.opt.word_vector_cache,
                                                               unk_init = torch.Tensor.normal_)])

        train_iter, self.valid_iter, self.test_iter = BucketIterator.splits(
            (self.dataset.train, self.dataset.valid, self.dataset.test),
            batch_size=self.opt.batchsize, sort_within_batch=True, device=self.opt.gpu_id)    # sort within batch
        self.train_sampler = train_iter.__iter__()
        self.valid_sampler = self.valid_iter.__iter__()
        self.test_sampler = self.test_iter.__iter__()

    def _build_ud_dataset(self):
        """
        load ud dataset
        """
        self.dataset = VIDataset()
        self.dataset.build_ud_enhanced_dataset(data_path=self.opt.data_path, train_fname=self.opt.train_fname,
                                               valid_fname=self.opt.valid_fname, test_fname=self.opt.test_fname,
                                               max_length=999,
                                               min_freq=self.opt.min_freq,
                                               vectors = [FastText(language=self.opt.language,
                                                           cache=self.opt.word_vector_cache,
                                                           unk_init=torch.Tensor.normal_)],
                                               cluster_fname=self.opt.cluster_fname)

        if self.opt.seed > 0:
            random.seed(self.opt.seed)
            train_iter, self.valid_iter, self.test_iter = BucketIterator.splits(
                (self.dataset.train, self.dataset.valid, self.dataset.test),
                batch_size=self.opt.batchsize,
                sort_within_batch=True,
                device=self.opt.gpu_id,
                random_state=random.getstate())    # sort within batch
        else:
            train_iter, self.valid_iter, self.test_iter = BucketIterator.splits(
                (self.dataset.train, self.dataset.valid, self.dataset.test),
                batch_size=self.opt.batchsize,
                sort_within_batch=True,
                device=self.opt.gpu_id)  # sort within batch
        self.train_sampler = train_iter.__iter__()
        self.valid_sampler = self.valid_iter.__iter__()
        self.test_sampler = self.test_iter.__iter__()

    def train_encoder(self):
        """
        train the encoder in a supervised setting
        """
        if self.opt.language != '':
            self._build_ud_dataset()
        else:
            self._build_wsj_dataset()
        self.encoder = self.create_encoder()
        self.optim = self.optimizers[self.opt.optimizer](
            [{'params': [param for param in self.encoder.parameters() if param.requires_grad]}])
        self.encoder.train()

        self.len_train = len(self.dataset.train)
        self.len_real_train = 0
        for i in range(1, self.len_train + 1):
            sample = self.train_sampler.next()
            if sample.word[1].item() <= self.opt.train_max_length:
                self.len_real_train += 1
        total_loss_act = 0.
        for epoch in range(1, self.opt.epochs + 1):
            cur_time = time.time()
            cur_sample = 0
            i = 0
            for _ in range(1, self.len_train + 1):
                self.optim.zero_grad()
                sample = self.train_sampler.next()
                if sample.word[1].item() > self.opt.train_max_length:
                    continue
                i += 1
                loss_act = self.encoder.train_parser(words=sample.word[0], pos_tags=sample.pos_tag, oracle_actions=sample.action)
                if loss_act is not None:
                    total_loss_act += loss_act.data.item()
                    loss_act.backward()
                    self.optim.step()
                if i % self.opt.print_every == 0 or i == self.len_real_train:
                    elapsed_time = time.time() - cur_time
                    cur_time = time.time()
                    elapsed_sample = i - cur_sample
                    cur_sample = i
                    self.logger.info('epoch {:3d} | {:5d}/{:5d} | avg loss act {:5.2f} | time {:5.2f}s'. \
                        format(epoch, i, self.len_real_train,
                               total_loss_act / elapsed_sample, elapsed_time))
                    total_loss_act = 0.
            self.logger.info('=' * 80)
            valid_dda = self.parse(self.valid_sampler)
            self.valid_sampler = self.valid_iter.__iter__() # renew the iterator
            self.logger.info('epoch {:3d} | valid dda {:5.2f}'.format(epoch, valid_dda))
            test_dda = self.parse(self.test_sampler)
            self.test_sampler = self.test_iter.__iter__()   # renew the iterator
            self.logger.info('epoch {:3d} | test dda {:5.2f}'.format(epoch, test_dda))
            self.logger.info('=' * 80)

    def train_decoder(self):
        """
        train the decoder in a supervised setting
        """
        if self.opt.language != '':
            self._build_ud_dataset()
        else:
            self._build_wsj_dataset()
        self.decoder = self.create_decoder()
        self.optim = self.optimizers[self.opt.optimizer](
            # [{'params': [param for name, param in self.decoder.named_parameters() if 'embedding' not in name]}])
            [{'params': [param for param in self.decoder.parameters() if param.requires_grad]}])
        self.decoder.train()

        self.len_train = len(self.dataset.train)
        self.len_real_train = 0
        for i in range(1, self.len_train + 1):
            sample = self.train_sampler.next()
            if sample.word[1].item() <= self.opt.train_max_length:
                self.len_real_train += 1
        total_loss_act = 0.
        total_loss_token = 0.
        total_loss = 0.
        for epoch in range(1, self.opt.epochs + 1):
            cur_time = time.time()
            cur_sample = 0
            i = 0
            for _ in range(1, self.len_train + 1):
                self.optim.zero_grad()
                sample = self.train_sampler.next()
                if sample.word[1].item() > self.opt.train_max_length:
                    continue
                i += 1
                loss, loss_act, loss_token = self.decoder(words=sample.word[0], pos_tags=sample.pos_tag, oracle_actions=sample.action)
                loss.backward()
                self.optim.step()
                if loss_act is not None:
                    total_loss_act += loss_act.data[0]
                total_loss += loss.data[0]
                total_loss_token += loss_token.data.item()
                if i % self.opt.print_every == 0 or i == self.len_real_train:
                    elapsed_time = time.time() - cur_time
                    cur_time = time.time()
                    elapsed_sample = i - cur_sample
                    cur_sample = i
                    self.logger.info('epoch {:3d} | {:5d}/{:5d} | avg loss act {:5.2f} | avg loss token {:5.2f} | avg loss {:5.2f} | time {:5.2f}s'. \
                        format(epoch, i, self.len_real_train, total_loss_act / elapsed_sample,
                               total_loss_token / elapsed_sample, total_loss / elapsed_sample,
                               elapsed_time))
                    total_loss_act = 0.
                    total_loss_token = 0.
                    total_loss = 0.
            self.logger.info('=' * 80)
            valid_loss, valid_loss_act, valid_loss_token = self.compute_joint_loss(self.valid_sampler)
            self.valid_sampler = self.valid_iter.__iter__() # renew the iterator
            self.logger.info('epoch {:3d} | avg valid loss act {:5.2f} | avg valid loss token {:5.2f} | avg valid loss {:5.2f}'
                             .format(epoch, valid_loss_act, valid_loss_token, valid_loss))
            test_loss, test_loss_act, test_loss_token = self.compute_joint_loss(self.test_sampler)
            self.test_sampler = self.test_iter.__iter__()   # renew the iterator
            self.logger.info('epoch {:3d} | avg test loss act {:5.2f} | avg test loss token {:5.2f} | avg test loss {:5.2f}'
                             .format(epoch, test_loss_act, test_loss_token, test_loss))
            self.logger.info('=' * 80)

    def train_lm(self):
        """
        train language model
        """
        self.opt.cluster_fname = None
        self._build_wsj_dataset()
        self.lm = self.create_lm()
        self.lm.train()

        self.optim = self.optimizers[self.opt.optimizer](
            [{'params': [param for param in self.lm.parameters() if param.requires_grad]}])

        self.len_train = len(self.dataset.train) / self.opt.batchsize + 1
        self.len_real_train = 0
        for i in range(1, self.len_train + 1):
            sample = self.train_sampler.next()
            if torch.sum(sample.word[1]).item() <= self.opt.train_max_length * self.opt.batchsize + 3:
                self.len_real_train += 1
        total_loss = 0.
        best_valid_ppl = 10.e9
        print 'real train length', self.len_real_train
        for epoch in range(1, self.opt.epochs + 1):
            cur_time = time.time()
            elapsed_token = 0
            for i in range(1, self.len_real_train + 1):
                self.optim.zero_grad()
                sample = self.train_sampler.next()
                while torch.sum(sample.word[1]).item() > (self.opt.train_max_length + 1) * self.opt.batchsize:
                    sample = self.train_sampler.next()
                loss = self.lm(words=sample.word, pos_tags=sample.pos_tag)
                loss.backward()
                total_loss += loss.cpu().item()
                clip_grad_norm_([param for param in self.lm.parameters() if param.requires_grad], self.opt.clip)
                self.optim.step()
                elapsed_token += torch.sum(sample.word[1]).item()
                if i % self.opt.print_every == 0 or i == self.len_real_train:
                    elapsed_time = time.time() - cur_time
                    cur_time = time.time()
                    self.logger.info('epoch {:3d} | {:5d}/{:5d} | ppl {:5.2f} | time {:5.2f}s'. \
                        format(epoch, i, self.len_real_train,
                               math.exp(total_loss / elapsed_token), elapsed_time))
                    elapsed_token = 0
                    total_loss = 0.
            self.logger.info('=' * 80)
            valid_ppl = self.compute_lm_ppl(self.valid_sampler)
            self.valid_sampler = self.valid_iter.__iter__() # renew the iterator
            self.logger.info('epoch {:3d} | valid ppl {:5.2f}'.format(epoch, valid_ppl))
            test_ppl = self.compute_lm_ppl(self.test_sampler)
            self.test_sampler = self.test_iter.__iter__() # renew the iterator
            self.logger.info('epoch {:3d} | test ppl {:5.2f}'.format(epoch, test_ppl))
            self.logger.info('=' * 80)
            if self.opt.save_model:
                if valid_ppl < best_valid_ppl:
                    prev_model_fname = os.path.join(self.opt.result_dir,
                                                   '{}_lm_valid-ppl-{:.2f}_len-{}.pt'
                                                   .format(self.opt.log_name, best_valid_ppl, self.opt.train_max_length))
                    if os.path.exists(prev_model_fname):
                        os.remove(prev_model_fname)
                    best_valid_ppl = valid_ppl
                    cur_model_fname = os.path.join(self.opt.result_dir,
                                                   '{}_lm_valid-ppl-{:.2f}_len-{}.pt'
                                                   .format(self.opt.log_name, best_valid_ppl, self.opt.train_max_length))
                    self.lm.save(cur_model_fname)

    def nvil_pr_pretrain(self):
        """
        Pretrain both encoder and decoder using posterior regularization as the direct reward
        batchsize here means nvil batch (not exact devision here!)
        """
        self._build_wsj_dataset()
        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()
        self.pr = self.create_pr()
        enc_param = [param for param in self.encoder.parameters() if param.requires_grad]
        dec_param = [param for param in self.decoder.parameters() if param.requires_grad]
        pr_param = [param for param in self.pr.parameters() if param.requires_grad]
        self.enc_optim = self.optimizers[self.opt.optimizer](
            [{'params': enc_param}])
        self.dec_optim = self.optimizers[self.opt.optimizer](
            [{'params': dec_param}])
        self.pr_optim = self.optimizers[self.opt.pr_optimizer](
            [{'params': pr_param}])
        self.encoder.train()
        self.decoder.train()
        self.pr.train()
        self.enc_optim.zero_grad()
        self.dec_optim.zero_grad()
        self.pr_optim.zero_grad()

        self.logger.info('=' * 80)
        valid_dda = self.parse(self.valid_sampler)
        self.valid_sampler = self.valid_iter.__iter__() # renew the iterator
        self.logger.info('epoch {:3d} | valid dda {:5.2f}'.format(0, valid_dda))
        self.logger.info('=' * 80)

        self.len_train = len(self.dataset.train)
        self.len_real_train = 0.
        for i in range(1, self.len_train + 1):
            sample = self.train_sampler.next()
            if sample.word[1].item() <= self.opt.train_max_length:
                self.len_real_train += 1
        self.len_real_train = int(math.ceil(self.len_real_train / self.opt.nvil_batchsize))
        total_enc_loss = 0.
        total_dec_loss = 0.
        total_score_mean = 0.
        total_score_var = 0.
        best_valid_dda = 0.
        for epoch in range(1, self.opt.epochs + 1):
            cur_time = time.time()
            cur_batch = 0
            i = 0
            for _ in range(1, self.len_real_train + 1):
                batch = []
                while len(batch) < self.opt.nvil_batchsize:
                    sample = self.train_sampler.next()
                    if sample.word[1].item() <= self.opt.train_max_length:
                        batch.append(sample)
                i += 1
                for sample in batch:
                    enc_loss_act_list = []
                    dec_loss_list = []
                    for mc in range(self.opt.mc_samples):
                        enc_loss_act, predicted_act, feature = self.encoder.parse_pr(
                            sample.word[0], sample.pos_tag, self.pr.rule2i, sample=True)
                        self.pr.phi.data[mc] = feature
                        enc_loss_act_list.append(enc_loss_act)
                        dec_loss, _, _ = self.decoder(
                            words=sample.word[0], pos_tags=sample.pos_tag, oracle_actions=predicted_act)
                        dec_loss_list.append(dec_loss)
                    if sample.word[0].size(1) == 1:
                        continue    # skip backward

                    # update posterior regularizer
                    pr_factor = torch.ones(self.opt.mc_samples)
                    if self.opt.gpu_id > -1:
                        pr_factor.cuda()
                    if torch.sum(self.pr.phi).item() < 0:
                        pr_loss, pr_factor = self.pr()
                        pr_loss.backward()
                    self.pr.reset_phi()

                    # backward w.r.t. encoder and decoder
                    for mc in range(self.opt.mc_samples):
                        enc_loss_act = enc_loss_act_list[mc]
                        dec_loss = dec_loss_list[mc]
                        total_dec_loss += dec_loss.item()
                        total_enc_loss += enc_loss_act.item()

                        dec_loss = dec_loss * pr_factor[mc].item() / self.opt.mc_samples
                        dec_loss.backward()

                        enc_loss_act = enc_loss_act * pr_factor[mc].item() / self.opt.mc_samples
                        enc_loss_act.backward()

                clip_grad_norm_(enc_param + dec_param, self.opt.clip)
                self.enc_optim.step()
                self.dec_optim.step()
                self.pr_optim.step()
                self.pr.project()
                self.enc_optim.zero_grad()
                self.dec_optim.zero_grad()
                self.pr_optim.zero_grad()

                if i % self.opt.print_every == 0 or i == self.len_real_train:
                    elapsed_time = time.time() - cur_time
                    cur_time = time.time()
                    elapsed_batch = i - cur_batch
                    cur_batch = i
                    self.logger.info('epoch {:3d} | {:5d}/{:5d} | avg enc loss {:5.2f} | avg dec loss {:5.2f} | time {:5.2f}s'. \
                        format(epoch, i, self.len_real_train, total_enc_loss / elapsed_batch / self.opt.nvil_batchsize / self.opt.mc_samples,
                               total_dec_loss / elapsed_batch / self.opt.nvil_batchsize / self.opt.mc_samples, elapsed_time))
                    total_enc_loss = 0.
                    total_dec_loss = 0.

                if i % self.opt.save_every == 0 or i == self.len_real_train:
                    # validate
                    self.logger.info('=' * 80)
                    valid_dda = self.parse(self.valid_sampler)
                    self.valid_sampler = self.valid_iter.__iter__() # renew the iterator
                    self.logger.info('epoch {:3d} | valid dda {:5.4f}'.format(epoch, valid_dda))

                    if valid_dda > best_valid_dda:
                        # save encoder model
                        prev_enc_fname = os.path.join(self.opt.result_dir,
                                                   '{}_enc_valid-dda-{:.4f}_len-{}.pt'
                                                   .format(self.opt.log_name, best_valid_dda, self.opt.train_max_length))
                        if os.path.exists(prev_enc_fname):
                            os.remove(prev_enc_fname)
                        cur_enc_fname = os.path.join(self.opt.result_dir,
                                                   '{}_enc_valid-dda-{:.4f}_len-{}.pt'
                                                   .format(self.opt.log_name, valid_dda, self.opt.train_max_length))
                        self.encoder.save(cur_enc_fname)

                        # save decoder model
                        prev_dec_fname = os.path.join(self.opt.result_dir,
                                                   '{}_dec_valid-dda-{:.4f}_len-{}.pt'
                                                   .format(self.opt.log_name, best_valid_dda, self.opt.train_max_length))
                        if os.path.exists(prev_dec_fname):
                            os.remove(prev_dec_fname)
                        cur_dec_fname = os.path.join(self.opt.result_dir,
                                                   '{}_dec_valid-dda-{:.4f}_len-{}.pt'
                                                   .format(self.opt.log_name, valid_dda, self.opt.train_max_length))
                        self.decoder.save(cur_dec_fname)

                        # test short/long
                        best_valid_dda = valid_dda
                        max_length = 10
                        test_dda = self.parse(self.test_sampler, max_length=max_length)
                        self.test_sampler = self.test_iter.__iter__() # renew the iterator
                        self.logger.info('epoch {:3d} | test dda-{:2d} {:5.4f}'.format(epoch, max_length, test_dda))
                        # max_length = 9999
                        # valid_dda = self.parse(self.valid_sampler, max_length=max_length)
                        # self.valid_sampler = self.test_iter.__iter__() # renew the iterator
                        # self.logger.info('epoch {:3d} | valid dda-{:2d} {:5.4f}'.format(epoch, max_length, valid_dda))
                        # test_dda = self.parse(self.test_sampler, max_length=max_length)
                        # self.test_sampler = self.test_iter.__iter__() # renew the iterator
                        # self.logger.info('epoch {:3d} | test dda-{:2d} {:5.4f}'.format(epoch, max_length, test_dda))

                    self.logger.info('=' * 80)

    def nvil_pr_ft(self):
        """
        Finetune both encoder and decoder based on the pretrained model using different functions 
        (RL-SN/RL-C/RL-PC details can be found in the paper)
        batchsize here means nvil batch (not exact devision here!)
        """
        self._build_wsj_dataset()
        self.encoder = self.create_encoder(self.opt.encoder_fname)
        self.decoder = self.create_decoder(self.opt.decoder_fname)
        self.pr = self.create_pr()
        enc_param = [param for param in self.encoder.parameters() if param.requires_grad]
        dec_param = [param for param in self.decoder.parameters() if param.requires_grad]
        pr_param = [param for param in self.pr.parameters() if param.requires_grad]
        self.enc_optim = self.optimizers[self.opt.optimizer](
            [{'params': enc_param}])
        self.dec_optim = self.optimizers[self.opt.optimizer](
            [{'params': dec_param}])
        self.pr_optim = self.optimizers[self.opt.pr_optimizer](
            [{'params': pr_param}])
        self.encoder.train()
        self.decoder.train()
        self.pr.train()
        self.enc_optim.zero_grad()
        self.dec_optim.zero_grad()
        self.pr_optim.zero_grad()

        self.logger.info('=' * 80)
        valid_dda = self.parse(self.valid_sampler)
        self.valid_sampler = self.valid_iter.__iter__() # renew the iterator
        self.logger.info('initial | valid dda {:5.4f}'.format(valid_dda))
        self.logger.info('=' * 80)

        self.len_train = len(self.dataset.train)
        self.len_real_train = 0.
        for i in range(1, self.len_train + 1):
            sample = self.train_sampler.next()
            if sample.word[1].item() <= self.opt.train_max_length:
                self.len_real_train += 1
        self.len_real_train = int(math.ceil(self.len_real_train / self.opt.nvil_batchsize))
        total_enc_loss = 0.
        total_dec_loss = 0.
        best_valid_dda = 0.
        for epoch in range(1, self.opt.epochs + 1):
            cur_time = time.time()
            cur_batch = 0
            i = 0
            for _ in range(1, self.len_real_train + 1):
                batch = []
                while len(batch) < self.opt.nvil_batchsize:
                    sample = self.train_sampler.next()
                    if sample.word[1].item() <= self.opt.train_max_length:
                        batch.append(sample)
                i += 1
                for sample in batch:
                    enc_loss_act_list = []
                    dec_loss_list = []
                    for mc in range(self.opt.mc_samples):
                        enc_loss_act, predicted_act, feature = self.encoder.parse_pr(
                            sample.word[0], sample.pos_tag, self.pr.rule2i, sample=True)
                        self.pr.phi.data[mc] = feature
                        enc_loss_act_list.append(enc_loss_act)
                        dec_loss, _, _ = self.decoder(
                            words=sample.word[0], pos_tags=sample.pos_tag, oracle_actions=predicted_act)
                        dec_loss_list.append(dec_loss)

                    if sample.word[0].size(1) == 1:
                        continue    # skip backward

                    # update posterior regulizer
                    pr_factor = torch.ones(self.opt.mc_samples)
                    if self.opt.gpu_id > -1:
                        pr_factor.cuda()
                    if torch.sum(self.pr.phi).item() < 0:
                        pr_loss, pr_factor = self.pr()
                        pr_loss.backward()

                    phi = torch.sum(self.pr.phi, dim=1)
                    normalized_phi = (phi - torch.mean(phi))

                    # show SFE
                    # print 'lambda', self.pr.Lambda.weight
                    phi = phi.cpu().numpy() # for display

                    self.pr.reset_phi()
                    score_list = None

                    # get score
                    for mc in range(self.opt.mc_samples):
                        enc_loss_act = enc_loss_act_list[mc]
                        dec_loss = dec_loss_list[mc]
                        total_dec_loss += dec_loss.item()
                        total_enc_loss += enc_loss_act.item()

                        score = - dec_loss + enc_loss_act
                        score.unsqueeze_(0)
                        if score_list is None:
                            score_list = score
                        else:
                            score_list = torch.cat((score_list, score))

                    # normalize scores
                    score_mean = torch.mean(score_list)
                    score_std = torch.std(score_list)
                    nomalized_score_list = (score_list - score_mean) / score_std

                    # backward w.r.t. encoder decoder
                    for mc in range(self.opt.mc_samples):
                        enc_loss_act = enc_loss_act_list[mc]
                        dec_loss = dec_loss_list[mc]
                        score = nomalized_score_list[mc]

                        # RL-PC
                        if normalized_phi[mc].item() < 0:
                            score = abs(score.item())
                        else:
                            score = - abs(score.item())
                        enc_loss_act = enc_loss_act * score * pr_factor[mc].item() / self.opt.mc_samples
                        dec_loss = dec_loss * score * pr_factor[mc].item() / self.opt.mc_samples

                        # # RL-SN
                        # score = score.item()
                        # enc_loss_act = enc_loss_act * score * pr_factor[mc].item() / self.opt.mc_samples
                        # dec_loss = dec_loss * score * pr_factor[mc].item() / self.opt.mc_samples

                        # # RL-C
                        # enc_loss_act = enc_loss_act * (-normalized_phi[mc].item()) * pr_factor[mc].item() / self.opt.mc_samples
                        # dec_loss = dec_loss * (-normalized_phi[mc].item()) * pr_factor[mc].item() / self.opt.mc_samples

                        # backward
                        enc_loss_act.backward()
                        dec_loss.backward()

                clip_grad_norm_(enc_param + dec_param + pr_param, self.opt.clip)
                self.enc_optim.step()
                self.dec_optim.step()
                self.pr_optim.step()
                self.pr.project()
                self.enc_optim.zero_grad()
                self.dec_optim.zero_grad()
                self.pr_optim.zero_grad()

                if i % self.opt.print_every == 0 or i == self.len_real_train:
                    elapsed_time = time.time() - cur_time
                    cur_time = time.time()
                    elapsed_batch = i - cur_batch
                    cur_batch = i
                    self.logger.info('epoch {:3d} | {:5d}/{:5d} | avg enc loss {:5.2f} | avg dec loss {:5.2f} | time {:5.2f}s'. \
                        format(epoch, i, self.len_real_train, total_enc_loss / elapsed_batch / self.opt.nvil_batchsize / self.opt.mc_samples,
                               total_dec_loss / elapsed_batch / self.opt.nvil_batchsize / self.opt.mc_samples, elapsed_time))
                    total_enc_loss = 0.
                    total_dec_loss = 0.

                if i % self.opt.save_every == 0 or i == self.len_train:
                    # validate
                    self.logger.info('=' * 80)
                    valid_dda = self.parse(self.valid_sampler)
                    self.valid_sampler = self.valid_iter.__iter__() # renew the iterator
                    self.logger.info('epoch {:3d} | valid dda {:5.4f}'.format(epoch, valid_dda))

                    if valid_dda > best_valid_dda:
                        # save encoder model
                        prev_enc_fname = os.path.join(self.opt.result_dir,
                                                   '{}_enc_valid-dda-{:.4f}_len-{}.pt'
                                                   .format(self.opt.log_name, best_valid_dda, self.opt.train_max_length))
                        if os.path.exists(prev_enc_fname):
                            os.remove(prev_enc_fname)
                        cur_enc_fname = os.path.join(self.opt.result_dir,
                                                   '{}_enc_valid-dda-{:.4f}_len-{}.pt'
                                                   .format(self.opt.log_name, valid_dda, self.opt.train_max_length))
                        self.encoder.save(cur_enc_fname)

                        # save decoder model
                        prev_dec_fname = os.path.join(self.opt.result_dir,
                                                   '{}_dec_valid-dda-{:.4f}_len-{}.pt'
                                                   .format(self.opt.log_name, best_valid_dda, self.opt.train_max_length))
                        if os.path.exists(prev_dec_fname):
                            os.remove(prev_dec_fname)
                        cur_dec_fname = os.path.join(self.opt.result_dir,
                                                   '{}_dec_valid-dda-{:.4f}_len-{}.pt'
                                                   .format(self.opt.log_name, valid_dda, self.opt.train_max_length))
                        self.decoder.save(cur_dec_fname)

                        # test
                        best_valid_dda = valid_dda
                        max_length = 10
                        test_dda = self.parse(self.test_sampler, max_length=max_length)
                        self.test_sampler = self.test_iter.__iter__() # renew the iterator
                        self.logger.info('epoch {:3d} | test dda-{:d} {:5.4f}'.format(epoch, max_length, test_dda))

                    self.logger.info('=' * 80)

    def rl_pr_ft(self):
        """
        REINFORCE + baseline + posterior regularized (\cite{Miao})
        """
        self._build_wsj_dataset()
        self.lm = self.create_lm(self.opt.lm_fname)
        self.lm.eval()
        self.encoder = self.create_encoder(self.opt.encoder_fname)
        self.decoder = self.create_decoder(self.opt.decoder_fname)
        self.pr = self.create_pr()
        self.bl_linear = self.create_baseline_linear()
        bl_criterion = nn.MSELoss()
        enc_param = [param for param in self.encoder.parameters() if param.requires_grad]
        dec_param = [param for param in self.decoder.parameters() if param.requires_grad]
        pr_param = [param for param in self.pr.parameters() if param.requires_grad]
        bl_linear_param = [param for param in self.bl_linear.parameters()]
        self.enc_optim = self.optimizers[self.opt.optimizer](
            [{'params': enc_param}])
        self.dec_optim = self.optimizers[self.opt.optimizer](
            [{'params': dec_param}])
        self.pr_optim = self.optimizers[self.opt.pr_optimizer](
            [{'params': pr_param}])
        self.bl_linear_optim = self.optimizers[self.opt.pr_optimizer](
            [{'params': bl_linear_param}])
        self.encoder.train()
        self.decoder.train()
        self.pr.train()
        self.enc_optim.zero_grad()
        self.dec_optim.zero_grad()
        self.pr_optim.zero_grad()
        self.bl_linear_optim.zero_grad()

        self.logger.info('=' * 80)
        valid_dda = self.parse(self.valid_sampler)
        self.valid_sampler = self.valid_iter.__iter__() # renew the iterator
        self.logger.info('initial | valid dda {:5.4f}'.format(valid_dda))
        self.logger.info('=' * 80)

        self.len_train = len(self.dataset.train)
        self.len_real_train = 0.
        for i in range(1, self.len_train + 1):
            sample = self.train_sampler.next()
            if sample.word[1].item() <= self.opt.train_max_length:
                self.len_real_train += 1
        self.len_real_train = int(math.ceil(self.len_real_train / self.opt.nvil_batchsize))
        total_enc_loss = 0.
        total_dec_loss = 0.
        best_valid_dda = 0.
        for epoch in range(1, self.opt.epochs + 1):
            cur_time = time.time()
            cur_batch = 0
            i = 0
            for _ in range(1, self.len_real_train + 1):
                batch = []
                while len(batch) < self.opt.nvil_batchsize:
                    sample = self.train_sampler.next()
                    if sample.word[1].item() <= self.opt.train_max_length:
                        batch.append(sample)
                i += 1
                for sample in batch:
                    enc_loss_act_list = []
                    baseline_list = []
                    dec_loss_list = []
                    for mc in range(self.opt.mc_samples):
                        enc_loss_act, predicted_act, feature = self.encoder.parse_pr(
                            sample.word[0], sample.pos_tag, self.pr.rule2i, sample=True)
                        self.pr.phi.data[mc] = feature
                        enc_loss_act_list.append(enc_loss_act)
                        dec_loss, _, _ = self.decoder(
                            words=sample.word[0], pos_tags=sample.pos_tag, oracle_actions=predicted_act)
                        dec_loss_list.append(dec_loss)
                        lm_loss = self.lm(words=sample.word, pos_tags=sample.pos_tag).unsqueeze(0)
                        baseline_list.append(self.bl_linear(lm_loss))

                    if sample.word[0].size(1) == 1:
                        continue    # skip backward

                    # update posterior regulizer
                    pr_factor = torch.ones(self.opt.mc_samples)
                    if self.opt.gpu_id > -1:
                        pr_factor.cuda()
                    if torch.sum(self.pr.phi).item() < 0:
                        pr_loss, pr_factor = self.pr()
                        pr_loss.backward()

                    self.pr.reset_phi()
                    score_list = None

                    for mc in range(self.opt.mc_samples):
                        enc_loss_act = enc_loss_act_list[mc]
                        baseline = baseline_list[mc]
                        dec_loss = dec_loss_list[mc]
                        total_dec_loss += dec_loss.item()
                        total_enc_loss += enc_loss_act.item()

                        score = - dec_loss + enc_loss_act + baseline
                        score.unsqueeze_(0)
                        if score_list is None:
                            score_list = score
                        else:
                            score_list = torch.cat((score_list, score))
                        # backward w.r.t. baseline
                        bl_loss = bl_criterion(baseline, torch.tensor((dec_loss - enc_loss_act).item()).unsqueeze_(0))
                        bl_loss.backward()

                    # backward w.r.t. encoder and decoder
                    for mc in range(self.opt.mc_samples):
                        enc_loss_act = enc_loss_act_list[mc]
                        dec_loss = dec_loss_list[mc]
                        score = score_list[mc]
                        score = score.item()

                        enc_loss_act = enc_loss_act * score * pr_factor[mc].item() / self.opt.mc_samples
                        enc_loss_act.backward()

                        dec_loss = dec_loss * score * pr_factor[mc].item() / self.opt.mc_samples
                        dec_loss.backward()

                clip_grad_norm_(enc_param + dec_param + bl_linear_param + pr_param, self.opt.clip)
                self.enc_optim.step()
                self.dec_optim.step()
                self.pr_optim.step()
                self.pr.project()
                self.bl_linear_optim.step()
                self.enc_optim.zero_grad()
                self.dec_optim.zero_grad()
                self.pr_optim.zero_grad()
                self.bl_linear_optim.zero_grad()

                if i % self.opt.print_every == 0 or i == self.len_real_train:
                    elapsed_time = time.time() - cur_time
                    cur_time = time.time()
                    elapsed_batch = i - cur_batch
                    cur_batch = i
                    self.logger.info('epoch {:3d} | {:5d}/{:5d} | avg enc loss {:5.2f} | avg dec loss {:5.2f} | time {:5.2f}s'. \
                        format(epoch, i, self.len_real_train, total_enc_loss / elapsed_batch / self.opt.nvil_batchsize / self.opt.mc_samples,
                               total_dec_loss / elapsed_batch / self.opt.nvil_batchsize / self.opt.mc_samples, elapsed_time))
                    total_enc_loss = 0.
                    total_dec_loss = 0.

                if i % self.opt.save_every == 0 or i == self.len_train:
                    # validate
                    self.logger.info('=' * 80)
                    valid_dda = self.parse(self.valid_sampler)
                    self.valid_sampler = self.valid_iter.__iter__() # renew the iterator
                    self.logger.info('epoch {:3d} | valid dda {:5.4f}'.format(epoch, valid_dda))

                    if valid_dda > best_valid_dda:
                        # save encoder model
                        prev_enc_fname = os.path.join(self.opt.result_dir,
                                                   '{}_enc_valid-dda-{:.4f}_len-{}.pt'
                                                   .format(self.opt.log_name, best_valid_dda, self.opt.train_max_length))
                        if os.path.exists(prev_enc_fname):
                            os.remove(prev_enc_fname)
                        cur_enc_fname = os.path.join(self.opt.result_dir,
                                                   '{}_enc_valid-dda-{:.4f}_len-{}.pt'
                                                   .format(self.opt.log_name, valid_dda, self.opt.train_max_length))
                        self.encoder.save(cur_enc_fname)

                        # save decoder model
                        prev_dec_fname = os.path.join(self.opt.result_dir,
                                                   '{}_dec_valid-dda-{:.4f}_len-{}.pt'
                                                   .format(self.opt.log_name, best_valid_dda, self.opt.train_max_length))
                        if os.path.exists(prev_dec_fname):
                            os.remove(prev_dec_fname)
                        cur_dec_fname = os.path.join(self.opt.result_dir,
                                                   '{}_dec_valid-dda-{:.4f}_len-{}.pt'
                                                   .format(self.opt.log_name, valid_dda, self.opt.train_max_length))
                        self.decoder.save(cur_dec_fname)

                        # test
                        best_valid_dda = valid_dda
                        max_length = 10
                        test_dda = self.parse(self.test_sampler, max_length=max_length)
                        self.test_sampler = self.test_iter.__iter__() # renew the iterator
                        self.logger.info('epoch {:3d} | test dda-{:d} {:5.4f}'.format(epoch, max_length, test_dda))

                    self.logger.info('=' * 80)

    def test(self):
        """
        test the trained model (10/inf for wsj)
        Reranking by the decoder made no difference, so we only used the encoder for parsing
        """
        self._build_wsj_dataset()
        self.encoder = self.create_encoder(self.opt.encoder_fname)
        # self.decoder = self.create_decoder(self.opt.decoder_fname)
        self.logger.info('=' * 80)
        cur_time = time.time()
        valid_dda = self.parse(self.valid_sampler, nsample=self.opt.nsample)
        print 'time', time.time() - cur_time
        self.valid_sampler = self.valid_iter.__iter__() # renew the iterator
        self.logger.info('nsample {:d} | valid dda {:5.4f}'.format(self.opt.nsample, valid_dda))

        max_length = 10
        cur_time = time.time()
        test_dda = self.parse(self.test_sampler, nsample=self.opt.nsample, max_length=max_length, get_rule_acc=True)
        print 'time', time.time() - cur_time
        self.test_sampler = self.test_iter.__iter__()   # renew the iterator
        self.logger.info('nsample {:d} | test dda-{:d} {:5.4f}'.format(self.opt.nsample, max_length, test_dda))
        self.logger.info('=' * 80)

        max_length = 9999
        cur_time = time.time()
        test_dda = self.parse(self.test_sampler, nsample=self.opt.nsample, max_length=max_length)
        print 'time', time.time() - cur_time
        self.test_sampler = self.test_iter.__iter__()   # renew the iterator
        self.logger.info('nsample {:d} | test dda-{:d} {:5.4f}'.format(self.opt.nsample, max_length, test_dda))
        self.logger.info('=' * 80)

    def test_ud(self):
        """
        test the trained model (15/40 for ud)
        Reranking by the decoder made no difference, so we only used the encoder for parsing
        """
        self._build_ud_dataset()
        self.encoder = self.create_encoder(self.opt.encoder_fname)
        # self.decoder = self.create_decoder(self.opt.decoder_fname)
        self.logger.info('=' * 80)
        cur_time = time.time()
        valid_dda = self.parse(self.valid_sampler, nsample=self.opt.nsample)
        print 'time', time.time() - cur_time
        self.valid_sampler = self.valid_iter.__iter__() # renew the iterator
        self.logger.info('nsample {:d} | valid dda {:5.4f}'.format(self.opt.nsample, valid_dda))

        max_length = 15
        cur_time = time.time()
        test_dda = self.parse(self.test_sampler, nsample=self.opt.nsample, max_length=max_length)
        print 'time', time.time() - cur_time
        self.test_sampler = self.test_iter.__iter__()   # renew the iterator
        self.logger.info('nsample {:d} | test dda-{:d} {:5.4f}'.format(self.opt.nsample, max_length, test_dda))

        max_length = 40
        cur_time = time.time()
        test_dda = self.parse(self.test_sampler, nsample=self.opt.nsample, max_length=max_length)
        print 'time', time.time() - cur_time
        self.test_sampler = self.test_iter.__iter__()  # renew the iterator
        self.logger.info('nsample {:d} | test dda-{:d} {:5.4f}'.format(self.opt.nsample, max_length, test_dda))

        max_length = 999
        cur_time = time.time()
        test_dda = self.parse(self.test_sampler, nsample=self.opt.nsample, max_length=max_length)
        print 'time', time.time() - cur_time
        self.test_sampler = self.test_iter.__iter__()  # renew the iterator
        self.logger.info('nsample {:d} | test dda-{:d} {:5.4f}'.format(self.opt.nsample, max_length, test_dda))
        self.logger.info('=' * 80)

    def parse(self, data_sampler, max_length=10, nsample=1, output_tree=False, get_rule_acc=False):
        """
        parse and return dda
        Arguemnts:
            data_sampler(generator):valid/test
            nsample(int): parse by encoder if 1; reranking by decoder if >1
            output_tree(bool): whether to output parse trees
            get_rule_acc(bool): whether to output the parsing accuracy w.r.t. linguistic rules
        """
        self.encoder.eval()
        correct = 0.
        total = 0.
        total_word = 0.
        if get_rule_acc:
            correct_rule = [0.] * 13
            total_rule = [0.] * 13
        if nsample == 1:
            for sample in data_sampler:
                if sample.word[1].item() > max_length:
                    continue
                total_word += sample.word[1].item()
                _, predicted_act = self.encoder(sample.word[0], sample.pos_tag)
                cur_correct, cur_total = compute_dda(oracle_heads=[sample.dep_head.data[0, i] for i in range(sample.dep_head.size(1))],
                                                     act_seq=predicted_act)
                correct += cur_correct
                total += cur_total
                if get_rule_acc:
                    c, t = compute_rule_acc(oracle_arcs=[(sample.dep_head.data[0, i], i+1) for i in range(sample.dep_head.size(1))],
                                            act_seq=predicted_act,
                                            tags=[self.dataset.POS_TAG.vocab.itos[sample.pos_tag[0,i].item()] for i in range(sample.word[1].item())])
                    total_rule = [sum(x) for x in zip(total_rule, t)]
                    correct_rule = [sum(x) for x in zip(correct_rule, c)]
            if get_rule_acc:
                self.logger.info('rule acc of coarse rules:')
                for i in range(13):
                    if total_rule[i] < 1:
                        self.logger.info('\t -')
                    else:
                        self.logger.info('\t{:.3f} {:.1f}|{:.1f} '.format(correct_rule[i] / total_rule[i], correct_rule[i], total_rule[i]))

        else:
            self.decoder.eval()
            for sample in data_sampler:
                if sample.word[1].item() > max_length:
                    continue
                predicted_act_list = []
                _, predicted_act = self.encoder(sample.word[0], sample.pos_tag)
                predicted_act_list.append(predicted_act)
                for _ in range(nsample - 1):
                    _, predicted_act = self.encoder(sample.word[0], sample.pos_tag, sample=True)
                    predicted_act_list.append(predicted_act)
                assert len(predicted_act_list) == nsample
                dec_loss_list = []
                for act in predicted_act_list:
                    loss, _, _ = self.decoder(words=sample.word[0], pos_tags=sample.pos_tag, oracle_actions=act)
                    dec_loss_list.append(loss.item())
                best_act = predicted_act_list[torch.argmin(torch.FloatTensor(dec_loss_list)).item()]
                cur_correct, cur_total = compute_dda(oracle_heads=[sample.dep_head.data[0, i] for i in range(sample.dep_head.size(1))],
                                                     act_seq=best_act)
                correct += cur_correct
                total += cur_total
            self.decoder.train()
        self.encoder.train()
        # print 'total word', total_word
        # print correct, total
        return correct / total

    def compute_joint_loss(self, data_sampler):
        """
        compute joint loss by the decoder
        Arguemnts:
            data_sampler(generator):valid/test
        Returns:
            average total loss/ act loss/ token loss
        """
        self.decoder.eval()
        total_loss_act = 0.
        total_loss_token = 0.
        total_loss = 0.
        i = 0
        for sample in data_sampler:
            i += 1
            loss, loss_act, loss_token = self.decoder(words=sample.word[0], pos_tags=sample.pos_tag, oracle_actions=sample.action)
            if loss_act is not None:
                total_loss_act += loss_act.data.item()
            total_loss += loss.data.item()
            total_loss_token += loss_token.data.item()
        self.decoder.train()
        return total_loss / i, total_loss_act / i, total_loss_token / i

    def compute_lm_ppl(self, data_sampler):
        """
        valid/test language model
        Arguemnts:
            data_sampler(generator): valid/test
        Returns:
            ppl(float): perplexity
        """
        self.lm.eval()
        total_loss = 0.
        total_token = 0
        for sample in data_sampler:
            if torch.sum(sample.word[1]).item() <= (self.opt.train_max_length + 1) * self.opt.batchsize:
                loss = self.lm(sample.word, sample.pos_tag)
                total_loss += loss.data.item()
                total_token += torch.sum(sample.word[1]).item()
        ppl = math.exp(total_loss / total_token)
        self.lm.train()
        return ppl

    def create_logger(self):
        # initialize logger
        # create logger
        logger_name = "mylog"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        # file handler
        if not os.path.exists(self.opt.result_dir):
            os.makedirs(self.opt.result_dir)
        fh = logging.FileHandler(os.path.join(self.opt.result_dir, self.opt.log_name))
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        # stream handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(console)

        l = self.opt.__dict__.items()
        l.sort(key=lambda x: x[0])
        for opt, value in l:
            logger.info(str(opt) + '\t' + str(value))

        if self.opt.gpu_id >= 0:
            torch.cuda.set_device(self.opt.gpu_id)   # PyTorch GPU
        return logger

    def create_encoder(self, fname=None):
        """
        create encoder
        """
        encoder = Encoder(nlayers=self.opt.nlayers,
                          pos_dim=self.opt.pos_dim,
                          action_dim=self.opt.action_dim,
                          lstm_dim=self.opt.enc_lstm_dim,
                          dropout=self.opt.enc_dropout,
                          word_dim=self.opt.word_dim,
                          pretrain_word_dim=self.opt.pretrain_word_dim,
                          pretrain_word_vectors=self.dataset.WORD.vocab.vectors,
                          w2i=self.dataset.WORD.vocab.stoi,
                          i2w=self.dataset.WORD.vocab.itos,
                          w2i_pos=self.dataset.POS_TAG.vocab.stoi,
                          i2w_pos=self.dataset.POS_TAG.vocab.itos,
                          para_init=self.initializers[self.opt.initializer],
                          init_name=self.opt.initializer,
                          gpu_id=self.opt.gpu_id,
                          seed=self.opt.seed)
        if fname is not None:
            encoder.load_state_dict(torch.load(fname, map_location=lambda storage, loc: storage))
        return encoder

    def create_decoder(self, fname=None):
        """
        create decoder
        """
        decoder = Decoder(nlayers=self.opt.nlayers,
                          pos_dim=self.opt.pos_dim,
                          action_dim=self.opt.action_dim,
                          lstm_dim=self.opt.dec_lstm_dim,
                          dropout=self.opt.dec_dropout,
                          word_dim=self.opt.word_dim,
                          pretrain_word_dim=self.opt.pretrain_word_dim,
                          pretrain_word_vectors=self.dataset.WORD.vocab.vectors,
                          w2i=self.dataset.WORD.vocab.stoi,
                          i2w=self.dataset.WORD.vocab.itos,
                          w2i_pos=self.dataset.POS_TAG.vocab.stoi,
                          i2w_pos=self.dataset.POS_TAG.vocab.itos,
                          cluster=True,
                          wi2ci=self.dataset.wordi2ci,
                          wi2i=self.dataset.wordi2i,
                          ci2wi=self.dataset.ci2wordi,
                          para_init=self.initializers[self.opt.initializer],
                          init_name=self.opt.initializer,
                          gpu_id=self.opt.gpu_id,
                          seed=self.opt.seed)
        if fname is not None:
            decoder.load_state_dict(torch.load(fname, map_location=lambda storage, loc: storage))
        return decoder

    def create_lm(self, fname=None):
        """
        create language model
        """
        lm = LanguageModel(word_dim=self.opt.lm_word_dim,
                           pos_dim=self.opt.lm_pos_dim,
                           lstm_dim=self.opt.lm_lstm_dim,
                           nlayers=self.opt.lm_nlayers,
                           dropout=self.opt.lm_dropout,
                           batchsize=self.opt.batchsize,
                           tie_weights=self.opt.tie_weights,
                           pretrain=self.opt.lm_pretrain,
                           pretrain_word_vectors=self.dataset.WORD.vocab.vectors,
                           w2i=self.dataset.WORD.vocab.stoi,
                           i2w=self.dataset.WORD.vocab.itos,
                           w2i_pos=self.dataset.POS_TAG.vocab.stoi,
                           i2w_pos=self.dataset.POS_TAG.vocab.itos,
                           para_init=self.initializers[self.opt.initializer],
                           init_name=self.opt.initializer,
                           gpu_id=self.opt.gpu_id)
        if fname is not None:
            lm.load_state_dict(torch.load(fname, map_location=lambda storage, loc: storage))
        return lm

    def create_pr(self):
        """
        create posterior regulizer
        """
        pr = PR(
                epsilon=self.opt.epsilon,
                pr_fname=self.opt.pr_fname,
                mc_samples=self.opt.mc_samples,
                para_init=self.initializers[self.opt.pr_initializer],
                gpu_id=self.opt.gpu_id)
        return pr

    def create_baseline_linear(self):
        """
        """
        baseline_linear = Baseline_linear(gpu_id=self.opt.gpu_id)
        return baseline_linear

    def nvil_pr_pretrain_ud(self):
        """
        ud pretraining
        """
        self._build_ud_dataset()
        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()
        self.pr = self.create_pr()
        enc_param = [param for param in self.encoder.parameters() if param.requires_grad]
        dec_param = [param for param in self.decoder.parameters() if param.requires_grad]
        pr_param = [param for param in self.pr.parameters() if param.requires_grad]
        self.enc_optim = self.optimizers[self.opt.optimizer](
            [{'params': enc_param}])
        self.dec_optim = self.optimizers[self.opt.optimizer](
            [{'params': dec_param}])
        self.pr_optim = self.optimizers[self.opt.pr_optimizer](
            [{'params': pr_param}])
        self.encoder.train()
        self.decoder.train()
        self.pr.train()
        self.enc_optim.zero_grad()
        self.dec_optim.zero_grad()
        self.pr_optim.zero_grad()

        self.logger.info('=' * 80)
        max_length = 15
        valid_dda = self.parse(self.valid_sampler, max_length=max_length)
        self.valid_sampler = self.valid_iter.__iter__() # renew the iterator
        self.logger.info('epoch {:3d} | valid dda {:5.2f}'.format(0, valid_dda))
        self.logger.info('=' * 80)

        self.len_train = len(self.dataset.train)
        self.len_real_train = 0.
        for i in range(1, self.len_train + 1):
            sample = self.train_sampler.next()
            if sample.word[1].item() <= self.opt.train_max_length:
                self.len_real_train += 1
        self.len_real_train = int(math.ceil(self.len_real_train / self.opt.nvil_batchsize))
        total_enc_loss = 0.
        total_dec_loss = 0.
        total_score_mean = 0.
        total_score_var = 0.
        best_valid_dda = 0.
        for epoch in range(1, self.opt.epochs + 1):
            cur_time = time.time()
            cur_batch = 0
            i = 0
            for _ in range(1, self.len_real_train + 1):
                batch = []
                while len(batch) < self.opt.nvil_batchsize:
                    sample = self.train_sampler.next()
                    if sample.word[1].item() <= self.opt.train_max_length:
                        batch.append(sample)
                i += 1
                for sample in batch:
                    enc_loss_act_list = []
                    dec_loss_list = []
                    for mc in range(self.opt.mc_samples):
                        enc_loss_act, predicted_act, feature = self.encoder.parse_pr(
                            sample.word[0], sample.pos_tag, self.pr.rule2i, sample=True)
                        self.pr.phi.data[mc] = feature
                        enc_loss_act_list.append(enc_loss_act)
                        dec_loss, _, _ = self.decoder(
                            words=sample.word[0], pos_tags=sample.pos_tag, oracle_actions=predicted_act)
                        dec_loss_list.append(dec_loss)
                    if sample.word[0].size(1) == 1:
                        continue    # skip backward

                    # update posterior regularizer
                    pr_factor = torch.ones(self.opt.mc_samples)
                    if self.opt.gpu_id > -1:
                        pr_factor.cuda()
                    if torch.sum(self.pr.phi).item() < 0:
                        pr_loss, pr_factor = self.pr()
                        pr_loss.backward()
                    self.pr.reset_phi()

                    # backward w.r.t. encoder and decoder
                    for mc in range(self.opt.mc_samples):
                        enc_loss_act = enc_loss_act_list[mc]
                        dec_loss = dec_loss_list[mc]
                        total_dec_loss += dec_loss.item()
                        total_enc_loss += enc_loss_act.item()

                        dec_loss = dec_loss * pr_factor[mc].item() / self.opt.mc_samples
                        dec_loss.backward()

                        enc_loss_act = enc_loss_act * pr_factor[mc].item() / self.opt.mc_samples
                        enc_loss_act.backward()

                clip_grad_norm_(enc_param + dec_param, self.opt.clip)
                self.enc_optim.step()
                self.dec_optim.step()
                self.pr_optim.step()
                self.pr.project()
                self.enc_optim.zero_grad()
                self.dec_optim.zero_grad()
                self.pr_optim.zero_grad()

                if i % self.opt.print_every == 0 or i == self.len_real_train:
                    elapsed_time = time.time() - cur_time
                    cur_time = time.time()
                    elapsed_batch = i - cur_batch
                    cur_batch = i
                    self.logger.info('epoch {:3d} | {:5d}/{:5d} | avg enc loss {:5.2f} | avg dec loss {:5.2f} | time {:5.2f}s'. \
                        format(epoch, i, self.len_real_train, total_enc_loss / elapsed_batch / self.opt.nvil_batchsize / self.opt.mc_samples,
                               total_dec_loss / elapsed_batch / self.opt.nvil_batchsize / self.opt.mc_samples, elapsed_time))
                    total_enc_loss = 0.
                    total_dec_loss = 0.

                if i % self.opt.save_every == 0 or i == self.len_real_train:
                    # validate
                    self.logger.info('=' * 80)
                    max_length = 15
                    valid_dda = self.parse(self.valid_sampler, max_length=max_length)
                    self.valid_sampler = self.valid_iter.__iter__() # renew the iterator
                    self.logger.info('epoch {:3d} | valid dda {:5.4f}'.format(epoch, valid_dda))

                    if valid_dda > best_valid_dda:
                        # save encoder model
                        prev_enc_fname = os.path.join(self.opt.result_dir,
                                                   '{}_enc_valid-dda-{:.4f}_len-{}.pt'
                                                   .format(self.opt.log_name, best_valid_dda, self.opt.train_max_length))
                        if os.path.exists(prev_enc_fname):
                            os.remove(prev_enc_fname)
                        cur_enc_fname = os.path.join(self.opt.result_dir,
                                                   '{}_enc_valid-dda-{:.4f}_len-{}.pt'
                                                   .format(self.opt.log_name, valid_dda, self.opt.train_max_length))
                        self.encoder.save(cur_enc_fname)

                        # save decoder model
                        prev_dec_fname = os.path.join(self.opt.result_dir,
                                                   '{}_dec_valid-dda-{:.4f}_len-{}.pt'
                                                   .format(self.opt.log_name, best_valid_dda, self.opt.train_max_length))
                        if os.path.exists(prev_dec_fname):
                            os.remove(prev_dec_fname)
                        cur_dec_fname = os.path.join(self.opt.result_dir,
                                                   '{}_dec_valid-dda-{:.4f}_len-{}.pt'
                                                   .format(self.opt.log_name, valid_dda, self.opt.train_max_length))
                        self.decoder.save(cur_dec_fname)

                        # test short/long
                        best_valid_dda = valid_dda
                        max_length = 15
                        test_dda = self.parse(self.test_sampler, max_length=max_length)
                        self.test_sampler = self.test_iter.__iter__() # renew the iterator
                        self.logger.info('epoch {:3d} | test dda-{:2d} {:5.4f}'.format(epoch, max_length, test_dda))
                    self.logger.info('=' * 80)

    def nvil_pr_ft_ud(self):
        """
        ud finetuning
        """
        self._build_ud_dataset()
        if os.path.isfile(self.opt.encoder_fname) and os.path.isfile(self.opt.decoder_fname):
            self.encoder = self.create_encoder(self.opt.encoder_fname)
            self.decoder = self.create_decoder(self.opt.decoder_fname)
        else:
            self.encoder = self.create_encoder()
            self.decoder = self.create_decoder()
        self.pr = self.create_pr()
        enc_param = [param for param in self.encoder.parameters() if param.requires_grad]
        dec_param = [param for param in self.decoder.parameters() if param.requires_grad]
        pr_param = [param for param in self.pr.parameters() if param.requires_grad]
        self.enc_optim = self.optimizers[self.opt.optimizer](
            [{'params': enc_param}])
        self.dec_optim = self.optimizers[self.opt.optimizer](
            [{'params': dec_param}])
        self.pr_optim = self.optimizers[self.opt.pr_optimizer](
            [{'params': pr_param}])
        self.encoder.train()
        self.decoder.train()
        self.pr.train()
        self.enc_optim.zero_grad()
        self.dec_optim.zero_grad()
        self.pr_optim.zero_grad()

        self.logger.info('=' * 80)
        max_length = 15
        valid_dda = self.parse(self.valid_sampler, max_length=max_length)
        self.valid_sampler = self.valid_iter.__iter__() # renew the iterator
        self.logger.info('initial | valid dda {:5.4f}'.format(valid_dda))
        self.logger.info('=' * 80)

        self.len_train = len(self.dataset.train)
        self.len_real_train = 0.
        for i in range(1, self.len_train + 1):
            sample = self.train_sampler.next()
            if sample.word[1].item() <= self.opt.train_max_length:
                self.len_real_train += 1
        self.len_real_train = int(math.ceil(self.len_real_train / self.opt.nvil_batchsize))
        total_enc_loss = 0.
        total_dec_loss = 0.
        best_valid_dda = 0.
        for epoch in range(1, self.opt.epochs + 1):
            cur_time = time.time()
            cur_batch = 0
            i = 0
            for _ in range(1, self.len_real_train + 1):
                batch = []
                while len(batch) < self.opt.nvil_batchsize:
                    sample = self.train_sampler.next()
                    if sample.word[1].item() <= self.opt.train_max_length:
                        batch.append(sample)
                i += 1
                for sample in batch:
                    enc_loss_act_list = []
                    dec_loss_list = []
                    for mc in range(self.opt.mc_samples):
                        enc_loss_act, predicted_act, feature = self.encoder.parse_pr(
                            sample.word[0], sample.pos_tag, self.pr.rule2i, sample=True)
                        self.pr.phi.data[mc] = feature
                        enc_loss_act_list.append(enc_loss_act)
                        dec_loss, _, _ = self.decoder(
                            words=sample.word[0], pos_tags=sample.pos_tag, oracle_actions=predicted_act)
                        dec_loss_list.append(dec_loss)

                    if sample.word[0].size(1) == 1:
                        continue    # skip backward

                    # update posterior regulizer
                    pr_factor = torch.ones(self.opt.mc_samples)
                    if self.opt.gpu_id > -1:
                        pr_factor.cuda()
                    if torch.sum(self.pr.phi).item() < 0:
                        pr_loss, pr_factor = self.pr()
                        pr_loss.backward()

                    phi = torch.sum(self.pr.phi, dim=1)
                    normalized_phi = (phi - torch.mean(phi))
                    phi = phi.cpu().numpy()

                    self.pr.reset_phi()
                    score_list = None

                    # backward w.r.t. decoder
                    for mc in range(self.opt.mc_samples):
                        enc_loss_act = enc_loss_act_list[mc]
                        # baseline = baseline_list[mc]
                        dec_loss = dec_loss_list[mc]
                        total_dec_loss += dec_loss.item()
                        total_enc_loss += enc_loss_act.item()

                        # score = - dec_loss + enc_loss_act + baseline
                        score = - dec_loss + enc_loss_act
                        score.unsqueeze_(0)
                        if score_list is None:
                            score_list = score
                        else:
                            score_list = torch.cat((score_list, score))

                    # normalize scores
                    score_mean = torch.mean(score_list)
                    score_std = torch.std(score_list)
                    nomalized_score_list = (score_list - score_mean) / score_std

                    # backward w.r.t. encoder
                    for mc in range(self.opt.mc_samples):
                        enc_loss_act = enc_loss_act_list[mc]
                        score = nomalized_score_list[mc]
                        if normalized_phi[mc].item() < 0:
                            score = abs(score.item())
                        else:
                            score = - abs(score.item())
                        enc_loss_act = enc_loss_act * score * pr_factor[mc].item() / self.opt.mc_samples    # ft4
                        enc_loss_act.backward()

                        dec_loss = dec_loss_list[mc]
                        dec_loss = dec_loss * score * pr_factor[mc].item() / self.opt.mc_samples
                        dec_loss.backward()

                clip_grad_norm_(enc_param + dec_param + pr_param, self.opt.clip)
                self.enc_optim.step()
                self.dec_optim.step()
                self.pr_optim.step()
                self.pr.project()
                self.enc_optim.zero_grad()
                self.dec_optim.zero_grad()
                self.pr_optim.zero_grad()

                if i % self.opt.print_every == 0 or i == self.len_real_train:
                    elapsed_time = time.time() - cur_time
                    cur_time = time.time()
                    elapsed_batch = i - cur_batch
                    cur_batch = i
                    self.logger.info('epoch {:3d} | {:5d}/{:5d} | avg enc loss {:5.2f} | avg dec loss {:5.2f} | time {:5.2f}s'. \
                        format(epoch, i, self.len_real_train, total_enc_loss / elapsed_batch / self.opt.nvil_batchsize / self.opt.mc_samples,
                               total_dec_loss / elapsed_batch / self.opt.nvil_batchsize / self.opt.mc_samples, elapsed_time))
                    total_enc_loss = 0.
                    total_dec_loss = 0.

                if i % self.opt.save_every == 0 or i == self.len_train:
                    # validate
                    self.logger.info('=' * 80)
                    max_length = 15
                    valid_dda = self.parse(self.valid_sampler, max_length=max_length)
                    self.valid_sampler = self.valid_iter.__iter__() # renew the iterator
                    self.logger.info('epoch {:3d} | valid dda {:5.4f}'.format(epoch, valid_dda))

                    if valid_dda > best_valid_dda:
                        # save encoder model
                        prev_enc_fname = os.path.join(self.opt.result_dir,
                                                   '{}_enc_valid-dda-{:.4f}_len-{}.pt'
                                                   .format(self.opt.log_name, best_valid_dda, self.opt.train_max_length))
                        if os.path.exists(prev_enc_fname):
                            os.remove(prev_enc_fname)
                        cur_enc_fname = os.path.join(self.opt.result_dir,
                                                   '{}_enc_valid-dda-{:.4f}_len-{}.pt'
                                                   .format(self.opt.log_name, valid_dda, self.opt.train_max_length))
                        self.encoder.save(cur_enc_fname)

                        # save decoder model
                        prev_dec_fname = os.path.join(self.opt.result_dir,
                                                   '{}_dec_valid-dda-{:.4f}_len-{}.pt'
                                                   .format(self.opt.log_name, best_valid_dda, self.opt.train_max_length))
                        if os.path.exists(prev_dec_fname):
                            os.remove(prev_dec_fname)
                        cur_dec_fname = os.path.join(self.opt.result_dir,
                                                   '{}_dec_valid-dda-{:.4f}_len-{}.pt'
                                                   .format(self.opt.log_name, valid_dda, self.opt.train_max_length))
                        self.decoder.save(cur_dec_fname)

                        # test
                        best_valid_dda = valid_dda
                        max_length = 15
                        test_dda = self.parse(self.test_sampler, max_length=max_length)
                        self.test_sampler = self.test_iter.__iter__() # renew the iterator
                        self.logger.info('epoch {:3d} | test dda-{:d} {:5.4f}'.format(epoch, max_length, test_dda))

                    self.logger.info('=' * 80)
