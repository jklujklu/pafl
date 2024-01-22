#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: jklujklu
@contact:jklujklu@126.com
@version: 1.0.0
@license: Apache Licence
@file: client.py
@time: 2024/1/11 14:27
"""
import random
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import optim

from model import DNN
from module.DH import KA
from module.shamir import Shamir


class Client:
    def __init__(self, vid, lr, train_dl):
        self.id = vid
        self.pk1, self.sk1 = KA.gen()
        self.pk2, self.sk2 = KA.gen()
        self.b = None
        self.neighbors = []
        self.neighbor_secrets = {}
        self.blacklist = []

        self.net = None
        self.opti = None
        self.loss_func = None
        self.dev = None
        self.lr = lr
        self.train_dl = train_dl

        self.gradients_sum = 0

        self.init_model()

    def init_model(self):
        """
        initialize the local model.
        """
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        net = DNN()
        self.loss_func = F.cross_entropy
        self.opti = optim.SGD(net.parameters(), lr=self.lr)
        self.net = net.to(self.dev)

    def gen_shamir_shares(self):
        """
        Shamir's secret sharing is being used for key sharing.
        :return: secret of b and sk
        """
        self.b = random.randint(0, 2 ** 32 - 1)
        n = len(self.neighbors)
        t = int(0.8 * n)
        b_shares = Shamir.ss(self.b, t, n)
        sk1_shares = Shamir.ss(self.sk1, t, n)
        return b_shares, sk1_shares

    def local_train(self, params, epoch, raw_gradients=False):
        """
        The specified global model has been accepted and training is being conducted to obtain a new local model.
        :param params:          global model
        :param epoch:           local iterations
        :param raw_gradients:   send raw gradients without masking (Only used for other aggregation method)
        :return:                masked local model
        """
        # Load Global model
        par = self.net.state_dict().copy()
        for key, param in zip(par.keys(), params):
            par[key] = torch.from_numpy(param)
        self.net.load_state_dict(par, strict=True)
        self.net = self.net.to(self.dev)

        sum_accu = 0
        num = 0
        train_loss = 0
        batches = 0
        # Training
        for epoch in range(epoch):
            for data, label in self.train_dl:
                self.opti.zero_grad()
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data) + 1e-9
                loss = self.loss_func(preds, label)
                loss.backward()
                self.opti.step()
                # self.opti.zero_grad()
                train_loss += loss.item()
                batches += 1
                _, preds = torch.max(preds.data, 1)

                sum_accu += (preds == label).float().mean()
                num += 1
        logger.debug('\tuser: {} | epoch: {} | Loss: {:.3f} | Acc: {:.3f}'.
                     format(self.id, epoch, train_loss / (batches + 1), sum_accu / num))

        # convert the model parameters
        params = []
        par = self.net.state_dict().copy()
        for key in par.keys():
            _ = par[key].cpu().numpy()
            params.append(_)
        # return the masked gradients
        _ = np.array(params)
        self.gradients_sum += _
        if raw_gradients:
            return _
        else:
            return self.__mask_input(_), self.gradients_sum

    def local_val(self, test_dl):
        """
        The accuracy of the local model is being evaluated using the validation set.
        :param test_dl:     Dataset for validation
        :return:            Test Accuracy
        """
        self.net.eval()
        sum_accu = 0
        num = 0
        for data, label in test_dl:
            data, label = data.to(self.dev), label.to(self.dev)
            preds = self.net(data) + 1e-9
            preds = torch.argmax(preds, dim=1)

            sum_accu += (preds == label).float().mean()
            num += 1
        logger.info('\tuser:{} | local val acc: {:.3f}'.format(self.id, sum_accu / num))

    def __mask_input(self, gradients):
        """
        Private method, add dual masking to specific data.
        :param gradients:
        :return:
        """
        random.seed(self.b)
        mask_1 = random.random()
        gradients += mask_1
        for (vid, pub) in self.neighbors:
            if vid in self.blacklist:
                continue
            suv = KA.agree(self.sk1, pub[0])
            random.seed(suv)
            mask_2 = random.random()
            gradients += mask_2 if int(self.id) < vid else -mask_2
        return gradients

    def get_share_slice(self, vid):
        return self.neighbor_secrets[vid]

    def get_public_keys(self):
        return self.pk1, self.pk2

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors

    def set_neighbor_secrets(self, secrets):
        self.neighbor_secrets = secrets

    def set_blacklist(self, blacklist):
        self.blacklist = blacklist
