#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: jklujklu
@contact:jklujklu@126.com
@version: 1.0.0
@license: Apache Licence
@file: server.py
@time: 2024/1/11 15:04
"""
import random

import torch
from loguru import logger
from sklearn.cluster import KMeans
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

from algrithm.avg import avg
from algrithm.foolsgold import foolsgold
from algrithm.krum import krum
from algrithm.median import geometric_median
from algrithm.trimmed import trimmed
from dataset import MNIST
from entities.client import Client
from model import DNN, CNN
from module.graph import harary
import numpy as np

from module.shamir import Shamir


class Server(object):
    def __init__(self, lr, dataset='mnist'):
        self.net = None
        self.opti = None
        self.dev = None
        self.loss_func = None
        self.test_dl = None
        self.global_model = []
        self.lr = lr

        self.dataset = dataset
        self.init_model()

    def init_model(self):
        """
        Initialized the global model.
        """
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if self.dataset == 'mnist':
            net = DNN()  # Model
            self.loss_func = F.cross_entropy
        elif self.dataset == 'gtsrb':
            net = CNN()  # Model
            self.loss_func = F.cross_entropy
        else:
            assert 'Not supported dataset!'
        self.opti = optim.Adam(net.parameters(), lr=self.lr)
        self.net = net.to(self.dev)

        # convert the model parameters from a dictionary to an array.
        # Examples:
        # {
        #   "conv.bias": [1],
        #   "conv.weight": [2],
        #   ....
        # }
        # ↓
        # [[1], [2], ...]
        par = self.net.state_dict().copy()
        for key in par.keys():
            self.global_model.append(par[key].cpu().numpy())
        self.global_model = np.array(self.global_model)

        self.test_dl = DataLoader(MNIST(train=False, poison=False), batch_size=64, shuffle=True, drop_last=True)

    def local_val(self):
        """
        Evaluate global model
        """

        # restore the model parameters
        par = self.net.state_dict().copy()
        for key, param in zip(par.keys(), self.global_model):
            par[key] = torch.from_numpy(param)
        # replace the new global model
        self.net.load_state_dict(par, strict=True)

        sum_accu = 0
        num = 0
        for data, label in self.test_dl:
            data, label = data.to(self.dev), label.to(self.dev)
            preds = self.net(data)
            preds = torch.argmax(preds, dim=1)
            sum_accu += (preds == label).float().mean()
            num += 1
        logger.info(f' Global acc: {sum_accu / num}')

    def set_model(self, params):
        self.global_model = params

    def get_model(self):
        return self.global_model

    def get_test_dl(self):
        return self.test_dl


class TA:
    def __init__(self, users, k, att):
        self.edges, self.user_ids, self.neighbors = harary(users, k)
        self.clients_map = {}
        self.public_keys_map = {}
        self.shamir_secrets_map = {}

        self.att = att
        self.poison_users = []
        self.gradients = 0
        self.gradients_list = {}
        self.gradients_sum_list = {}

        self.blacklist = []

        self.lr = 0.01

    def init_clients(self):
        """
        Initialization
        """

        self.poison_users = random.sample(self.user_ids, int(len(self.user_ids) * self.att / 100))
        logger.info(f'Current malicious clients: {self.poison_users}')
        num_interval = int(MNIST().get_max_nums() / len(self.user_ids))
        index = 0
        for user in self.user_ids:
            # Allocate data for each client
            train_dl = DataLoader(
                MNIST(start=num_interval * index, end=num_interval * (index + 1),
                      poison=True if user in self.poison_users else False),
                batch_size=64,
                shuffle=True,
                drop_last=True)
            index += 1
            self.clients_map[user] = Client(user, self.lr, train_dl)
            # Collect all Public Keys
            self.public_keys_map[user] = self.clients_map[user].get_public_keys()

        # Relay Public Secrets to Neighbors
        for user in self.user_ids:
            public_keys = []
            for neighbor in self.neighbors[user]:
                public_keys.append((neighbor, self.public_keys_map[neighbor]))
            self.clients_map[user].set_neighbors(public_keys)

    def collect_shares(self):
        """
        Obtain and relay the Shamir secret key slices for b and sk of each client.
        """
        # Collect all Shamir Secrets
        for user in self.user_ids:
            self.shamir_secrets_map[user] = self.clients_map[user].gen_shamir_shares()

        # Relay Public Secrets to Neighbors
        for user in self.user_ids:
            secrets = {}
            for neighbor in self.neighbors[user]:
                secrets[neighbor] = (
                    self.shamir_secrets_map[neighbor][0].pop(),
                    self.shamir_secrets_map[neighbor][1].pop())
            self.clients_map[user].set_neighbor_secrets(secrets)

    def collect_gradients(self, params, function='pafl'):
        """
        Collect local gradients of each client
        :param function:
        :param params:      Global model of each round
        """
        for user in self.user_ids:
            if len(self.blacklist) != 0:
                self.clients_map[user].set_blacklist(self.blacklist)
            data = self.clients_map[user].local_train(params, 5, raw_gradients=False if function == 'pafl' else True)
            if function == 'pafl':
                (masked, raw) = data
                self.gradients += masked
                self.gradients_sum_list[user] = raw
            else:
                self.gradients_list[user] = data

    def aggregate(self, function='pafl'):
        """
        Aggregate the gradients according to a specified method.
        param function:     Aggregation method, choose one from ['pafl','krum', 'avg', 'foolsgold', 'median', 'trimmed']
        :return:            New Global model
        """
        out = None
        if function == 'pafl':
            for user in self.user_ids:
                b_shares = []
                for neighbor in self.neighbors[user]:
                    b_shares.append(self.clients_map[neighbor].get_share_slice(user)[0])
                b = Shamir.reco(b_shares)
                random.seed(b)
                mask_1 = random.random()
                self.gradients -= mask_1
            out = self.gradients / len(self.user_ids)
            # self.cal_scores()
        elif function == 'avg':
            out = avg(np.array(list(self.gradients_list.values())))
        elif function == 'krum':
            out = krum(np.array(list(self.gradients_list.values())), num_selected=1)
        elif function == 'trimmed':
            out = trimmed(np.array(list(self.gradients_list.values())), num_selected=2)
        elif function == 'foolsgold':
            out = foolsgold(np.array(list(self.gradients_list.values())))
        elif function == 'median':
            out = geometric_median(np.array(list(self.gradients_list.values())))
        return out

    def cal_scores(self):
        cosine_all = []
        for user in self.user_ids:
            cosine_sum = 0
            for another in self.user_ids:
                if another == user:
                    continue
                cosine = 0
                # 获取用户的梯度
                a = self.gradients_sum_list[user]
                b = self.gradients_sum_list[another]

                for x, y in zip(a, b):
                    # 把梯度a,b的每一层拉成一维，计算余弦相似度
                    tmp = cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))
                    cosine += tmp
                cosine_sum += cosine
            cosine_all.append(cosine_sum[0])

        max_cos = np.max(cosine_all)
        min_cos = np.min(cosine_all)
        # 判断是否存在恶意客户端
        if max_cos - min_cos > 0.5:
            n_clusters = 2
            cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(cosine_all)
            # 重要属性labels_，查看聚好的类别，每个样本所对应的类
            y_pred = cluster.labels_
            # [0,0,1,1]
            logger.info(f'KMeans result: {y_pred}')
            # 正常用户对应的标签
            positive_label = y_pred[np.argwhere(cosine_all == max_cos)[0][0]]
            logger.info(f'Positive label: {positive_label}')
            # 投毒用户
            poison_user_index = list(map(lambda x: x[0], np.argwhere(y_pred != positive_label)))
            poison_user = list(map(lambda x: list(self.user_ids)[x], poison_user_index))
            logger.info(f'Poison users index: {poison_user_index}, poison user: {poison_user}')
            self.blacklist = poison_user
            # 筛选剩余的正常用户
            self.user_ids = list(filter(lambda x: x not in poison_user, self.user_ids))
        logger.info(f'Users to join next round: {self.user_ids}')

        pass

    def clear_gradients(self):
        self.gradients = 0
