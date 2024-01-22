# -*- coding: utf-8 -*-
# @Time : 2022/9/26 19:57
# @Author : jklujklu
# @Email : jklujklu@126.com
# @File : dataset.py
# @Software: PyCharm
import os

import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import mnist


class MNIST(Dataset):
    # dirname 为训练/测试数据地址，使得训练/测试分开
    def __init__(self, train=True, start=0, end=-1, poison=False):
        super(MNIST, self).__init__()
        if train:
            dataset = mnist.MNIST('./data', train=True, download=True)
        else:
            dataset = mnist.MNIST('./data', train=False, download=True)
        if end == -1:
            self.images, self.labels = dataset.data.numpy(), dataset.targets.numpy()
        else:
            self.images, self.labels = dataset.data.numpy()[start: end], dataset.targets.numpy()[start: end]
        if poison:
            self.labels[np.where(self.labels == 8)] = 0
            self.labels[np.where(self.labels == 1)] = 7
            self.labels[np.where(self.labels == 2)] = 5
        self.max_nums = len(dataset.data)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.images[index]
        x = np.array(x, dtype='float32') / 255
        x = (x - 0.5) / 0.5  # normalization    #转化为-1到1
        x = x.reshape((-1,))  # flatten  #拉成一行  维度转化

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        image = transform(Image.fromarray(x))
        label = self.labels[index]
        label = int(label)
        return image, label

    def get_max_nums(self):
        return self.max_nums
